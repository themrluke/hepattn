import atexit
import contextlib
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import get_context, shared_memory
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Literal

import numpy as np
import scipy
import torch
from torch import nn

from hepattn.utils.import_utils import check_import_safe

_POOL_LOCK = Lock()
_THREAD_POOLS: dict[int, ThreadPool] = {}
_PROCESS_POOLS = {}

# Shared-memory buffer reused across matcher calls (parent side), and each worker's handle to it
_SHM_BUFFER: shared_memory.SharedMemory | None = None
_WORKER_SHM: shared_memory.SharedMemory | None = None
# Address of the buffer while it is page-locked for CUDA (overlap_layers), so it can be unlocked before it is freed
_PINNED_ADDRESS: int | None = None
# One thread that waits for each layer's copy to finish and hands its solves to the process pool
_DISPATCHER: ThreadPoolExecutor | None = None


def _get_thread_pool(n_jobs: int) -> ThreadPool:
    with _POOL_LOCK:
        pool = _THREAD_POOLS.get(n_jobs)
        if pool is None:
            pool = ThreadPool(processes=n_jobs)
            _THREAD_POOLS[n_jobs] = pool
        return pool


def _get_process_pool(n_jobs: int):
    """Get persistent multiprocessing pool using spawn method."""
    with _POOL_LOCK:
        pool = _PROCESS_POOLS.get(n_jobs)
        if pool is None:
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=n_jobs)
            _PROCESS_POOLS[n_jobs] = pool
        return pool


@atexit.register
def _close_pools() -> None:
    """Clean up thread and process pools at exit."""
    for pool in list(_THREAD_POOLS.values()):
        try:
            pool.close()
            pool.join()
        except Exception:  # noqa: BLE001, S110
            pass

    global _SHM_BUFFER  # noqa: PLW0603
    if _SHM_BUFFER is not None:
        with contextlib.suppress(Exception):
            _unpin_shm_buffer()
        with contextlib.suppress(Exception):
            _SHM_BUFFER.close()
        with contextlib.suppress(Exception):
            _SHM_BUFFER.unlink()
        _SHM_BUFFER = None

    for pool in list(_PROCESS_POOLS.values()):
        try:
            pool.close()
            pool.join(timeout=1.0)
        except Exception:  # noqa: BLE001,
            try:
                pool.terminate()
                pool.join(timeout=1.0)
            except Exception:  # noqa: BLE001, S110
                pass


def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return col_idx


SOLVERS = {
    "scipy": solve_scipy,
}

# Some compiled extension can cause SIGKILL errors if compiled for the wrong arch
# So we have to check they won't kill everything when we import them
if check_import_safe("lap1015"):
    import lap1015

    def solve_1015_early(cost):
        return lap1015.lap_early(cost)

    def solve_1015_late(cost):
        return lap1015.lap_late(cost)

    SOLVERS["lap1015_late"] = solve_1015_late
    # SOLVERS["lap1015_early"] = lap1015_early
else:
    warnings.warn(
        """Failed to import lap1015 solver. This could be because it is not installed,
    or because it was built targeting a different architecture than supported on the current machine.
    Rebuilding the package on the current machine may fix this.""",
        ImportWarning,
        stacklevel=2,
    )


def match_individual(solver_fn, cost: np.ndarray, default_idx: np.ndarray) -> np.ndarray:
    pred_idx = np.asarray(solver_fn(cost), dtype=np.int32)

    if solver_fn is SOLVERS["scipy"]:
        remaining = np.ones(default_idx.shape[0], dtype=np.bool_)
        remaining[pred_idx] = False
        pred_idx = np.concatenate([pred_idx, default_idx[remaining]])

    return pred_idx


def match_parallel(solver_fn, costs_t: np.ndarray, lengths_np: np.ndarray, pred_dim: int, n_jobs: int = 8) -> torch.Tensor:
    """Thread-based parallel matching across batch."""
    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs
    default_idx = np.arange(pred_dim, dtype=np.int32)

    if n_jobs <= 1 or batch_size <= 1:
        results = [match_individual(solver_fn, costs_t[i][: lengths_np[i]], default_idx) for i in range(batch_size)]
        return torch.from_numpy(np.stack(results, axis=0))

    def _run(i: int) -> np.ndarray:
        return match_individual(solver_fn, costs_t[i][: lengths_np[i]], default_idx)

    pool = _get_thread_pool(n_jobs)
    results = pool.map(_run, range(batch_size), chunksize=chunk_size)
    return torch.from_numpy(np.stack(results, axis=0))


def _mp_match_task(args: tuple[str, str, tuple[int, int, int], str, int, int, int]) -> np.ndarray:
    solver_name, shm_name, shape, dtype_str, i, length, pred_dim = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        costs_t = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        default_idx = np.arange(pred_dim, dtype=np.int32)
        cost = costs_t[i][:length]
        return match_individual(SOLVERS[solver_name], cost, default_idx)
    finally:
        shm.close()


def match_multiprocess(
    solver_name: str,
    costs_t: np.ndarray,
    lengths_np: np.ndarray,
    pred_dim: int,
    n_jobs: int = 8,
) -> torch.Tensor:
    """Multiprocess matching using shared memory to bypass GIL.

    Raises:
        ValueError: If solver_name is not in the available SOLVERS.
    """
    if solver_name not in SOLVERS:
        raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {list(SOLVERS.keys())}")

    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs

    shm = shared_memory.SharedMemory(create=True, size=costs_t.nbytes)
    try:
        shm_arr = np.ndarray(costs_t.shape, dtype=costs_t.dtype, buffer=shm.buf)
        shm_arr[...] = costs_t

        tasks = [(solver_name, shm.name, costs_t.shape, costs_t.dtype.str, i, int(lengths_np[i]), pred_dim) for i in range(batch_size)]

        pool = _get_process_pool(n_jobs)
        results = pool.map(_mp_match_task, tasks, chunksize=chunk_size)
        return torch.from_numpy(np.stack(results, axis=0))
    finally:
        try:
            shm.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()


def _shm_address(shm: shared_memory.SharedMemory) -> int:
    return np.frombuffer(shm.buf, dtype=np.uint8).ctypes.data


def _unpin_shm_buffer() -> None:
    global _PINNED_ADDRESS  # noqa: PLW0603
    if _PINNED_ADDRESS is not None:
        torch.cuda.cudart().cudaHostUnregister(_PINNED_ADDRESS)
        _PINNED_ADDRESS = None


def _pin_shm_buffer(shm: shared_memory.SharedMemory) -> bool:
    """Page-lock the buffer for CUDA, so device-to-host copies into it can run without blocking."""
    global _PINNED_ADDRESS  # noqa: PLW0603
    address = _shm_address(shm)
    with _POOL_LOCK:
        if address == _PINNED_ADDRESS:
            return True
        _unpin_shm_buffer()
        if torch.cuda.cudart().cudaHostRegister(address, shm.size, 0) != torch.cuda.cudart().cudaError.success:
            return False
        _PINNED_ADDRESS = address
        return True


def _get_shm_buffer(nbytes: int) -> shared_memory.SharedMemory:
    """Return the persistent shared-memory buffer, growing it if it is too small."""
    global _SHM_BUFFER  # noqa: PLW0603
    with _POOL_LOCK:
        if _SHM_BUFFER is None or _SHM_BUFFER.size < nbytes:
            if _SHM_BUFFER is not None:
                if _shm_address(_SHM_BUFFER) == _PINNED_ADDRESS:
                    _unpin_shm_buffer()
                _SHM_BUFFER.close()
                with contextlib.suppress(FileNotFoundError):
                    _SHM_BUFFER.unlink()
            _SHM_BUFFER = shared_memory.SharedMemory(create=True, size=nbytes)
        return _SHM_BUFFER


def _mp_match_task_persistent(args: tuple) -> np.ndarray:
    """Like _mp_match_task, but the worker keeps its handle to the buffer between calls.

    args are (solver_name, shm_name, shape, i, length, pred_dim) plus an optional width. Only the first
    `width` query columns are solved, either because the buffer holds only those (shape[2] < pred_dim) or
    because a smaller width is given; the queries beyond them are then appended, in order, after the
    solver's output.
    """
    global _WORKER_SHM  # noqa: PLW0603
    solver_name, shm_name, shape, i, length, pred_dim, *optional_width = args
    if _WORKER_SHM is None or _WORKER_SHM.name != shm_name:
        if _WORKER_SHM is not None:
            _WORKER_SHM.close()
        _WORKER_SHM = shared_memory.SharedMemory(name=shm_name)
    costs_t = np.ndarray(shape, dtype=np.float32, buffer=_WORKER_SHM.buf)
    width = min(optional_width[0], shape[2]) if optional_width else shape[2]
    default_idx = np.arange(width, dtype=np.int32)
    cost = costs_t[i][:length]
    if width < shape[2]:
        # Same numbers as a buffer holding only `width` columns, laid out the same way for the solver
        cost = np.ascontiguousarray(cost[:, :width])
    pred_idx = match_individual(SOLVERS[solver_name], cost, default_idx)
    if width < pred_dim:
        pred_idx = np.concatenate([pred_idx, np.arange(width, pred_dim, dtype=np.int32)])
    return pred_idx


def _valid_query_width_on_device(query_valid_mask: torch.Tensor, lengths: torch.Tensor, pred_dim: int) -> torch.Tensor:
    """_valid_query_width as a tensor on the masks' device, so it can be read later without waiting for the device."""
    counts = query_valid_mask.sum(dim=1)
    positions = torch.arange(pred_dim, device=query_valid_mask.device)
    leading = (query_valid_mask == (positions.unsqueeze(0) < counts.unsqueeze(1))).all()
    needed = torch.maximum(counts.max(), lengths.max() if lengths.numel() else lengths.new_zeros(())).clamp(max=pred_dim)
    return torch.where(leading, needed, torch.full_like(needed, pred_dim)).to(torch.int32)


def _valid_query_width(query_valid_mask: torch.Tensor, lengths_np: np.ndarray, pred_dim: int) -> int:
    """Number of leading query columns the solver needs, or pred_dim if the queries can't be trimmed.

    Trimming is only possible when every row's valid queries come first (true for dynamic queries).
    The width never drops below the number of targets, so the problem stays solvable.
    """
    counts = query_valid_mask.sum(dim=1)
    positions = torch.arange(pred_dim, device=query_valid_mask.device)
    if not torch.equal(query_valid_mask, positions.unsqueeze(0) < counts.unsqueeze(1)):
        return pred_dim
    return min(max(int(counts.max()), int(lengths_np.max(initial=0))), pred_dim)


def match_multiprocess_prepared(
    solver_name: str,
    costs: torch.Tensor,
    lengths_np: np.ndarray,
    query_valid_mask: torch.Tensor | None = None,
    n_jobs: int = 8,
    trim_padded_queries: bool = False,
) -> torch.Tensor:
    """Multiprocess matching with the masking, transpose and slicing done on the costs' device.

    Gives the same indices as compute_matching + match_multiprocess, but the costs are copied off the
    device once, straight into a shared-memory buffer that is reused between calls.

    With trim_padded_queries, the padded query columns are dropped before solving. Every target is
    matched to the same query as without trimming, but lap1015 may order the unmatched queries
    differently: padded queries always come last, in their original slots.

    Raises:
        ValueError: If solver_name is not in the available SOLVERS.
    """
    if solver_name not in SOLVERS:
        raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {list(SOLVERS.keys())}")

    costs = costs.detach().to(torch.float32)
    batch_size, pred_dim, num_target = costs.shape
    width = pred_dim

    # Padded queries get a high cost so they won't be matched to valid targets
    if query_valid_mask is not None:
        query_valid_mask = query_valid_mask.detach().bool().to(costs.device)
        costs = costs.masked_fill(~query_valid_mask.unsqueeze(-1), float(np.finfo(np.float32).max / 10))
        if trim_padded_queries:
            width = _valid_query_width(query_valid_mask, lengths_np, pred_dim)

    # [batch, pred, true] -> [batch, true, pred], keeping only the target rows and query columns the solver reads
    max_len = min(int(lengths_np.max(initial=0)), num_target)
    costs_t = costs[:, :width].transpose(1, 2)[:, :max_len].contiguous()

    shape = (batch_size, max_len, width)
    nbytes = max(costs_t.numel() * costs_t.element_size(), 1)
    shm = _get_shm_buffer(nbytes)
    shm_arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    torch.from_numpy(shm_arr).copy_(costs_t)

    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs
    tasks = [(solver_name, shm.name, shape, i, int(lengths_np[i]), pred_dim) for i in range(batch_size)]
    pool = _get_process_pool(n_jobs)
    results = pool.map(_mp_match_task_persistent, tasks, chunksize=chunk_size)
    return torch.from_numpy(np.stack(results, axis=0))


def _get_dispatcher() -> ThreadPoolExecutor:
    global _DISPATCHER  # noqa: PLW0603
    with _POOL_LOCK:
        if _DISPATCHER is None:
            _DISPATCHER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="matcher-dispatch")
        return _DISPATCHER


class LayerwiseMatching:
    """Matching for a stack of cost matrices that become ready one at a time, e.g. one per decoder layer.

    Each slot's costs are masked and transposed on their device and copied, without blocking, into that
    slot's part of the shared-memory buffer (page-locked, so the copy runs in the background). A dispatcher
    thread waits for the copy and hands the slot's solves to the process pool, so they run while the device
    computes the next slots. Every solve reads the same numbers as match_multiprocess_prepared on the
    stacked costs, so the indices are identical.
    """

    def __init__(self, solver_name: str, num_slots: int, object_valid_mask: torch.Tensor, n_jobs: int, trim_padded_queries: bool = False):
        if solver_name not in SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {list(SOLVERS.keys())}")
        self.solver_name = solver_name
        self.num_slots = num_slots
        self.n_jobs = n_jobs
        self.batch_size, self.num_target = object_valid_mask.shape
        self.device = object_valid_mask.device
        self.trim_padded_queries = trim_padded_queries

        # The target counts are only read by the dispatcher, after the first slot's copy, so they can come
        # over without blocking too: the copy is queued before the slots', on the same stream
        self._lengths_device = object_valid_mask.detach().bool().sum(dim=1).to(torch.int32)
        self._lengths = self._to_host(self._lengths_device)
        # Number of query columns solved with trim_padded_queries, set from the first slot's query mask
        self._width: torch.Tensor | None = None

        self._shm: shared_memory.SharedMemory | None = None
        self._shape: tuple[int, int, int] | None = None
        self._pinned = False
        self._sources: list[torch.Tensor] = []
        self._pending: list[Future] = []

    def _to_host(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy to the host without blocking (into pinned memory), to be read only after a later slot's event."""
        if self.device.type != "cuda":
            return tensor
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        return host

    def _setup_buffer(self, pred_dim: int) -> None:
        # All target rows are copied (the solver reads only the first `length`), so nothing has to wait for the counts
        self._shape = (self.num_slots * self.batch_size, self.num_target, pred_dim)
        self._shm = _get_shm_buffer(max(int(np.prod(self._shape)) * 4, 1))
        self._buffer = torch.from_numpy(np.ndarray(self._shape, dtype=np.float32, buffer=self._shm.buf))
        self._pinned = self.device.type == "cuda" and _pin_shm_buffer(self._shm)

    def submit(self, slot: int, costs: torch.Tensor, query_valid_mask: torch.Tensor | None = None) -> None:
        """Queue one slot's costs, shape [batch, pred, true], and the solves that need them.

        Raises:
            ValueError: If the costs' shape differs from the first slot's.
        """
        costs = costs.detach().to(torch.float32)
        if self._shm is None:
            self._setup_buffer(costs.shape[1])
        if costs.shape[1:] != self._shape[1:][::-1]:
            raise ValueError(f"Slot {slot} costs have shape {tuple(costs.shape)}, expected [*, {self._shape[2]}, {self._shape[1]}]")

        # Padded queries get a high cost so they won't be matched to valid targets
        if query_valid_mask is not None:
            query_valid_mask = query_valid_mask.detach().bool().to(costs.device)
            costs = costs.masked_fill(~query_valid_mask.unsqueeze(-1), float(np.finfo(np.float32).max / 10))
            # Every slot has the same queries (one stack of layers), so the width is worked out once.
            # All columns are still copied; the workers solve only the first `width` of them.
            if self.trim_padded_queries and self._width is None:
                self._width = self._to_host(_valid_query_width_on_device(query_valid_mask, self._lengths_device, costs.shape[1]))

        # [batch, pred, true] -> [batch, true, pred]
        costs_t = costs.transpose(1, 2).contiguous()
        rows = slice(slot * self.batch_size, (slot + 1) * self.batch_size)
        self._buffer[rows].copy_(costs_t, non_blocking=self._pinned)

        event = None
        if self._pinned:
            event = torch.cuda.Event()
            event.record()
            # Keep the source alive until the copy has run
            self._sources.append(costs_t)
        self._pending.append(_get_dispatcher().submit(self._dispatch, event, slot))

    def _dispatch(self, event: torch.cuda.Event | None, slot: int) -> list:
        if event is not None:
            event.synchronize()
        lengths = self._lengths.numpy()
        pool = _get_process_pool(self.n_jobs)
        pred_dim = self._shape[2]
        width = int(self._width) if self._width is not None else pred_dim
        return [
            pool.apply_async(
                _mp_match_task_persistent,
                ((self.solver_name, self._shm.name, self._shape, slot * self.batch_size + b, int(lengths[b]), pred_dim, width),),
            )
            for b in range(self.batch_size)
        ]

    def results(self) -> list[torch.Tensor]:
        """Wait for every submitted slot; one [batch, pred] index tensor per slot, in submission order."""
        out = [torch.from_numpy(np.stack([r.get() for r in pending.result()], axis=0)) for pending in self._pending]
        self._sources.clear()
        self._pending.clear()
        # Drop the view so the buffer can be closed if it ever has to grow
        self._buffer = None
        return out


class Matcher(nn.Module):
    def __init__(
        self,
        default_solver: str = "scipy",
        adaptive_solver: bool = True,
        adaptive_check_interval: int = 1000,
        parallel_solver: bool = False,
        parallel_backend: Literal["thread", "process"] = "thread",
        n_jobs: int = 8,
        prepare_on_device: bool = False,
        trim_padded_queries: bool = False,
        overlap_layers: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        """ Used to match predictions to targets based on a given cost matrix.

        Parameters
        ----------
        default_solver : str
            The default solving algorithm to use.
        adaptive_solver : bool
            If true, then after every adaptive_check_interval calls of the solver,
            each solver algorithm is timed and used to determine the fastest solver, which
            is then set as the current solver.
        adaptive_check_interval : bool
            Interval for checking which solver is the fastest.
        parallel_solver : bool
            If true, then the solver will use a parallel implementation to speed up the matching.
        parallel_backend : str
            Parallel backend when parallel_solver is True. One of: 'thread', 'process'.
        n_jobs: int
            Number of jobs to use for parallel matching. Only used if parallel_solver is True.
        prepare_on_device : bool
            If true (process backend only), the costs are masked, transposed and sliced on their own
            device (usually the GPU) and copied once into a reused shared-memory buffer, instead of
            being copied and reshuffled on the CPU. The matching is identical; only the copying changes.
        trim_padded_queries : bool
            If true (needs prepare_on_device), padded query columns are dropped before solving, which makes
            the solve faster. Every target gets the same query, but with lap1015 the unmatched queries can
            come back in a different order (padded ones always last), so results are not bit-identical.
        overlap_layers : bool
            If true (needs prepare_on_device), a model can match each decoder layer's costs as soon as
            they exist (start_layerwise), so the solves run while the GPU computes the later layers. The
            matching is identical; only when the solves start changes. Combines with trim_padded_queries.
        verbose : bool
            If true, extra information on solver timing is printed.
        """
        if default_solver not in SOLVERS:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVERS.keys())}")
        if parallel_backend not in {"thread", "process"}:
            raise ValueError(f"parallel_backend must be 'thread' or 'process', got: {parallel_backend}")
        self.solver = default_solver
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.parallel_solver = parallel_solver
        self.parallel_backend = parallel_backend
        self.n_jobs = n_jobs
        self.prepare_on_device = prepare_on_device
        self.trim_padded_queries = trim_padded_queries
        if trim_padded_queries and not (prepare_on_device and parallel_solver and parallel_backend == "process"):
            raise ValueError("trim_padded_queries needs prepare_on_device=True, parallel_solver=True and parallel_backend='process'")
        self.overlap_layers = overlap_layers
        if overlap_layers and not (prepare_on_device and parallel_solver and parallel_backend == "process"):
            raise ValueError("overlap_layers needs prepare_on_device=True, parallel_solver=True and parallel_backend='process'")
        self.step = 0
        self.verbose = verbose

    def compute_matching(self, costs, object_valid_mask=None, query_valid_mask=None):
        if object_valid_mask is None:
            object_valid_mask = torch.ones((costs.shape[0], costs.shape[1]), dtype=torch.bool)

        object_valid_mask = object_valid_mask.detach().bool()
        batch_obj_lengths = torch.sum(object_valid_mask, dim=1).unsqueeze(-1)
        lengths_np = batch_obj_lengths.squeeze(-1).cpu().numpy().astype(np.int32, copy=False)

        pred_dim = costs.shape[1]

        # If we have invalid/padded queries, set their costs to a high value
        # so they won't be matched to valid targets
        if query_valid_mask is not None:
            query_valid_mask = query_valid_mask.detach().bool()
            # Set costs for invalid queries to max float32 value
            # costs shape: [batch, num_pred, num_target]
            invalid_query_mask = ~query_valid_mask.unsqueeze(-1)  # [batch, num_pred, 1]
            costs = np.where(invalid_query_mask.cpu().numpy(), np.finfo(np.float32).max / 10, costs)

        if self.parallel_solver:
            # Transpose costs: [batch, pred, true] -> [batch, true, pred]
            costs_t = np.ascontiguousarray(costs.swapaxes(1, 2))
            if self.parallel_backend == "thread":
                return match_parallel(SOLVERS[self.solver], costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)
            return match_multiprocess(self.solver, costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)

        # Sequential matching
        costs_t = costs.swapaxes(1, 2)
        default_idx = np.arange(pred_dim, dtype=np.int32)
        idxs = []

        for k in range(len(costs)):
            cost = costs_t[k][: lengths_np[k]]
            pred_idx = match_individual(SOLVERS[self.solver], cost, default_idx)
            idxs.append(pred_idx)

        return torch.from_numpy(np.stack(idxs))

    def start_layerwise(self, object_valid_mask: torch.Tensor, num_slots: int) -> LayerwiseMatching | None:
        """Start matching up to num_slots cost matrices one at a time, counting as one call of forward.

        Returns None when the costs should go through forward instead: overlap_layers is off, or this
        call is due an adaptive solver check, which times the solvers on all the costs at once.
        """
        if not self.overlap_layers or (self.adaptive_solver and self.step % self.adaptive_check_interval == 0):
            return None
        self.step += 1
        return LayerwiseMatching(self.solver, num_slots, object_valid_mask, self.n_jobs, trim_padded_queries=self.trim_padded_queries)

    @torch.no_grad()
    def forward(self, costs, object_valid_mask=None, query_valid_mask=None):
        # Convert costs to numpy on CPU for solver compatibility
        if self.prepare_on_device and self.parallel_solver and self.parallel_backend == "process":
            if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
                self.adapt_solver(costs.detach().to(torch.float32).cpu().numpy())
            if object_valid_mask is None:
                lengths_np = np.full(costs.shape[0], costs.shape[1], dtype=np.int32)
            else:
                lengths_np = object_valid_mask.detach().bool().sum(dim=1).cpu().numpy().astype(np.int32, copy=False)
            pred_idxs = match_multiprocess_prepared(
                self.solver, costs, lengths_np, query_valid_mask, n_jobs=self.n_jobs, trim_padded_queries=self.trim_padded_queries
            )
            self.step += 1
            assert torch.all(pred_idxs >= 0), "Matcher error!"
            return pred_idxs

        costs = costs.detach().to(torch.float32).cpu().numpy()

        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs)

        pred_idxs = self.compute_matching(costs, object_valid_mask, query_valid_mask)
        self.step += 1

        assert torch.all(pred_idxs >= 0), "Matcher error!"
        return pred_idxs

    def adapt_solver(self, costs):
        solver_times = {}

        if self.verbose:
            print("\nAdaptive LAP Solver: Starting solver check...")

        for solver in SOLVERS:
            self.solver = solver
            start_time = time.time()
            self.compute_matching(costs)
            solver_times[solver] = time.time() - start_time

            if self.verbose:
                print(f"Adaptive LAP Solver: Evaluated {solver}, took {solver_times[solver]:.2f}s")

        fastest_solver = min(solver_times, key=solver_times.get)

        if self.verbose:
            if fastest_solver != self.solver:
                print(f"Adaptive LAP Solver: Switching from {self.solver} solver to {fastest_solver} solver\n")
            else:
                print(f"Adaptive LAP Solver: Sticking with {self.solver} solver\n")

        self.solver = fastest_solver
