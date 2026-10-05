import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hepattn.models import matcher as matcher_module
from hepattn.models.matcher import SOLVERS, Matcher

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ORIGINAL_COMMIT = "ef88a68"  # matcher.py before prepare_on_device / trim_padded_queries existed
PROCESS = {"adaptive_solver": False, "parallel_solver": True, "parallel_backend": "process", "n_jobs": 4}


@pytest.fixture(scope="module")
def original_matcher(tmp_path_factory):
    """The Matcher class exactly as it was before these changes, loaded from git."""
    repo = Path(__file__).resolve().parents[2]
    src = subprocess.run(
        ["git", "show", f"{ORIGINAL_COMMIT}:src/hepattn/models/matcher.py"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    path = tmp_path_factory.mktemp("original") / "original_matcher.py"
    path.write_text(src)
    # The process pool's spawned workers unpickle original_matcher._mp_match_task by module name,
    # so they must be able to import it too (spawn passes sys.path on to the workers)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("original_matcher", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["original_matcher"] = module
    spec.loader.exec_module(module)
    return module.Matcher


def make_inputs(rng: np.random.Generator, batch_size: int, num_pred: int, num_target: int, device: str, pad_value: float = float("nan")):
    """Random costs with dynamic-query-style masks: the valid targets and the valid queries come first."""
    costs = torch.from_numpy(rng.random((batch_size, num_pred, num_target), dtype=np.float32) * 10)
    object_valid = torch.zeros((batch_size, num_target), dtype=torch.bool)
    query_valid = torch.zeros((batch_size, num_pred), dtype=torch.bool)
    for i in range(batch_size):
        object_valid[i, : int(rng.integers(1, num_target // 2))] = True
        query_valid[i, : int(rng.integers(num_target // 2, num_pred + 1))] = True
    # Padded targets may hold anything (NaN in practice), the matcher must not read them
    costs[~object_valid.unsqueeze(1).expand(-1, num_pred, -1)] = pad_value
    return costs.to(device), object_valid.to(device), query_valid.to(device)


SIZES = [(60, 60), (200, 200), (120, 200), (200, 200)]  # varying, so the reused buffer grows and is reused smaller


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prepare_on_device_identical_to_original(original_matcher, solver, device, dtype):
    rng = np.random.default_rng(0)
    original = original_matcher(default_solver=solver, **PROCESS)
    new = Matcher(default_solver=solver, **PROCESS, prepare_on_device=True)
    for num_pred, num_target in SIZES:
        costs, object_valid, query_valid = make_inputs(rng, 4, num_pred, num_target, device)
        costs = costs.to(dtype)
        expected = original(costs, object_valid, query_valid)
        got = new(costs, object_valid, query_valid)
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected)  # the full permutation, unmatched queries included
        assert torch.equal(new(costs, object_valid), original(costs, object_valid))


@pytest.mark.parametrize("device", DEVICES)
def test_prepare_on_device_without_masks(original_matcher, device):
    rng = np.random.default_rng(1)
    costs = torch.from_numpy(rng.random((3, 50, 50), dtype=np.float32)).to(device)
    kwargs = {"default_solver": "scipy", **PROCESS, "n_jobs": 2}
    assert torch.equal(Matcher(**kwargs, prepare_on_device=True)(costs), original_matcher(**kwargs)(costs))


def test_prepare_on_device_reuses_buffer():
    rng = np.random.default_rng(2)
    matcher = Matcher(default_solver="scipy", **PROCESS, prepare_on_device=True)
    costs, object_valid, query_valid = make_inputs(rng, 2, 80, 80, "cpu")
    matcher(costs, object_valid, query_valid)
    name = matcher_module._SHM_BUFFER.name
    matcher(costs, object_valid, query_valid)
    assert matcher_module._SHM_BUFFER.name == name


def test_prepare_on_device_with_adaptive_solver(original_matcher):
    rng = np.random.default_rng(3)
    kwargs = {**PROCESS, "default_solver": "scipy", "adaptive_solver": True, "adaptive_check_interval": 2, "n_jobs": 2}
    original, new = original_matcher(**kwargs), Matcher(**kwargs, prepare_on_device=True)
    for _ in range(3):
        # The adaptive solver's benchmark ignores the masks (in both versions), so pad with finite costs
        costs, object_valid, query_valid = make_inputs(rng, 2, 80, 80, "cpu", pad_value=1e4)
        assert torch.equal(new(costs, object_valid, query_valid), original(costs, object_valid, query_valid))


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("device", DEVICES)
def test_trim_matches_every_target_to_the_same_query(original_matcher, solver, device):
    rng = np.random.default_rng(4)
    original = original_matcher(default_solver=solver, **PROCESS)
    trimmed = Matcher(default_solver=solver, **PROCESS, prepare_on_device=True, trim_padded_queries=True)
    for num_pred, num_target in SIZES:
        costs, object_valid, query_valid = make_inputs(rng, 4, num_pred, num_target, device)
        expected = original(costs, object_valid, query_valid)
        got = trimmed(costs, object_valid, query_valid)
        for k in range(len(got)):
            n_targets, n_queries = int(object_valid[k].sum()), int(query_valid[k].sum())
            # Same query for every target
            assert torch.equal(got[k, :n_targets], expected[k, :n_targets])
            # Still a permutation of all queries
            assert torch.equal(got[k].sort().values, torch.arange(num_pred, dtype=got.dtype))
            # Unmatched valid queries fill the valid slots, padded queries stay in their own (masked) slots
            assert set(got[k, n_targets:n_queries].tolist()) == set(range(n_queries)) - set(got[k, :n_targets].tolist())
            assert torch.equal(got[k, n_queries:], torch.arange(n_queries, num_pred, dtype=got.dtype))
        if solver == "scipy":
            # scipy lists unmatched queries in index order, so with scipy trimming changes nothing at all
            assert torch.equal(got, expected)


def test_trim_falls_back_when_valid_queries_are_not_first(original_matcher):
    rng = np.random.default_rng(5)
    costs, object_valid, query_valid = make_inputs(rng, 2, 120, 120, "cpu")
    query_valid = query_valid.flip(-1)  # valid queries at the end: can't trim, must give the original result
    kwargs = {"default_solver": "lap1015_late" if "lap1015_late" in SOLVERS else "scipy", **PROCESS}
    expected = original_matcher(**kwargs)(costs, object_valid, query_valid)
    got = Matcher(**kwargs, prepare_on_device=True, trim_padded_queries=True)(costs, object_valid, query_valid)
    assert torch.equal(got, expected)


def test_trim_keeps_enough_columns_when_queries_are_fewer_than_targets(original_matcher):
    rng = np.random.default_rng(6)
    costs, object_valid, query_valid = make_inputs(rng, 2, 100, 100, "cpu", pad_value=1.0)  # masks are reset below
    object_valid[:] = False
    object_valid[:, :60] = True
    query_valid[:] = False
    query_valid[:, :40] = True  # fewer valid queries than targets: padded queries must still be matchable
    got = Matcher(default_solver="scipy", **PROCESS, prepare_on_device=True, trim_padded_queries=True)(costs, object_valid, query_valid)
    expected = original_matcher(default_solver="scipy", **PROCESS)(costs, object_valid, query_valid)
    assert torch.equal(got[:, :60], expected[:, :60])


def test_trim_needs_prepare_on_device():
    with pytest.raises(ValueError, match="trim_padded_queries needs"):
        Matcher(default_solver="scipy", **PROCESS, trim_padded_queries=True)
