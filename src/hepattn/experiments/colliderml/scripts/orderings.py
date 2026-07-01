"""Hit ordering (sorting) techniques for locality benchmarking.

Each ordering takes per-hit ``eta`` and ``phi`` arrays and returns a permutation
``order`` such that ``hits[order]`` is the sorted sequence the model would see.
The orderings here are deliberately pure-numpy and model-independent so they can
be reused by the benchmark, notebooks, or dropped into the data pipeline.

Available orderings (see ``SORTERS``):
    - ``random``          : random permutation (baseline / worst case).
    - ``phi``             : sort by azimuthal angle only (the model's current
                            ``input_sort_field: phi``).
    - ``hilbert``         : Hilbert space-filling curve over (eta, phi).
    - ``lsh``             : random-projection strip LSH over (eta, phi).

All orderings are deterministic given the passed ``numpy`` random generator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Ordering = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]

_EPS = 1e-12


def _quantize(values: np.ndarray, bits: int) -> np.ndarray:
    """Map a 1D float array onto integer grid coordinates in ``[0, 2**bits)``.

    Uses per-array min/max so it is robust to whatever range the coordinate
    happens to span (rank-preserving, which is all the curve needs).
    """
    n = 1 << bits
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < _EPS:
        return np.zeros(values.shape, dtype=np.int64)
    scaled = (values - vmin) / (vmax - vmin) * (n - 1)
    return np.clip(np.floor(scaled + 0.5), 0, n - 1).astype(np.int64)


def _hilbert_xy2d(side: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised Hilbert curve index for integer coords in ``[0, side)``.

    ``side`` must be a power of two. Returns the distance ``d`` along the curve;
    sorting by ``d`` yields a 2D-locality-preserving ordering. Faithful port of
    the canonical iterative algorithm (Wikipedia, "Hilbert curve").
    """
    x = x.astype(np.int64, copy=True)
    y = y.astype(np.int64, copy=True)
    d = np.zeros(x.shape, dtype=np.int64)

    s = side // 2
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)

        # rotate/reflect the quadrant (rot(): reflect when ry == 0)
        swap = ry == 0
        flip = swap & (rx == 1)
        x[flip] = (side - 1) - x[flip]
        y[flip] = (side - 1) - y[flip]
        x_swapped = np.where(swap, y, x)
        y_swapped = np.where(swap, x, y)
        x, y = x_swapped, y_swapped

        s //= 2
    return d


def order_random(eta: np.ndarray, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(eta.shape[0])


def order_phi(eta: np.ndarray, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Matches the model's `input_sort_field: phi` (ascending argsort).
    return np.argsort(phi, kind="stable")


def order_hilbert(eta: np.ndarray, phi: np.ndarray, rng: np.random.Generator, *, bits: int = 16) -> np.ndarray:
    """Order by a Hilbert curve over the (eta, phi) plane."""
    side = 1 << bits
    x = _quantize(eta, bits)
    y = _quantize(phi, bits)
    d = _hilbert_xy2d(side, x, y)
    return np.argsort(d, kind="stable")


def order_lsh(
    eta: np.ndarray,
    phi: np.ndarray,
    rng: np.random.Generator,
    *,
    n_bands: int | None = None,
) -> np.ndarray:
    """Random-projection strip LSH over (eta, phi).

    Standardise the coordinates, apply a random 2D rotation, cut the first
    rotated axis into ``n_bands`` equal-width strips, then read the strips out in
    a serpentine (boustrophedon) order along the second axis. Nearby points fall
    in the same or adjacent strips with high probability, so the resulting index
    preserves 2D locality. Randomised via ``rng``; ``n_bands`` defaults to
    ``round(sqrt(N))`` which balances locality along the two axes.
    """
    num = eta.shape[0]
    if num == 0:
        return np.zeros(0, dtype=np.int64)
    if n_bands is None:
        n_bands = max(1, int(round(np.sqrt(num))))

    e = (eta - eta.mean()) / (eta.std() + _EPS)
    p = (phi - phi.mean()) / (phi.std() + _EPS)

    theta = rng.uniform(0.0, np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    u = cos_t * e + sin_t * p  # primary axis -> strips
    v = -sin_t * e + cos_t * p  # secondary axis -> within-strip order

    # Assign each point to a strip via rank-based quantiles (balanced strips).
    u_rank = np.argsort(np.argsort(u, kind="stable"), kind="stable")
    band = np.minimum((u_rank * n_bands) // num, n_bands - 1)

    # Serpentine: even strips ascending in v, odd strips descending.
    direction = np.where(band % 2 == 0, 1.0, -1.0)
    key = band.astype(np.float64) * (2.0 * num) + direction * v
    return np.argsort(key, kind="stable")


SORTERS: dict[str, Ordering] = {
    "random": order_random,
    "phi": order_phi,
    "hilbert": order_hilbert,
    "lsh": order_lsh,
}
