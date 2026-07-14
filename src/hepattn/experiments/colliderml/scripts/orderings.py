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
    - ``lsh``             : HEPTv2 E2LSH serialization over (eta, phi) (one hash table).

Only ``phi`` is periodic: sorting by the azimuthal angle maps the circle onto
the (wrapped) 1D sequence, so phi = +/-pi is healed by the model's
``window_wrap: true``. The 2D orderings (``hilbert``, ``lsh``) treat phi as a
plain linear coordinate; a 1D window cannot heal the interior phi crossings a
2D curve makes, so no periodic handling is applied to them.

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


def order_hilbert(eta: np.ndarray, phi: np.ndarray, rng: np.random.Generator, *, bits: int = 10) -> np.ndarray:
    """Order by a Hilbert curve over the (eta, phi) plane (2**bits grid per axis)."""
    side = 1 << bits
    x = _quantize(eta, bits)
    y = _quantize(phi, bits)
    d = _hilbert_xy2d(side, x, y)
    return np.argsort(d, kind="stable")


def _hept_region_counts(num_regions: int, rng: np.random.Generator, num_and_hashes: int = 2) -> np.ndarray:
    """Per-axis region counts ``[eta_count, phi_count]`` (HEPTv2 ``get_regions``).

    Their product is approximately ``num_regions`` (rounded to multiples of 1/3),
    giving the number of equal-occupancy quantile bins along eta and phi.
    """
    lb = 2.0
    ub = 2 * num_regions ** (1 / num_and_hashes) - lb
    r = rng.random(num_and_hashes) * (ub - lb) + lb
    r = (num_regions / r.prod()) ** (1 / num_and_hashes) * r
    return np.round(r * 3) / 3


def _quantile_region(v: np.ndarray, count: float) -> np.ndarray:
    """Equal-occupancy quantile-bin index (1-based) of each element."""
    n = v.shape[0]
    rank = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.float64)
    region_size = np.ceil(n / count)
    return rank // region_size + 1


def order_lsh(eta: np.ndarray, phi: np.ndarray, rng: np.random.Generator, *, num_regions: int = 100) -> np.ndarray:
    """HEPTv2 locality-sensitive-hashing serialization over (eta, phi).

    Reproduces one OR-table/head of HEPTv2's LSH serialization (arXiv:2606.20437):
    a fixed random Gaussian projection ``alpha.(eta, phi)`` (E2LSH) plus
    equal-occupancy quantile-bin indices in eta and phi, combined as

        ``o = alpha.(eta, phi) + R_eta * D + R_phi * D * (ceil(eta_count) + 1)``

    where ``D`` is the range of the projection. Sorting by ``o`` orders hits by
    their 2D (eta, phi) quantile cell (phi-major, eta-minor) and, within a cell, by
    the random projection, so hits nearby in (eta, phi) stay close in the sequence.

    The projection and region counts are drawn from ``rng``; a single call is one
    hash table. HEPTv2 uses several *independent* such orderings (multiple heads)
    and unions them -- see the reachability table, which runs this at several seeds.
    """
    n = eta.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    counts = _hept_region_counts(num_regions, rng)  # [eta_count, phi_count]
    alpha = rng.normal(0.0, 1.0, size=2)  # E2LSH random projection
    hashed = eta * alpha[0] + phi * alpha[1]
    hash_shift = max(float(hashed.max() - hashed.min()), _EPS)

    region_eta = _quantile_region(eta, counts[0])
    region_phi = _quantile_region(phi, counts[1])
    o = hashed + region_eta * hash_shift + region_phi * hash_shift * (np.ceil(counts[0]) + 1)
    return np.argsort(o, kind="stable")


SORTERS: dict[str, Ordering] = {
    "random": order_random,
    "phi": order_phi,
    "hilbert": order_hilbert,
    "lsh": order_lsh,
}
