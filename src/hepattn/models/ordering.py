"""E2LSH geometric ordering for locality-aware attention (HEPTv2, arXiv:2606.20437).

This is the single source of truth for the HEPT E2LSH serialization used both by
the benchmark (``experiments/colliderml/scripts/orderings.py`` delegates here) and
by the model's per-head ordering. Keeping one implementation guarantees the model
uses exactly the ordering the benchmark validated.

An ordering is built from two frozen pieces per hash table:

    * ``alpha``  -- a random Gaussian E2LSH projection of the 2D ``(eta, phi)``
      coordinates (the tie-break *within* a quantile cell).
    * ``region_counts`` -- ``[eta_count, phi_count]`` equal-occupancy quantile-bin
      counts (the AND-construction *cell* structure).

The scalar ordering value of hit ``i`` is

    ``o_i = alpha . (eta_i, phi_i)  +  R_eta . D  +  R_phi . D . (ceil(eta_count) + 1)``

where ``R_eta`` / ``R_phi`` are the hit's 1-based quantile-bin indices and ``D`` is
the span of the projection. Sorting by ``o`` groups hits by 2D quantile cell
(phi-major, eta-minor) and, within a cell, by the projection -- so hits close in
``(eta, phi)`` stay close in the sequence. ``argsort(o)`` is the ordering
(permutation); ``argsort(argsort(o))`` is the ``rank`` (inverse permutation) that a
sliding-window ``mask_mod`` consumes.

Both ``alpha`` and ``region_counts`` are drawn once and frozen: a given event always
serialises the same way. The RNG draw lives in :func:`draw_lsh_table` (numpy, so it
reproduces the benchmark's ``np.random.default_rng(seed)`` tables bit-for-bit); the
ordering arithmetic lives in :func:`lsh_order_values` (torch, batched over a grid of
tables).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

_EPS = 1e-12


def hept_region_counts(num_regions: int, rng: np.random.Generator, num_axes: int = 2) -> np.ndarray:
    """Per-axis quantile-bin counts ``[eta_count, phi_count]`` (HEPTv2 ``get_regions``).

    Their product is approximately ``num_regions`` (each rounded to a multiple of
    1/3), giving the number of equal-occupancy bins along eta and phi.
    """
    lb = 2.0
    ub = 2 * num_regions ** (1 / num_axes) - lb
    r = rng.random(num_axes) * (ub - lb) + lb
    r = (num_regions / r.prod()) ** (1 / num_axes) * r
    return np.round(r * 3) / 3


def draw_lsh_table(rng: np.random.Generator, num_regions: int = 100, num_axes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Draw one frozen hash table: ``(alpha, region_counts)``, each shape ``(num_axes,)``.

    The draw order -- region counts first, then the Gaussian projection -- is fixed
    so that ``draw_lsh_table(np.random.default_rng(k))`` reproduces the benchmark's
    ``lsh_k`` table exactly. Do not reorder the two draws.
    """
    region_counts = hept_region_counts(num_regions, rng, num_axes)
    alpha = rng.normal(0.0, 1.0, size=num_axes)
    return alpha, region_counts


def lsh_order_values(coords: Tensor, alpha: Tensor, region_counts: Tensor) -> Tensor:
    """HEPT E2LSH ordering values for a grid of hash tables.

    Args:
        coords: ``(..., N, 2)`` per-hit ``(eta, phi)``. Any leading batch dims are
            allowed and broadcast through.
        alpha: ``(G, 2)`` frozen Gaussian projections (``G`` tables).
        region_counts: ``(G, 2)`` frozen ``[eta_count, phi_count]`` per table.

    Returns:
        ``(..., G, N)`` ordering values ``o``. ``argsort(o, dim=-1)`` is the ordering.

    All arithmetic runs in ``alpha.dtype`` (use float64 to match the numpy
    benchmark exactly; float32 is fine for the model). The two O(N log N) coordinate
    ranks are computed once and shared across all ``G`` tables; only the cheap
    per-table region size varies, so the grid adds negligible cost over one table.
    """
    dtype = alpha.dtype
    coords = coords.to(dtype)
    n = coords.shape[-2]
    if n == 0:
        return coords.new_zeros((*coords.shape[:-2], alpha.shape[0], 0))

    eta, phi = coords[..., 0], coords[..., 1]  # (..., N)
    eta_count, phi_count = region_counts[:, 0], region_counts[:, 1]  # (G,)

    # E2LSH projection alpha.(eta, phi) per table, and its span D.
    hashed = torch.einsum("...nd,gd->...gn", coords, alpha)  # (..., G, N)
    span = (hashed.amax(dim=-1) - hashed.amin(dim=-1)).clamp_min(_EPS).unsqueeze(-1)  # (..., G, 1)

    # Equal-occupancy quantile-bin index (1-based) per axis; the argsort-of-argsort
    # rank is data-only, so compute it once and broadcast the per-table bin size.
    eta_rank = _rank(eta).unsqueeze(-2)  # (..., 1, N)
    phi_rank = _rank(phi).unsqueeze(-2)
    region_eta = torch.div(eta_rank, torch.ceil(n / eta_count).view(-1, 1), rounding_mode="floor") + 1  # (..., G, N)
    region_phi = torch.div(phi_rank, torch.ceil(n / phi_count).view(-1, 1), rounding_mode="floor") + 1

    return hashed + region_eta * span + region_phi * span * (torch.ceil(eta_count) + 1).view(-1, 1)


def _rank(v: Tensor) -> Tensor:
    """Stable rank of each element along the last dim (inverse of ``argsort``)."""
    return torch.argsort(torch.argsort(v, dim=-1, stable=True), dim=-1, stable=True).to(v.dtype)


class E2LSHOrderingGrid(nn.Module):
    """A frozen ``(num_heads, n_hashes)`` grid of HEPT E2LSH orderings for one layer.

    Each ``(head, hash)`` cell owns an independent, frozen hash table; ``forward``
    turns per-hit coordinates into the ``rank`` (inverse-permutation) tables a
    per-head sliding-window ``mask_mod`` consumes. The projections are registered as
    buffers, so they persist in the checkpoint and never train -- a given event
    always serialises the same way (see module docstring).

    Per-cell seeds are ``base_seed + layer_idx * num_heads * n_hashes
    + head * n_hashes + hash``, so layer 0 / head 0 reproduces the benchmark's
    ``lsh_0, lsh_1, ...`` tables and every other cell is a distinct ordering.

    Args:
        num_heads: Attention heads (the ``H`` axis of the mask).
        n_hashes: OR hash tables per head (the ``c`` axis merged by LSE).
        num_regions: Target number of quantile cells (HEPTv2 default 100).
        base_seed: Global seed offset for the whole network.
        layer_idx: This layer's index, so each layer redraws distinct orderings.
        dtype: Compute dtype for the frozen projections (float32 for the model).
    """

    def __init__(
        self,
        num_heads: int,
        n_hashes: int,
        *,
        num_regions: int = 100,
        base_seed: int = 0,
        layer_idx: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_hashes = n_hashes
        self.num_regions = num_regions

        cells_per_layer = num_heads * n_hashes
        alpha = np.empty((num_heads, n_hashes, 2), dtype=np.float64)
        region_counts = np.empty((num_heads, n_hashes, 2), dtype=np.float64)
        for head in range(num_heads):
            for h in range(n_hashes):
                seed = base_seed + layer_idx * cells_per_layer + head * n_hashes + h
                alpha[head, h], region_counts[head, h] = draw_lsh_table(np.random.default_rng(seed), num_regions)

        # Frozen: buffers persist in the checkpoint and carry no gradient.
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=dtype))
        self.register_buffer("region_counts", torch.as_tensor(region_counts, dtype=dtype))

    def forward(self, coords: Tensor) -> Tensor:
        """Rank tables for the grid.

        Args:
            coords: ``(..., N, 2)`` per-hit ``(eta, phi)``.

        Returns:
            ``(..., num_heads, n_hashes, N)`` int64 ``rank`` tables: entry ``[.., head,
            hash, t]`` is the position of token ``t`` in that cell's ordering.
        """
        alpha = self.alpha.flatten(0, 1)  # (G, 2)
        region_counts = self.region_counts.flatten(0, 1)  # (G, 2)
        o = lsh_order_values(coords, alpha, region_counts)  # (..., G, N)
        ranks = torch.argsort(torch.argsort(o, dim=-1, stable=True), dim=-1, stable=True)  # (..., G, N)
        return ranks.unflatten(-2, (self.num_heads, self.n_hashes))
