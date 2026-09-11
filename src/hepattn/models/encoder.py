from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from hepattn.flex import per_head_window_mask_mod, relative_position, relative_position_wrapped
from hepattn.flex.sliding_window import sliding_window_mask, sliding_window_mask_wrapped
from hepattn.models.attention import FLASH_ATTN_TYPES, Attention, repad_from_flash_varlen, unpad_for_flash_varlen
from hepattn.models.dense import Dense
from hepattn.models.norm import NORM_TYPES, get_hybrid_norm_config
from hepattn.models.ordering import E2LSHOrderingGrid

create_block_mask = torch.compile(create_block_mask, dynamic=True)  # ty: ignore[invalid-assignment]

SCORE_MODS = {
    "relative_position": relative_position,
    "relative_position_wrapped": relative_position_wrapped,
}


class DropPath(nn.Module):
    """Randomly drop layers: https://arxiv.org/abs/1603.09382."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(mask_shape, dtype=x.dtype, device=x.device)
        return x * mask.floor().div(keep_prob)


class LayerScale(nn.Module):
    """Learn residual strength: https://arxiv.org/abs/2103.17239."""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Residual(nn.Module):
    def __init__(
        self,
        fn: nn.Module,
        dim: int,
        norm: str | None,
        post_norm: bool = False,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
    ) -> None:
        """Neatly wrap x = x + drop(scale * fn(norm(x))).

        Args:
            dim: Dimension of the input and output.
            fn: The module to wrap. Must be non-resizing.
            norm: The normalization layer.
            post_norm: Whether to apply hybrid norm [2503.04598] style post norm.
            layer_scale: Initial value for the layer_scale. If None, no layer_scale is applied.
            drop_path: Drop path rate.

        Raises:
            ValueError: If the input arguments are invalid.
        """
        super().__init__()

        if post_norm and not norm:
            raise ValueError("post_norm is True but no norm is provided.")
        if norm is not None and not isinstance(norm, str):
            raise ValueError("norm must be a string or None.")
        if norm is not None and norm not in NORM_TYPES:
            raise ValueError(f"Unsupported norm: {norm}. Must be one of {list(NORM_TYPES.keys())}")

        self.fn = fn
        self.ls = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.dp = DropPath(drop_path) if drop_path else nn.Identity()
        self.post_norm = post_norm

        self.norm = NORM_TYPES[norm](dim) if norm else nn.Identity()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.post_norm:
            x = self.norm(x)
            return x + self.dp(self.ls(self.fn(x, **kwargs)))
        return x + self.dp(self.ls(self.fn(self.norm(x), **kwargs)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 0,
        norm: str = "LayerNorm",
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        value_residual: bool = False,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ) -> None:
        """Encoder layer: self-attention followed by feed-forward.

        Args:
            dim: Dimension of the embeddings.
            depth: The depth of the layer.
            norm: The normalization layer.
            drop_path: Drop path rate.
            layer_scale: Initial layer_scale value.
            value_residual: Whether to apply a residual connection from initial values.
            qkv_norm: Whether to use qkv norm in the Attention layer
            hybrid_norm: Whether to use HybridNorm from 2503.04598.
            dense_kwargs: Keyword arguments for dense layer.
            attn_kwargs: Keyword arguments for self-attention layer.
        """
        super().__init__()

        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

        # handle value residual
        attn_kwargs["value_residual"] = value_residual
        attn_kwargs["is_first_layer"] = depth == 0

        self.dim = dim
        residual = partial(Residual, dim=dim, layer_scale=layer_scale, drop_path=drop_path)
        self.attn = residual(Attention(self.dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.dense = residual(Dense(self.dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, **kwargs))


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        attn_type: str = "torch",
        window_size: int | None = None,
        window_wrap: bool = False,
        or_n_hashes: int | None = None,
        or_num_regions: int = 100,
        or_base_seed: int = 0,
        score_mod: str | None = None,
        value_residual: bool = False,
        num_register_tokens: int | None = None,
        **layer_kwargs,
    ) -> None:
        """Transformer encoder.

        Args:
            num_layers: Number of layers.
            dim: Dimension of the embeddings at each layer.
            attn_type: Type of attention to use.
            window_size: Window size for the sliding window.
            window_wrap: Whether to wrap the window by wrapping the input sequence or the mask, depending on the attn_type.
            or_n_hashes: OR hash tables per head. None = feature off.
            or_num_regions: Total number of (eta, phi) cells the detector is chopped into.
            or_base_seed: Global seed offset. Every cell's random draw is base_seed + layer_idx*(num_heads*n_hashes) + head*n_hashes + hash
                          so base_seed is the starting number of the sequence. 0 reproduces the benchmark.
            score_mod: Score modification function.
            value_residual: Add a residual connection from the initial layer values.
            num_register_tokens: Number of register tokens to add at the beginning of the sequence. If None, no register tokens are added.
                Register tokens are removed from the output by default.
            **layer_kwargs: Keyword arguments for EncoderLayer.
        """
        super().__init__()

        assert not window_wrap or window_size, "Window size must be set if window wrap is True."
        assert attn_type != "flex" or score_mod is None, "Score mod is only supported with flex attention."
        assert not (num_register_tokens is not None and window_size is not None), "Register tokens are not compatible with window attention."
        assert or_n_hashes is None or attn_type == "flex", "OR amplification requires flex attention."
        assert or_n_hashes is None or window_size, "OR amplification requires a window_size to be set."
        assert or_n_hashes is None or not window_wrap, (
            "OR amplification has no wrapped variant: LSH ordering treats phi as a linear coordinate, so not periodic."
        )

        layer_kwargs = layer_kwargs or {}

        self.num_layers = num_layers
        self.dim = dim
        self.attn_type = attn_type
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.or_n_hashes = or_n_hashes
        self.score_mod = SCORE_MODS[score_mod] if score_mod else None
        self.value_residual = value_residual
        self.num_register_tokens = num_register_tokens

        # Initialize register tokens if specified
        if self.num_register_tokens is not None:
            self.register_tokens = nn.Parameter(torch.randn(1, self.num_register_tokens, dim))
        else:
            self.register_tokens = None

        # handle masking
        self.mask_mod = None
        self.seq_len = None

        # handle attention
        attn_kwargs = layer_kwargs.get("attn_kwargs", None) or {}
        attn_kwargs["attn_type"] = attn_type
        layer_kwargs["value_residual"] = self.value_residual
        # Only the flash kernels take a window as an argument; flex expresses its window
        # in the mask_mod / BlockMask instead, and Attention rejects a window_size for it.
        attn_kwargs["window_size"] = window_size if attn_type in FLASH_ATTN_TYPES else None
        layer_kwargs["attn_kwargs"] = attn_kwargs

        self.layers = torch.nn.ModuleList([EncoderLayer(dim=dim, depth=i, **layer_kwargs) for i in range(num_layers)])
        # E2LSHOrderingGrid needs num_heads to size grid
        # Read this off a constructed layer so attn_kwargs override is tracked automatically
        num_heads = self.layers[0].attn.fn.num_heads
        self.ordering_grids = None
        if or_n_hashes is not None:
            self.ordering_grids = torch.nn.ModuleList([
                E2LSHOrderingGrid(num_heads, or_n_hashes, num_regions=or_num_regions, base_seed=or_base_seed, layer_idx=i) for i in range(num_layers)
            ])

    def set_backend(self, attn_type: str):
        self.attn_type = attn_type
        for layer in self.layers:
            self.attn_type = layer.attn.fn.set_backend(self.attn_type)

    def forward(
        self, x: Tensor, x_sort_value: Tensor | None = None, kv_mask: Tensor | None = None, x_coords: Tensor | None = None, **kwargs
    ) -> Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[-2]

        # x_coords (B, N, 2) carries the per-hit (eta, phi)
        assert self.ordering_grids is None or x_coords is not None, "OR amplification requires x_coords (B, N, 2) of (eta, phi)."

        # If value to sort on is provided, use it to sort the tokens
        # We don't need to use the stable sort assuming that the sort values are unique
        x_sort_idx = None
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, dim=-1)
            x = torch.gather(x, dim=-2, index=x_sort_idx.unsqueeze(-1).expand_as(x))  # x is a stack of token vectors

            # Also permute the kv mask if we have one
            if kv_mask is not None:
                kv_mask = torch.gather(kv_mask, dim=-1, index=x_sort_idx)

            # The coords must follow their tokens, otherwise the rank tables built from
            # them describe hits that are no longer at those positions
            if x_coords is not None:
                x_coords = torch.gather(x_coords, dim=-2, index=x_sort_idx.unsqueeze(-1).expand_as(x_coords))

        # Add register tokens at the beginning of the sequence
        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([register_tokens, x], dim=1)

            # Allow registers to participate in attention if a mask is provided
            if kv_mask is not None:
                register_mask = torch.full((1, self.num_register_tokens), True, device=kv_mask.device, dtype=kv_mask.dtype).expand(batch_size, -1)
                kv_mask = torch.cat([register_mask, kv_mask], dim=1)

        # Handle flash-varlen attention unpadding at encoder level
        varlen_kwargs = None
        if self.attn_type == "flash-varlen" and kv_mask is not None:
            x, indices, varlen_kwargs = unpad_for_flash_varlen(x, kv_mask)
            kwargs["varlen_kwargs"] = varlen_kwargs
        elif self.attn_type == "flash-varlen":
            raise ValueError("kv_mask must be provided for flash-varlen attention.")

        attn_mask = None
        if self.ordering_grids is None:
            # Initialise sliding window mask
            if self.mask_mod is None and self.attn_type != "flash" and self.window_size:
                self.seq_len = torch.tensor([1], device=x.device)
                self.mask_mod = (
                    sliding_window_mask(self.window_size) if not self.window_wrap else sliding_window_mask_wrapped(self.window_size, self.seq_len)
                )

            # Handle masking
            if self.attn_type == "torch" and self.mask_mod:
                attn_mask = create_mask(self.mask_mod, 1, 1, seq_len, seq_len, device=str(x.device))
            elif self.attn_type == "flex" and self.mask_mod:
                self.seq_len[0] = seq_len
                attn_mask = create_block_mask(self.mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=str(x.device))

        # Add wrapping for flash attention with sliding window
        if self.attn_type == "flash" and self.window_wrap:
            x = torch.cat([x[:, -self.window_size // 2 :], x, x[:, : self.window_size // 2]], dim=1)

        # Apply layers
        initial_values = {} if self.value_residual else None
        for i, layer in enumerate(self.layers):
            layer_mask = attn_mask
            if self.ordering_grids is not None:
                ranks = self.ordering_grids[i](x_coords[0]).flatten(0, 1)  # (num_heads * n_hashes, N)
                layer_mask = create_block_mask(
                    per_head_window_mask_mod(ranks, self.window_size),
                    B=None,
                    H=ranks.shape[0],
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=str(x.device),
                )
            x = layer(
                x,
                attn_mask=layer_mask,
                score_mod=self.score_mod,
                initial_values=initial_values,
                kv_mask=kv_mask,
                or_n_hashes=self.or_n_hashes,
                **kwargs,
            )

        # Remove wrapping for flash attention with sliding window
        if self.attn_type == "flash" and self.window_wrap:
            x = x[:, self.window_size // 2 : -self.window_size // 2]

        # Repad sequence if flash-varlen attention is used
        if varlen_kwargs is not None:
            seq_len = seq_len if not self.num_register_tokens else seq_len + self.num_register_tokens
            x = repad_from_flash_varlen(x, batch_size, seq_len, indices)

        # Remove register tokens
        if self.register_tokens is not None:
            x = x[:, self.num_register_tokens :]
            if kv_mask is not None:
                kv_mask = kv_mask[:, self.num_register_tokens :]

        # If we sorted the tokens, undo the sorting
        if x_sort_value is not None and x_sort_idx is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            x = torch.gather(x, -2, x_unsort_idx.unsqueeze(-1).expand_as(x))

        return x


def change_attn_backends(module: nn.Module, backend: str) -> None:
    """Recursively change the attention backend of a module and all its children.

    Args:
        module: The module to update.
        backend: The attention backend to set.
    """
    if isinstance(module, Encoder):
        module.set_backend(backend)
        return
    if isinstance(module, Attention):
        module.set_backend(backend)
        return
    for child in module.children():
        change_attn_backends(child, backend)
