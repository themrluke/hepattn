"""MaskFormer wiring of hit coordinates for OR amplification (step 4.10).

``Encoder`` builds a per-head ordering from ``x_coords``, and a rank table is only meaningful
if row ``i`` of the coordinates describes the same hit as row ``i`` of the tokens. MaskFormer is
where that pairing can silently break, because the sequence may be permuted by a ``Sorter``
*before* it reaches the encoder while ``inputs`` stays in its original order.

Nothing about a misalignment is loud: the shapes are right, the values are finite, and the model
trains slightly worse for reasons nobody would trace back here. So these tests check the pairing
directly, by giving each hit an embedding that carries its own coordinates, and then asserting
that what arrives at the encoder still matches.

Two paths have to hold, and they keep the pairing by different mechanisms:

* **no sorter** -- coordinates reach the encoder unsorted, and ``Encoder.forward`` permutes them
  alongside the tokens using ``x_sort_value`` (step 4.6).
* **sorter** -- ``Sorter.sort_inputs`` permutes ``key_eta``/``key_phi`` along with everything
  else, so they are already in token order on arrival.
"""

import pytest
import torch
from torch import Tensor, nn

from hepattn.models import Encoder
from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.maskformer import MaskFormer
from hepattn.utils.sorter import Sorter

DIM = 32
WINDOW = 8
N1, N2 = 10, 15


class _CoordEchoInputNet(nn.Module):
    """Embeds each hit so that channel 0 is its eta and channel 1 is its phi.

    This is what makes misalignment observable. Ordinarily a token embedding says nothing about
    which hit produced it, so "are the coordinates still paired with their tokens" is not a
    question you can ask of the tensors. Copying the coordinates into the embedding makes the
    pairing checkable: at the encoder's door, ``key_embed[..., :2]`` must equal ``x_coords``.
    """

    def __init__(self, input_name: str, dim: int) -> None:
        super().__init__()
        self.input_name = input_name
        self.dim = dim

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        eta = inputs[f"{self.input_name}_eta"]
        phi = inputs[f"{self.input_name}_phi"]
        pad = torch.zeros(*eta.shape, self.dim - 2, device=eta.device, dtype=eta.dtype)
        return torch.cat([eta.unsqueeze(-1), phi.unsqueeze(-1), pad], dim=-1)


class _MockTask(nn.Module):
    def __init__(self, name: str = "test_task") -> None:
        super().__init__()
        self.name = name
        self.outputs = ["output"]
        self.has_intermediate_loss = False
        self.permute_loss = True

    def forward(self, x, outputs=None):
        return {"output": torch.randn(1, 2, N1 + N2)}

    def should_run_at_layer(self, layer_index):
        return True

    def should_permute_outputs(self, layer_name, layer_outputs):
        return False

    def attn_mask(self, outputs):
        return {}


class _MockMatcher(nn.Module):
    def forward(self, costs, target_valid_mask=None, query_valid_mask=None):
        batch_size, num_pred, _ = costs.shape
        return torch.arange(num_pred).unsqueeze(0).expand(batch_size, -1)


def _build_model(sorter: Sorter | None, input_sort_field: str | None) -> MaskFormer:
    # Route B requires flex + a window, and attention.py asserts batch size 1 whenever a
    # BlockMask is present, which is why every tensor here has a leading dim of 1.
    encoder = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=2)
    decoder = MaskFormerDecoder(
        num_queries=2,
        decoder_layer_config={"dim": DIM, "norm": "LayerNorm", "dense_kwargs": {}, "attn_kwargs": {}, "bidirectional_ca": True},
        num_decoder_layers=1,
        mask_attention=False,
    )
    return MaskFormer(
        input_nets=nn.ModuleList([_CoordEchoInputNet("inputone", DIM), _CoordEchoInputNet("inputtwo", DIM)]),
        encoder=encoder,
        decoder=decoder,
        tasks=nn.ModuleList([_MockTask()]),
        dim=DIM,
        matcher=_MockMatcher(),
        input_sort_field=input_sort_field,
        sorter=sorter,
    )


@pytest.fixture
def sample_inputs() -> dict[str, Tensor]:
    torch.manual_seed(0)
    inputs = {}
    for name, n in (("inputone", N1), ("inputtwo", N2)):
        inputs[f"{name}_eta"] = torch.randn(1, n)
        # Deliberately unsorted, and disjoint between the two input types so that a mix-up
        # between them would be visible rather than plausible.
        inputs[f"{name}_phi"] = torch.randperm(n).float().unsqueeze(0) + (0 if name == "inputone" else 100)
        inputs[f"{name}_valid"] = torch.ones(1, n, dtype=torch.bool)
    return inputs


def _capture_encoder_args(model: MaskFormer) -> tuple[dict, object]:
    """Record the tokens and coordinates as they arrive at the encoder.

    A forward pre-hook fires immediately before ``Encoder.forward`` runs, and receives its
    positional and keyword arguments -- which is exactly the moment the pairing must hold, before
    the encoder does anything of its own.
    """
    seen: dict = {}

    def pre_hook(module, args, kwargs):  # noqa: ARG001
        seen["x"] = args[0].detach()
        seen["x_coords"] = None if kwargs["x_coords"] is None else kwargs["x_coords"].detach()
        seen["x_sort_value"] = kwargs["x_sort_value"]

    handle = model.encoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    return seen, handle


def _run(model: MaskFormer, inputs: dict[str, Tensor]) -> dict:
    seen, handle = _capture_encoder_args(model)
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()
    return seen


def _assert_coords_paired_with_tokens(seen: dict) -> None:
    """Channel 0 of every token embedding is that hit's eta, channel 1 its phi."""
    x, coords = seen["x"], seen["x_coords"]
    assert coords is not None, "x_coords never reached the encoder"
    assert coords.shape == (1, N1 + N2, 2), coords.shape
    torch.testing.assert_close(coords[..., 0], x[..., 0], msg="eta is not paired with its token")
    torch.testing.assert_close(coords[..., 1], x[..., 1], msg="phi is not paired with its token")


def test_coords_reach_encoder_paired_without_a_sorter(sample_inputs):
    model = _build_model(sorter=None, input_sort_field="phi")
    seen = _run(model, sample_inputs)

    _assert_coords_paired_with_tokens(seen)

    # On this path the encoder does the sorting itself, so it must have been given a sort value
    # and the coordinates must still be unsorted on arrival.
    assert seen["x_sort_value"] is not None, "encoder was not given a value to sort by"
    phi = seen["x_coords"][0, :, 1]
    assert not bool((phi[1:] >= phi[:-1]).all()), "coords arrived pre-sorted; this path expects them raw"


def test_coords_reach_encoder_paired_with_a_sorter(sample_inputs):
    model = _build_model(sorter=Sorter(input_sort_field="phi"), input_sort_field=None)
    seen = _run(model, sample_inputs)

    _assert_coords_paired_with_tokens(seen)

    # Here the Sorter has already permuted the sequence, so the encoder is given no sort value
    # and the coordinates must arrive already in phi order -- proof they rode along with it.
    assert seen["x_sort_value"] is None, "encoder should not re-sort when a Sorter is active"
    phi = seen["x_coords"][0, :, 1]
    assert bool((phi[1:] >= phi[:-1]).all()), "the sorter did not permute the coordinates"


def test_coords_are_not_built_when_the_feature_is_off(sample_inputs):
    # Everything above is gated on or_n_hashes, so an ordinary model must be untouched: no
    # x_coords, and no stray key_eta/key_phi work done on its behalf.
    model = _build_model(sorter=None, input_sort_field="phi")
    model.encoder = Encoder(num_layers=1, dim=DIM, attn_type="torch", window_size=WINDOW)

    seen = _run(model, sample_inputs)
    assert seen["x_coords"] is None, "coords were built for a model that does not use them"


def test_or_amplification_requires_a_sorted_sequence():
    # The per-head mask is defined over token index, so without a sort the sequence carries no
    # geometric locality, every mask comes out fully dense, and OR amplification does nothing --
    # while still paying to replicate q/k/v across the hashes. Refuse that configuration rather
    # than let it train as a silent no-op.
    with pytest.raises(AssertionError, match="sorted sequence"):
        _build_model(sorter=None, input_sort_field=None)

    # Either way of supplying an order is fine.
    _build_model(sorter=None, input_sort_field="phi")
    _build_model(sorter=Sorter(input_sort_field="phi"), input_sort_field=None)
