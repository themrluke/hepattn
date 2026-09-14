"""``Encoder.set_backend`` — switching attention backend after construction.

Used to evaluate a model with a different kernel from the one it trained with. Two things must
survive the switch, and neither did before: the sliding window, and the refusal to carry an
OR-amplification model onto a backend that cannot express its mask.

Both failures are quiet. Nothing raises; you just benchmark or evaluate a different model from the
one you meant to.
"""

import pytest

from hepattn.models import Encoder
from hepattn.models.attention import Attention


def _is_compiled(attn: Attention) -> bool:
    """torch.compile wraps the callable; the original is kept on _torchdynamo_orig_callable."""
    return hasattr(attn.attn, "_torchdynamo_orig_callable")


DIM = 32
WINDOW = 64


def test_switching_backend_keeps_the_sliding_window():
    # Attention.set_backend takes window_size=None by default, which resets the flash window to
    # (-1, -1) -- i.e. unwindowed. Encoder.set_backend has to pass its own window back in, or a
    # model flipped to flash at eval time silently attends globally.
    encoder = Encoder(num_layers=2, dim=DIM, attn_type="torch", window_size=WINDOW)
    encoder.set_backend("flash")

    got = [layer.attn.fn.window_size for layer in encoder.layers]
    assert all(w == (WINDOW // 2, WINDOW // 2) for w in got), got


def test_switching_backend_without_a_window_is_unchanged():
    # (-1, -1) is flash's "no window" sentinel and must stay reachable.
    encoder = Encoder(num_layers=1, dim=DIM, attn_type="torch")
    encoder.set_backend("flash")
    assert encoder.layers[0].attn.fn.window_size == (-1, -1)


@pytest.mark.parametrize("backend", ["torch", "flash", "flash-varlen"])
def test_or_amplification_cannot_follow_the_model_to_another_backend(backend):
    # Route B expresses its window in a flex BlockMask. On any other backend `forward` would still
    # build that mask and hand it to a kernel that rejects it, so refuse the switch up front.
    encoder = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=3)
    with pytest.raises(AssertionError, match="flex"):
        encoder.set_backend(backend)


def test_or_amplification_may_switch_to_flex():
    encoder = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=3)
    encoder.set_backend("flex")
    assert encoder.attn_type == "flex"


def test_plain_encoder_still_switches_freely():
    encoder = Encoder(num_layers=1, dim=DIM, attn_type="torch", window_size=WINDOW)
    for backend in ("flash", "flex", "torch"):
        encoder.set_backend(backend)
    assert encoder.attn_type == "torch"


def test_flex_is_compiled_by_default_and_other_backends_are_not():
    # flex without torch.compile falls back to materialising the whole N x N score matrix, which
    # throws away the block-sparsity saving and OOMs at realistic N. Measured at N=4096 with 24
    # mask rows: 44.5 ms / 8.5 GB uncompiled against 11.2 ms / 0.11 GB compiled.
    assert _is_compiled(Attention(DIM, attn_type="flex"))
    for backend in ("torch", "flash", "flash-varlen"):
        assert not _is_compiled(Attention(DIM, attn_type=backend)), backend


def test_explicit_torch_compile_still_wins():
    assert not _is_compiled(Attention(DIM, attn_type="flex", torch_compile=False))
    assert _is_compiled(Attention(DIM, attn_type="torch", torch_compile=True))


def test_compilation_preference_survives_a_backend_switch():
    # Same class of bug as the window: set_backend used to default the flag to False, so an
    # explicitly compiled attention silently became uncompiled after switching.
    attn = Attention(DIM, attn_type="torch", torch_compile=True)
    attn.set_backend("torch")
    assert _is_compiled(attn)

    auto = Attention(DIM, attn_type="torch")
    auto.set_backend("flex")
    assert _is_compiled(auto)
    auto.set_backend("torch")
    assert not _is_compiled(auto)
