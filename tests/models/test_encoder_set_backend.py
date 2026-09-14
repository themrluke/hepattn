"""``Encoder.set_backend`` — switching attention backend after construction.

Used to evaluate a model with a different kernel from the one it trained with. Two things must
survive the switch, and neither did before: the sliding window, and the refusal to carry an
OR-amplification model onto a backend that cannot express its mask.

Both failures are quiet. Nothing raises; you just benchmark or evaluate a different model from the
one you meant to.
"""

import pytest

from hepattn.models import Encoder

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
