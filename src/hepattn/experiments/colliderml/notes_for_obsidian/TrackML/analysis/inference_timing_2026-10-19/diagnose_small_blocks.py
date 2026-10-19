"""Why are flex BlockMask blocks of 32 and 16 so slow in the compiled encoder? Times (a) building the BlockMask alone and
(b) the whole flex encoder at window 64, compiled and uncompiled, for block sizes 128 / 64 / 32 / 16 (4 events)."""
import functools
import time

import torch
from timing_common import CONFIGS, build_encoder, capture_encoder_inputs, load_module, run_encoder, set_block_size, test_events
from torch.nn.attention.flex_attention import create_block_mask

from hepattn.models.encoder import sliding_window_mask_wrapped

torch._dynamo.config.cache_size_limit = 256
LOGS = "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs"
module, dm = load_module(str(CONFIGS / "tracking-eta4-pt600-or.yaml"), f"{LOGS}/TRK-Pix0.6-DQMA-OR_20261005-T152222/ckpts/epoch=028-val_loss=0.30994.ckpt")
inputs = capture_encoder_inputs(module, test_events(dm, 4))
del module


def gpu_ms(fn, n=3):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


N = inputs[0]["x"].shape[1]
seq = torch.tensor([N], device="cuda")
mask_mod = sliding_window_mask_wrapped(64, seq)
for bs in (128, 64, 32, 16):
    cbm = torch.compile(functools.partial(create_block_mask, BLOCK_SIZE=bs), dynamic=True)
    ms = gpu_ms(lambda: cbm(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device="cuda"))
    print(f"BlockMask alone, N={N}, block {bs:3d}: {ms:7.2f} ms", flush=True)
for compiled in (False, True):
    for bs in (128, 64, 32, 16):
        set_block_size(bs)
        enc = build_encoder("flex", 64)
        if compiled:
            enc.compile(dynamic=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ms = gpu_ms(lambda: [run_encoder(enc, i) for i in inputs]) / len(inputs)
        print(f"flex encoder w64, {'compiled' if compiled else 'uncompiled'} module, block {bs:3d}: {ms:7.2f} ms/event", flush=True)
