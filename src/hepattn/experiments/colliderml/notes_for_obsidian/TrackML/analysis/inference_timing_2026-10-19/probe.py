"""Probe: do flex BlockMask block sizes 128/64/32/16 run, and give the same encoder output, at window 64?"""
import time

import torch
from timing_common import build_encoder, capture_encoder_inputs, load_module, run_encoder, set_block_size, test_events

LOGS = "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs"
module, dm = load_module(str(__import__("timing_common").CONFIGS / "tracking-eta4-pt600-or.yaml"), f"{LOGS}/TRK-Pix0.6-DQMA-OR_20261005-T152222/ckpts/epoch=028-val_loss=0.30994.ckpt")
inputs = capture_encoder_inputs(module, test_events(dm, 4))
print("captured", len(inputs), "events; keys", list(inputs[0]), "; hits", [i["x"].shape[1] for i in inputs], flush=True)
for kind in ("flex", "or"):
    ref = None
    for bs in (128, 64, 32, 16):
        set_block_size(bs)
        enc = build_encoder(kind, 64)
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for inp in inputs:  # warm up / compile
                    run_encoder(enc, inp)
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(5):
                    for inp in inputs:
                        out = run_encoder(enc, inp)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) / (5 * len(inputs)) * 1e3
            if ref is None:
                ref = out.float()
            rel = ((out.float() - ref).norm() / ref.norm()).item()
            print(f"{kind:4s} w64 block {bs:3d}: {ms:6.2f} ms/event, output vs block 128: rel diff {rel:.2e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{kind:4s} w64 block {bs:3d}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
