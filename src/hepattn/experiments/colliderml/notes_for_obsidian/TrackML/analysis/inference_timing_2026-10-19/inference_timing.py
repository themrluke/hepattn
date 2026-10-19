"""Inference timing study (one repeat per process): encoder-only time vs window, flex BlockMask block size and depth,
for the flex sliding-window encoder and OR; and the whole trained model's forward pass, split into encoder and
decoder. Inference as in testing: eval mode, no grad, bf16 autocast, encoder and decoder compiled.

    python inference_timing.py REPEAT OUT.json
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from timing_common import CONFIGS, build_encoder, capture_encoder_inputs, load_module, run_encoder, set_block_size, test_events

LOGS = "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs"
# Each event has a different number of hits; with the default limit of 8 recompilations torch.compile would give up
# after 8 sizes and run the encoder uncompiled (flex then builds the full N x N matrix). Warm-up also runs every timed
# event once, so no compilation happens inside the timed loop; recompilations there are counted and recorded anyway.
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 4096
N_WARM, N_TIME = 10, 40
WINDOWS = (512, 256, 128, 64, 32)
BLOCKS = (128, 64, 32, 16)
DEPTHS = (2, 4)
# Trained models at their own window: (label, config, checkpoint)
MODELS = [
    ("flex w512", "tracking-eta4-pt600-flex.yaml", "TRK-Pix0.6-DQMA-flex_20261005-T152222/ckpts/epoch=029-val_loss=0.32403.ckpt"),
    ("OR w512", "tracking-eta4-pt600-or.yaml", "TRK-Pix0.6-DQMA-OR_20261005-T152222/ckpts/epoch=028-val_loss=0.30994.ckpt"),
    ("flex w64", "tracking-eta4-pt600-flex-w64.yaml", "TRK-Pix0.6-DQMA-flex-w64-s42_20261014-T144700/ckpts/epoch=028-val_loss=0.47998.ckpt"),
    ("OR w64", "tracking-eta4-pt600-or-w64.yaml", "TRK-Pix0.6-DQMA-OR-w64-s42_20261014-T144902/ckpts/epoch=028-val_loss=0.28729.ckpt"),
    ("flex w32", "tracking-eta4-pt600-flex-w32.yaml", "TRK-Pix0.6-DQMA-flex-w32-s42_20261016-T095008/ckpts/epoch=029-val_loss=0.61979.ckpt"),
    ("OR w32", "tracking-eta4-pt600-or-w32.yaml", "TRK-Pix0.6-DQMA-OR-w32-s42_20261016-T095210/ckpts/epoch=028-val_loss=0.29443.ckpt"),
]


def gpu_state():
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()
    return {"util_mem": q, "gpu_processes": len(apps), "load_avg": os.getloadavg()[0]}


def summary(ms):
    ms = np.array(ms)
    return {"mean": float(ms.mean()), "min": float(ms.min()), "median": float(np.median(ms)), "per_event": ms.tolist()}


def compiles():
    from torch._dynamo.utils import counters

    return sum(counters["stats"].values()) + sum(counters["recompiles"].values()) if "recompiles" in counters else sum(counters["stats"].values())


def time_calls(fn, items):
    """Per-item GPU time in ms (CUDA events). Warm-up runs N_WARM extra items and then every timed item once."""
    for it in items[: N_WARM + N_TIME]:
        fn(it)
    torch.cuda.synchronize()
    time_calls.compiles_before = compiles()
    out = []
    for it in items[N_WARM : N_WARM + N_TIME]:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(it)
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e))
    time_calls.compiles_during = compiles() - time_calls.compiles_before
    return out


def encoder_study(inputs, results):
    configs = [(k, w, b, 8) for k in ("flex", "or") for w in WINDOWS for b in BLOCKS]
    configs += [(k, w, 128, d) for k in ("flex", "or") for w in (512, 64) for d in DEPTHS]
    for kind, window, block, depth in configs:
        key = f"encoder {kind} w{window} block{block} depth{depth}"
        t0 = time.time()
        try:
            set_block_size(block)
            enc = build_encoder(kind, window, depth)
            enc.compile(dynamic=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ms = time_calls(lambda inp: run_encoder(enc, inp), inputs)
            results[key] = {**summary(ms), "gpu": gpu_state(), "wall_s": time.time() - t0, "compiles_during_timing": time_calls.compiles_during}
            print(f"{key}: mean {results[key]['mean']:.2f} ms, min {results[key]['min']:.2f} ms, compiles while timing {time_calls.compiles_during}", flush=True)
        except Exception as e:  # noqa: BLE001
            results[key] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"{key}: FAILED {results[key]['error']}", flush=True)
        del enc
        torch.cuda.empty_cache()


def model_study(events, results):
    set_block_size(128)
    for label, config, ckpt in MODELS:
        key = f"model {label}"
        try:
            module, _ = load_module(str(CONFIGS / config), f"{LOGS}/{ckpt}")
            model = module.model
            for name in ("encoder", "decoder"):  # as the Compile callback does for testing
                getattr(model, name).compile(dynamic=True)
            marks = {}

            # The hooks must run eagerly: on a compiled module they would otherwise be traced into the graph and
            # the CUDA events never recorded ("Both events must be recorded before calculating elapsed time")
            def pre(name):
                return torch.compiler.disable(lambda *_a, **_k: marks.__setitem__(name, torch.cuda.Event(enable_timing=True)) or marks[name].record())

            def post(name):
                return torch.compiler.disable(lambda *_a, **_k: marks.__setitem__(name + "_end", torch.cuda.Event(enable_timing=True)) or marks[name + "_end"].record())

            hooks = [model.encoder.register_forward_pre_hook(pre("enc")), model.encoder.register_forward_hook(post("enc")),
                     model.decoder.register_forward_pre_hook(pre("dec")), model.decoder.register_forward_hook(post("dec"))]
            parts = {"enc": [], "dec": []}

            def call(ev):
                model(ev[0])

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for ev in events[: N_WARM + N_TIME]:  # warm-up, including every timed event once
                    call(ev)
                torch.cuda.synchronize()
                c0 = compiles()
                total = []
                n_queries = []
                for ev in events[N_WARM : N_WARM + N_TIME]:
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    out = model(ev[0])
                    e.record()
                    e.synchronize()
                    total.append(s.elapsed_time(e))
                    parts["enc"].append(marks["enc"].elapsed_time(marks["enc_end"]))
                    parts["dec"].append(marks["dec"].elapsed_time(marks["dec_end"]))
                    qm = out.get("encoder", {}).get("query_mask")
                    n_queries.append(int(qm.sum()) if qm is not None else -1)
            for h in hooks:
                h.remove()
            results[key] = {"total": summary(total), "encoder": summary(parts["enc"]), "decoder": summary(parts["dec"]),
                            "mean_queries": float(np.mean(n_queries)), "gpu": gpu_state(), "compiles_during_timing": compiles() - c0}
            r = results[key]
            print(f"{key}: total mean {r['total']['mean']:.1f} ms (min {r['total']['min']:.1f}); encoder {r['encoder']['mean']:.1f}, decoder {r['decoder']['mean']:.1f}; "
                  f"{r['mean_queries']:.0f} queries", flush=True)
            del module, model
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            results[key] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"{key}: FAILED {results[key]['error']}", flush=True)


if __name__ == "__main__":
    repeat, out = int(sys.argv[1]), sys.argv[2]
    models_only = len(sys.argv) > 3 and sys.argv[3] == "models"  # rerun only the trained-model part, into the same file
    results = json.load(open(out)) if models_only else {"repeat": repeat, "start": time.strftime("%F %T"), "gpu_start": gpu_state(), "torch": torch.__version__}
    results["models_start"] = time.strftime("%F %T")
    module, dm = load_module(str(CONFIGS / "tracking-eta4-pt600-or.yaml"), f"{LOGS}/{MODELS[1][2]}")
    events = test_events(dm, N_WARM + N_TIME)
    if not models_only:
        inputs = capture_encoder_inputs(module, events)
        results["hits_per_event"] = [int(i["x"].shape[1]) for i in inputs]
    del module
    torch.cuda.empty_cache()
    if not models_only:
        encoder_study(inputs, results)
        json.dump(results, open(out, "w"))
    model_study(events, results)
    results["end"] = time.strftime("%F %T")
    results["gpu_end"] = gpu_state()
    json.dump(results, open(out, "w"))
