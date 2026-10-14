"""Is lap1015_late always faster than scipy on real cost matrices? (decides whether the adaptive solver can go)

Same matrices the solver gets in training: masked padded queries, real-particle rows, [true, pred].
    python solver_timing.py CKPT N_EVENTS LABEL
"""
import sys
import time

import numpy as np
import torch
import yaml

if __name__ == "__main__":
    ckpt, n_events, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, "/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml")
    import tempfile

    from lightning.pytorch.cli import LightningCLI

    from hepattn.experiments.trackml.data import TrackMLDataModule
    from hepattn.models.matcher import SOLVERS, match_individual
    from run_tracking import TrackMLTracker

    cfg = yaml.safe_load(open("/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml/configs/tracking-eta4-pt600-flex.yaml"))
    for key in ("ckpt_path", "name", "matmul_precision"):
        cfg.pop(key, None)
    cfg["model"]["name"] = "solver-timing"
    cfg["trainer"].update(logger=False, callbacks=[])
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, f)
    f.close()
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(model_class=TrackMLTracker, datamodule_class=TrackMLDataModule, args=["--config", f.name], run=False)
    module, dm = cli.model, cli.datamodule
    module.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    module.cuda().eval()
    dm.setup("fit")

    times = {s: [] for s in SOLVERS}
    for i, (inputs, targets) in enumerate(dm.val_dataloader()):
        if i == n_events:
            break
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = module.model(inputs)
            targets, _, decoder_outputs = module.model._prepare_targets_and_outputs(outputs, targets)
            costs = module.model._compute_decoder_costs(decoder_outputs, targets)
        # What the adaptive check times (adapt_solver: compute_matching on all layers, no masks), and what training
        # then runs (toggle-1 path: masks, 4 layers in parallel)
        from hepattn.models.matcher import Matcher
        stacked_costs = torch.stack([c for c in costs.values()]).flatten(0, 1).float()
        valid = targets["particle_valid"].repeat(len(costs), 1)
        qvalid = targets["query_mask"].repeat(len(costs), 1)
        for s in SOLVERS:
            m = Matcher(default_solver=s, adaptive_solver=False, parallel_solver=True, parallel_backend="process", n_jobs=4, prepare_on_device=True)
            m(stacked_costs, valid, qvalid)  # warm the pool
            t = time.perf_counter(); m.compute_matching(stacked_costs.cpu().numpy()); t_check = time.perf_counter() - t
            t = time.perf_counter(); m(stacked_costs, valid, qvalid); t_train = time.perf_counter() - t
            times.setdefault(f"{s}_check", []).append(t_check)
            times.setdefault(f"{s}_train", []).append(t_train)
        print(f"  adaptive check (unmasked 4x{stacked_costs.shape[1]}x{stacked_costs.shape[2]}): " + ", ".join(f"{s} {times[f'{s}_check'][-1]:.3f} s" for s in SOLVERS)
              + " | training call (masked, parallel): " + ", ".join(f"{s} {times[f'{s}_train'][-1]:.3f} s" for s in SOLVERS), flush=True)
        n_true = int(targets["particle_valid"].sum())
        qmask = targets["query_mask"][0]
        for name, c in costs.items():
            c = c[0].float().masked_fill(~qmask.unsqueeze(-1), float(np.finfo(np.float32).max / 10))
            cost = np.ascontiguousarray(c.T[:n_true].cpu().numpy())
            idx = np.arange(cost.shape[1], dtype=np.int32)
            for s in SOLVERS:
                t = time.perf_counter()
                match_individual(SOLVERS[s], cost, idx)
                times[s].append(time.perf_counter() - t)
        print(f"event {i}: {n_true} particles, {int(qmask.sum())} valid queries; single solves " + ", ".join(f"{s} {np.mean(times[s][-4:]):.3f} s" for s in SOLVERS), flush=True)
    for kind in ("check", "train"):
        r = np.array(times[f"scipy_{kind}"]) / np.array(times[f"lap1015_late_{kind}"])
        print(f"{label} {kind}: scipy / lap1015_late time ratio: min {r.min():.2f}, median {np.median(r):.2f}, max {r.max():.2f}")
    ratio = np.array(times["scipy"]) / np.array(times["lap1015_late"])
    print(f"{label}: {len(ratio)} solves; scipy / lap1015_late time ratio: min {ratio.min():.1f}, median {np.median(ratio):.1f}, max {ratio.max():.1f}")
