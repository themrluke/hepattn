"""trim_target_costs: are the costs computed for the real particles only bitwise equal to those columns of the full
costs, and is the matching the same? Same forward pass (bf16 autocast, as in training), both costs from its outputs.

    python cost_rows_check.py CKPT N LABEL
"""
import sys
import tempfile

import torch
import yaml

if __name__ == "__main__":
    ckpt, n_events, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, "/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml")
    from lightning.pytorch.cli import LightningCLI

    from hepattn.experiments.trackml.data import TrackMLDataModule
    from hepattn.models import matcher as matcher_module
    from run_tracking import TrackMLTracker

    assert "hepattn-overlap" in matcher_module.__file__
    cfg = yaml.safe_load(open("/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml/configs/tracking-eta4-pt600-flex.yaml"))
    for key in ("ckpt_path", "name", "matmul_precision"):
        cfg.pop(key, None)
    cfg["model"]["name"] = "cost-rows"
    cfg["trainer"].update(logger=False, callbacks=[])
    cfg["model"]["model"]["init_args"]["matcher"]["init_args"].update(prepare_on_device=True)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, f)
    f.close()
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(model_class=TrackMLTracker, datamodule_class=TrackMLDataModule, args=["--config", f.name], run=False)
    module, dm = cli.model, cli.datamodule
    module.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    module.cuda().train()
    model = module.model
    dm.setup("fit")

    n_same_cost = n_same_match = n_layers = 0
    stats = []
    for i, (inputs, targets) in enumerate(dm.val_dataloader()):
        if i == n_events:
            break
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            prepared, _, decoder_outputs = model._prepare_targets_and_outputs(outputs, targets)
            model.trim_target_costs = False
            full = model._compute_decoder_costs(decoder_outputs, prepared)
            model.trim_target_costs = True
            trimmed = model._compute_decoder_costs(decoder_outputs, prepared)
        n_real = int(prepared["particle_valid"].sum())
        line = []
        for name in full:
            f_cost, t_cost = full[name], trimmed[name]
            same_cost = t_cost.shape[-1] == n_real and torch.equal(f_cost[..., :n_real], t_cost)
            valid, qmask = prepared["particle_valid"], prepared["query_mask"]
            idx_full = model.matcher(f_cost, valid, qmask)[0, :n_real].long()
            idx_trim = model.matcher(t_cost, valid[:, :n_real], qmask)[0, :n_real].long()
            same_match = torch.equal(idx_full, idx_trim)
            real = f_cost[0, :, :n_real][qmask[0]].float()  # valid queries x real particles
            rel = ((real - t_cost[0][qmask[0]].float()).abs() / real.abs().clamp_min(1e-6)).max().item()
            max_diff = rel
            # Both matchings scored on the full costs: is the trimmed one just another (near-)optimal assignment?
            fc = f_cost[0].float()
            parts = torch.arange(n_real, device=fc.device)
            gap = (fc[idx_trim.to(fc.device), parts].sum() - fc[idx_full.to(fc.device), parts].sum()).item()
            total = fc[idx_full.to(fc.device), parts].sum().item()
            changed = (idx_full != idx_trim).float().mean().item()
            stats.append((changed, gap / abs(total), rel))
            n_same_cost += same_cost
            n_same_match += same_match
            n_layers += 1
            line.append(f"{name}: {changed:.2%} particles re-paired")
        print(f"event {i} ({n_real} particles): " + "; ".join(line), flush=True)
    import numpy as np
    st = np.array(stats)
    print(f"{label}: {n_layers} layer costs, bitwise identical {n_same_cost}, identical matching {n_same_match}; "
          f"max relative cost difference {st[:, 2].max():.1e}; particles re-paired: mean {st[:, 0].mean():.3%}, max {st[:, 0].max():.3%}; "
          f"total cost of the trimmed matching vs the optimum (on the full costs): max {st[:, 1].max():+.1e} relative")
