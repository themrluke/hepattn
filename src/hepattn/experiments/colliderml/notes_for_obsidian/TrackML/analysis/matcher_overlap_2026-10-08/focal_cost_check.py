"""Single-product focal cost (mask_focal single_product): how do the costs and the matching change? Same forward pass
(bf16 autocast, as in training), both costs from its outputs, all target slots (as without trim_target_costs).

    python focal_cost_check.py CKPT N LABEL
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
    cfg["model"]["name"] = "focal-cost"
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
    better = []
    for i, (inputs, targets) in enumerate(dm.val_dataloader()):
        if i == n_events:
            break
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            prepared, _, decoder_outputs = model._prepare_targets_and_outputs(outputs, targets)
            mask_task = next(t for t in model.tasks if t.name == "track_hit_valid")
            mask_task.cost_kwargs = {}
            full = model._compute_decoder_costs(decoder_outputs, prepared)
            mask_task.cost_kwargs = {"mask_focal": {"single_product": True}}
            single = model._compute_decoder_costs(decoder_outputs, prepared)
            mask_task.cost_kwargs = {}
            # The focal part in float64: the exact cost, to see which version's matching is really closer to optimal
            from hepattn.models.loss import mask_focal_cost
            weight = mask_task.costs["mask_focal"]
            exact = {}
            for name, layer_outputs in decoder_outputs.items():
                if full[name] is None or "track_hit_valid" not in layer_outputs:
                    exact[name] = None
                    continue
                logit = layer_outputs["track_hit_valid"]["track_hit_logit"].detach()
                tgt = prepared["particle_hit_valid"]
                pad = prepared["hit_valid"]
                f32 = weight * mask_focal_cost(logit.float(), tgt.float(), input_pad_mask=pad)
                f64 = weight * mask_focal_cost(logit.double(), tgt.double(), input_pad_mask=pad)
                exact[name] = full[name].double() - f32.double() + f64
        n_real = int(prepared["particle_valid"].sum())
        trimmed = {k: v[..., :n_real] for k, v in single.items()}
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
            ex = exact[name][0].double()
            e_full = ex[idx_full.to(ex.device), parts].sum().item()
            e_single = ex[idx_trim.to(ex.device), parts].sum().item()
            better.append(e_single - e_full)
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
          f"total cost of the single-product matching vs the optimum (on the two-product costs): max {st[:, 1].max():+.1e} relative")
    b = np.array(better)
    print(f"{label}: scored on the exact (float64) cost, the single-product matching is lower in {(b < 0).sum()}, equal in {(b == 0).sum()}, "
          f"higher in {(b > 0).sum()} of {len(b)} layer matchings; sum of differences {b.sum():+.3e}")
