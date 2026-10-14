"""Real-model identity check for overlap_layers: same weights, same events, training-mode forward + loss + backward
with the compiled encoder/decoder as in training. Saves per-layer matchings, every loss term and a hash of every
gradient, for compare.py.

    PYTHONPATH=<shim> HEPATTN_SRC=/shared/projects/hepattn-overlap/src python identity.py CKPT N MODE OUT.pt

MODE: on / off / trimon / trimoff / rows... (8 Oct runs), or "+"-joined options (14 Oct): base, overlap, trim, rows,
perm (fast_permute), slice (slice_valid_targets), focal1 (single-product focal cost), scipy, det (deterministic
algorithms, to try to make the backward repeatable), eager (no torch.compile).
"""
import hashlib
import sys
import tempfile

import torch
import yaml

if __name__ == "__main__":
    ckpt, n_events, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    sys.path.insert(0, "/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml")
    from lightning.pytorch.cli import LightningCLI

    from hepattn.experiments.trackml.data import TrackMLDataModule
    from hepattn.models import maskformer
    from hepattn.models import matcher as matcher_module
    from run_tracking import TrackMLTracker

    assert "hepattn-overlap" in matcher_module.__file__, matcher_module.__file__

    cfg = yaml.safe_load(open("/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml/configs/tracking-eta4-pt600-flex.yaml"))
    for key in ("ckpt_path", "name", "matmul_precision"):
        cfg.pop(key, None)
    cfg["model"]["name"] = "identity"  # normally linked from the top-level name by the trackml CLI
    cfg["trainer"]["logger"] = False
    cfg["trainer"]["callbacks"] = []
    m = cfg["model"]["model"]["init_args"]["matcher"]["init_args"]
    if "+" in mode or mode == "base" or mode.startswith("base"):
        opts = set(mode.split("+")) - {"base"}
        overlap, trim, rows = "overlap" in opts, "trim" in opts, "rows" in opts
    else:
        # 8 Oct names: on / off (overlap with toggle 1), trimon / trimoff (with toggle 2), "rows" prefix: trim_target_costs
        opts = set()
        overlap, trim, rows = mode.endswith("on"), "trim" in mode, mode.startswith("rows")
    init = cfg["model"]["model"]["init_args"]
    init["trim_target_costs"] = rows
    init["fast_permute"] = "perm" in opts
    init["slice_valid_targets"] = "slice" in opts
    if "focal1" in opts:
        for task in init["tasks"]["init_args"]["modules"]:
            if task["init_args"].get("name") == "track_hit_valid":
                task["init_args"]["cost_kwargs"] = {"mask_focal": {"single_product": True}}
    if "det" in opts:
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)  # strict: memory-efficient attention is only deterministic then
    m.update(default_solver="scipy" if "scipy" in opts else "lap1015_late", adaptive_solver=False, prepare_on_device=True, trim_padded_queries=trim, overlap_layers=overlap)
    cfg_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, cfg_file)
    cfg_file.close()

    torch.set_float32_matmul_precision("high")
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True  # as the Compile callback
    cli = LightningCLI(model_class=TrackMLTracker, datamodule_class=TrackMLDataModule, args=["--config", cfg_file.name], run=False)
    module, dm = cli.model, cli.datamodule
    module.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    module.cuda().train()
    assert module.model.matches_during_forward == overlap
    if "eager" not in opts:
        for name in ("encoder", "decoder"):  # as the Compile callback
            getattr(module.model, name).compile(dynamic=True)
    dm.setup("fit")

    matchings = []
    orig_permute = maskformer.MaskFormer._permute_outputs

    def recording_permute(self, decoder_outputs, layer_names, layer_pred_idxs):
        matchings.append({name: idx.clone() for name, idx in zip(layer_names, layer_pred_idxs, strict=True)})
        return orig_permute(self, decoder_outputs, layer_names, layer_pred_idxs)

    maskformer.MaskFormer._permute_outputs = recording_permute

    def to_cuda(x):
        if isinstance(x, dict):
            return {k: to_cuda(v) for k, v in x.items()}
        return x.cuda() if torch.is_tensor(x) else x

    def leaves(losses, prefix=""):
        if isinstance(losses, dict):
            for k in sorted(losses):
                yield from leaves(losses[k], f"{prefix}/{k}")
        else:
            yield prefix, losses

    results = []
    for i, (inputs, targets) in enumerate(dm.val_dataloader()):
        if i == n_events:
            break
        inputs, targets = to_cuda(inputs), to_cuda(targets)
        torch.manual_seed(i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = module._forward_for_loss(inputs, targets)
            outputs, _, losses = module.model.loss(outputs, targets)
        terms = dict(leaves(losses))
        total = sum(terms.values())
        total.backward()
        grads = {n: hashlib.sha1(p.grad.detach().float().cpu().numpy().tobytes()).hexdigest() for n, p in module.named_parameters() if p.grad is not None}
        # The backward pass is not bitwise repeatable (two identical runs differ), so keep the values to measure by how much
        grad_values = torch.cat([p.grad.detach().float().flatten().cpu() for _, p in sorted(module.named_parameters()) if p.grad is not None])
        module.zero_grad(set_to_none=True)
        results.append({"losses": {k: v.detach().double().item() for k, v in terms.items()}, "grads": grads, "grad_values": grad_values, "matching": matchings[-1]})
        print(f"event {i}: loss {total.item():.6f}, {len(matchings[-1])} layers matched", flush=True)
    torch.save(results, out)
