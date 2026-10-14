"""Which source lines does the GPU time come from? Forward + loss + backward with the compiled model (bench
overlaptrimrows config, late paper-run weights), profiled with Python stacks; GPU time grouped by call site."""
import sys
import tempfile

import torch
import yaml

if __name__ == "__main__":
    ckpt, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, "/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml")
    from lightning.pytorch.cli import LightningCLI

    from hepattn.experiments.trackml.data import TrackMLDataModule
    from run_tracking import TrackMLTracker

    cfg = yaml.safe_load(open("configs/overlaptrimrows.yaml"))
    for key in ("ckpt_path", "name", "matmul_precision"):
        cfg.pop(key, None)
    cfg["model"]["name"] = "profile-stack"
    cfg["trainer"].update(logger=False, callbacks=[])
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, f)
    f.close()
    torch.set_float32_matmul_precision("high")
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    cli = LightningCLI(model_class=TrackMLTracker, datamodule_class=TrackMLDataModule, args=["--config", f.name], run=False)
    module, dm = cli.model, cli.datamodule
    module.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    module.cuda().train()
    for name in ("encoder", "decoder"):
        getattr(module.model, name).compile(dynamic=True)
    dm.setup("fit")

    def step(inputs, targets):
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = module._forward_for_loss(inputs, targets)
            outputs, targets, losses = module.model.loss(outputs, targets)
        total = sum(v for layer in losses.values() for task in layer.values() for v in task.values())
        total.backward()
        module.zero_grad(set_to_none=True)

    it = iter(dm.train_dataloader())
    for _ in range(10):  # compile and settle
        step(*next(it))
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True) as prof:
        for _ in range(5):
            step(*next(it))
        torch.cuda.synchronize()
    import collections

    # GPU time of every eager op, by the innermost hepattn source line that called it (compiled graphs are their own group)
    by_site = collections.defaultdict(lambda: [0.0, 0, collections.Counter()])
    for e in prof.events():
        dev = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if dev <= 0 or e.device_type == torch.autograd.DeviceType.CUDA:
            continue
        site = next((fr for fr in (e.stack or []) if "hepattn" in fr and "profile_stack" not in fr), None)
        if e.name.startswith("## Call CompiledFxGraph"):
            site = "compiled encoder/decoder graphs (forward and backward)"
        site = site or f"(no hepattn frame: {e.name})"
        by_site[site][0] += dev
        by_site[site][1] += 1
        by_site[site][2][e.name] += dev
    total = sum(v[0] for v in by_site.values())
    with open(out, "w") as fh:
        fh.write(f"GPU time per step by call site (5 steps; total {total / 5 / 1e3:.1f} ms/step)\n\n")
        for site, (dev, n, names) in sorted(by_site.items(), key=lambda x: -x[1][0])[:40]:
            top = ", ".join(f"{k} {v / 5 / 1e3:.1f}" for k, v in names.most_common(3))
            fh.write(f"{dev / 5 / 1e3:7.2f} ms/step  {site}\n          {top}\n")
