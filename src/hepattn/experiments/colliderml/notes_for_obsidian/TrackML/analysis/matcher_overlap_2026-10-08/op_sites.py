"""GPU time of selected ops by the hepattn line that called them: a dispatch mode times each op with CUDA events and
records the innermost hepattn frame. Forward + loss + backward, eager (no compile), 2 steps after 3 warm-up steps.
The backward ops have no Python frame, so they are grouped by op only."""
import collections
import sys
import tempfile
import traceback

import torch
import yaml
from torch.utils._python_dispatch import TorchDispatchMode

OPS = ("nonzero", "index.Tensor", "index_put", "_index_put_impl", "mm", "bmm", "masked_select", "copy_", "_efficient_attention", "item", "_local_scalar_dense", "index_select", "gather", "scatter")


class Timer(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = collections.defaultdict(lambda: [0.0, 0])
        self.pending = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.overloadpacket.__name__) + ("." + func._overloadname if func._overloadname != "default" else "")
        if not any(o in name for o in OPS):
            return func(*args, **(kwargs or {}))
        frames = [f for f in traceback.extract_stack() if "/hepattn/" in f.filename and "op_sites" not in f.filename]
        site = f"{frames[-1].filename.split('/hepattn/')[-1]}:{frames[-1].lineno} ({frames[-1].name})" if frames else "(backward / no frame)"
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = func(*args, **(kwargs or {}))
        e.record()
        self.pending.append(((name, site), s, e))
        return out

    def collect(self):
        torch.cuda.synchronize()
        for key, s, e in self.pending:
            self.rows[key][0] += s.elapsed_time(e)
            self.rows[key][1] += 1
        self.pending.clear()


if __name__ == "__main__":
    ckpt, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, "/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml")
    from lightning.pytorch.cli import LightningCLI

    from hepattn.experiments.trackml.data import TrackMLDataModule
    from run_tracking import TrackMLTracker

    cfg = yaml.safe_load(open("configs/overlaptrimrows.yaml"))
    for key in ("ckpt_path", "name", "matmul_precision"):
        cfg.pop(key, None)
    cfg["model"]["name"] = "op-sites"
    cfg["trainer"].update(logger=False, callbacks=[])
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, f)
    f.close()
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(model_class=TrackMLTracker, datamodule_class=TrackMLDataModule, args=["--config", f.name], run=False)
    module, dm = cli.model, cli.datamodule
    module.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    module.cuda().train()
    dm.setup("fit")
    it = iter(dm.train_dataloader())

    def step(timer=None):
        inputs, targets = next(it)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = module._forward_for_loss(inputs, targets)
            outputs, targets, losses = module.model.loss(outputs, targets)
        total = sum(v for layer in losses.values() for task in layer.values() for v in task.values())
        total.backward()
        module.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    timer = Timer()
    n = 2
    for _ in range(n):
        with timer:
            step()
        timer.collect()
    with open(out, "w") as fh:
        fh.write(f"GPU time of selected ops per step, by calling hepattn line (eager, {n} steps)\n\n")
        for (name, site), (ms, calls) in sorted(timer.rows.items(), key=lambda x: -x[1][0])[:45]:
            fh.write(f"{ms / n:8.2f} ms/step  {calls // n:5d} calls  {name:28s} {site}\n")
