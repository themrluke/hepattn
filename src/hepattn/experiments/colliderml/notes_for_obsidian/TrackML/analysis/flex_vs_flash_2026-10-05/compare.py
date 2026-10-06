# Same weights, same events: encoder output and loss with flash vs flex attention (and optionally other configs).
import sys
import torch

sys.path.insert(0, "/shared/projects/hepattn/src/hepattn/experiments/trackml")
from run_tracking import TrackMLTracker
from hepattn.experiments.trackml.data import TrackMLDataModule
from lightning.pytorch.cli import LightningCLI as CLI

CKPT = sys.argv[1]
N = int(sys.argv[2])
CONFIGS = sys.argv[3:]


def to_cuda(x, fp32=False):
    if isinstance(x, dict):
        return {k: to_cuda(v, fp32) for k, v in x.items()}
    if torch.is_tensor(x):
        x = x.cuda()
        return x.float() if fp32 and x.is_floating_point() else x
    return x


def total(losses):
    if isinstance(losses, dict):
        return sum(total(v) for v in losses.values())
    return float(losses)


def run(spec):
    cfg, _, prec = spec.partition(":")   # "flex.yaml:fp32" runs without bf16 autocast
    cli = CLI(model_class=TrackMLTracker, datamodule_class=TrackMLDataModule, args=["--config", cfg], run=False)
    module, dm = cli.model, cli.datamodule
    module.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"])
    module.cuda().eval()
    dm.setup("fit")
    enc_out = []
    hook = module.model.encoder.register_forward_hook(lambda m, i, o: enc_out.append((o[0] if isinstance(o, tuple) else o).float().cpu()))
    losses = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec != "fp32"):
        for i, (inputs, targets) in enumerate(dm.val_dataloader()):
            if i == N:
                break
            outputs = module.model(to_cuda(inputs, prec == "fp32"))
            _, _, l = module.model.loss(outputs, to_cuda(targets, prec == "fp32"))
            losses.append(total(l))
    hook.remove()
    return enc_out, losses


if __name__ == "__main__":
    results = {cfg: run(cfg) for cfg in CONFIGS}
    ref = CONFIGS[0]
    for cfg in CONFIGS[1:]:
        for i, (a, b) in enumerate(zip(results[ref][0], results[cfg][0])):
            rel = ((a - b).norm() / a.norm()).item()
            print(f"event {i:2d}: encoder output rel. diff {ref} vs {cfg}: {rel:.2e}   loss {results[ref][1][i]:.4f} vs {results[cfg][1][i]:.4f}")
        import numpy as np
        prof = []
        for a, b in zip(results[ref][0], results[cfg][0]):
            a, b = a[0], b[0]                               # [hits, dim], in the encoder's (phi-sorted) order
            d = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)
            n = len(d); edges = np.linspace(0, n, 21).astype(int)
            prof.append([d[edges[k]:edges[k + 1]].mean().item() for k in range(20)])
            near = torch.cat([d[:1024], d[-1024:]]).mean().item(); mid = d[n // 2 - 1024 : n // 2 + 1024].mean().item()
        p = np.mean(prof, axis=0)
        print("rel. diff per hit along the phi-sorted sequence, 20 bins (seam at both ends):")
        print("  " + " ".join(f"{x:.3f}" for x in p))
        la, lb = results[ref][1], results[cfg][1]
        print(f"mean loss over {len(la)} events: {ref} {sum(la)/len(la):.4f}   {cfg} {sum(lb)/len(lb):.4f}")
