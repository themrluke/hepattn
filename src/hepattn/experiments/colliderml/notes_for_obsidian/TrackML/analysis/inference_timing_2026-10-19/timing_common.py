"""Shared pieces of the inference timing study: load a trained TrackML model, capture real encoder inputs, build encoder
variants (flex sliding window or OR, any window / depth) and set the flex BlockMask block size."""
import functools
import sys
import tempfile
from pathlib import Path

import torch
import yaml

TRACKML = Path("/shared/projects/hepattn/src/hepattn/experiments/trackml")
CONFIGS = TRACKML / "configs"
sys.path.insert(0, str(TRACKML))

import hepattn.models.attention as attention_module  # noqa: E402
import hepattn.models.encoder as encoder_module  # noqa: E402
from hepattn.models import Encoder  # noqa: E402
from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask  # noqa: E402
from torch.nn.attention.flex_attention import flex_attention as _flex_attention  # noqa: E402

_BLOCK_MASK_FNS = {}


def set_block_size(block_size: int) -> None:
    """Make the encoder build its flex BlockMask with this block size (the default is 128). Only the sparsity
    granularity changes: which (query, key) pairs attend is set by the mask_mod, so the result is the same."""
    if block_size not in _BLOCK_MASK_FNS:
        _BLOCK_MASK_FNS[block_size] = torch.compile(functools.partial(_create_block_mask, BLOCK_SIZE=block_size), dynamic=True)
    encoder_module.create_block_mask = _BLOCK_MASK_FNS[block_size]
    # Mask blocks smaller than the kernel's tiles are rejected ("Q and KV block size must be divisible by BLOCK_M and
    # BLOCK_N"), so below 128 the kernel tiles are set to the block size too. Takes effect for encoders built afterwards.
    attention_module.ATTN_TYPES["flex"] = (
        _flex_attention if block_size >= 128 else functools.partial(_flex_attention, kernel_options={"BLOCK_M": block_size, "BLOCK_N": block_size})
    )


def load_module(config: str, ckpt: str | None, overrides: dict | None = None):
    """A TrackMLTracker from a config (and checkpoint), on the GPU in eval mode, plus its datamodule."""
    from lightning.pytorch.cli import LightningCLI
    from run_tracking import TrackMLTracker

    from hepattn.experiments.trackml.data import TrackMLDataModule

    cfg = yaml.safe_load(open(config))
    for key in ("ckpt_path", "name", "matmul_precision"):
        cfg.pop(key, None)
    cfg["model"]["name"] = "inference-timing"
    cfg["trainer"].update(logger=False, callbacks=[])
    for path, value in (overrides or {}).items():
        node = cfg
        *keys, last = path.split(".")
        for k in keys:
            node = node[k]
        node[last] = value
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, f)
    f.close()
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(model_class=TrackMLTracker, datamodule_class=TrackMLDataModule, args=["--config", f.name], run=False)
    module, dm = cli.model, cli.datamodule
    if ckpt:
        module.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    return module.cuda().eval(), dm


def test_events(dm, n: int):
    """The first n test-set events, moved to the GPU."""
    dm.setup("test")
    out = []
    for i, (inputs, targets) in enumerate(dm.test_dataloader()):
        if i == n:
            break
        out.append(({k: v.cuda() for k, v in inputs.items()}, {k: v.cuda() for k, v in targets.items()}))
    return out


def capture_encoder_inputs(module, events):
    """Run the model on each event and keep what the encoder receives (embedded hits, sort value, mask, coords)."""
    captured = []

    def hook(_mod, args, kwargs):
        captured.append({"x": args[0].detach().clone(), **{k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}})

    h = module.model.encoder.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for inputs, _ in events:
            module.model(inputs)
    h.remove()
    return captured


def encoder_args(kind: str, window: int, num_layers: int = 8) -> dict:
    """Encoder init args of the Pix0.6 configs: 'flex' (sliding window with phi wrap) or 'or' (3 hashes x 100 regions)."""
    cfg = yaml.safe_load(open(CONFIGS / "tracking-eta4-pt600-or-w64.yaml"))
    args = dict(cfg["model"]["model"]["init_args"]["encoder"]["init_args"])
    args.update(window_size=window, num_layers=num_layers)
    if kind == "flex":
        for k in ("or_n_hashes", "or_num_regions", "or_base_seed", "or_impl"):
            args.pop(k, None)
        args["window_wrap"] = True
    return args


def build_encoder(kind: str, window: int, num_layers: int = 8, seed: int = 0) -> Encoder:
    torch.manual_seed(seed)  # same weights for the same (kind, window, depth), whatever the block size
    return Encoder(**encoder_args(kind, window, num_layers)).cuda().eval()


def run_encoder(enc, inp):
    return enc(inp["x"], x_sort_value=inp.get("x_sort_value"), kv_mask=inp.get("kv_mask"), x_coords=inp.get("x_coords"))
