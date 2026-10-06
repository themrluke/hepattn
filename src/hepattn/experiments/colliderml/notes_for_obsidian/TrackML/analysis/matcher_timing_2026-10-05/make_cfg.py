# Make a throwaway training config: no Comet, no checkpoints, logs in scratch, N steps.
import sys, yaml
src, out, name, steps, root = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
matcher_args = dict(a.split("=") for a in sys.argv[6:])
cfg = yaml.safe_load(open(src))
cfg["name"] = name
t = cfg["trainer"]
t.update(logger=False, enable_checkpointing=False, default_root_dir=root, max_steps=steps, limit_val_batches=0, num_sanity_val_steps=0)
t["callbacks"] = [c for c in t["callbacks"] if c["class_path"].endswith(("Compile", "TQDMProgressBar"))]
for c in t["callbacks"]:
    if c["class_path"].endswith("TQDMProgressBar"):
        c.setdefault("init_args", {})["refresh_rate"] = 50
m = cfg["model"]["model"]["init_args"]["matcher"]["init_args"]
for k, v in matcher_args.items():
    m[k] = {"true": True, "false": False}.get(v, v)
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(out, m)
