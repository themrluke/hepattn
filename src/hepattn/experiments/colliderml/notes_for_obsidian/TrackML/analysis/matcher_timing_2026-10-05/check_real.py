# On captured real events: original matcher (from git) vs prepare_on_device vs trim.
import glob, importlib.util, subprocess, sys, tempfile
from pathlib import Path
import torch
from hepattn.models.matcher import Matcher


def load_original():
    src = subprocess.run(["git", "show", "ef88a68:src/hepattn/models/matcher.py"], cwd="/shared/projects/hepattn-matcher",
                         capture_output=True, text=True, check=True).stdout
    d = Path(tempfile.mkdtemp()); (d / "original_matcher.py").write_text(src); sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("original_matcher", d / "original_matcher.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["original_matcher"] = mod; spec.loader.exec_module(mod)
    return mod.Matcher


if __name__ == "__main__":
    kw = dict(default_solver="lap1015_late", adaptive_solver=False, parallel_solver=True, parallel_backend="process", n_jobs=4)
    original, prep = load_original()(**kw), Matcher(**kw, prepare_on_device=True)
    trim = Matcher(**kw, prepare_on_device=True, trim_padded_queries=True)
    for stage in sys.argv[1:]:
        files = sorted(glob.glob(f"{stage}/costs_*.pt"))
        n_prep = n_match = moved = total_unmatched = 0
        for f in files:
            d = torch.load(f)
            c, ov, qv = d["costs"].cuda(), d["object_valid"].cuda(), d["query_valid"].cuda()
            a, b, t = original(c, ov, qv), prep(c, ov, qv), trim(c, ov, qv)
            n_prep += torch.equal(a, b)
            for k in range(len(a)):  # the 4 layers
                L, Q = int(ov[k].sum()), int(qv[k].sum())
                n_match += torch.equal(a[k, :L], t[k, :L])
                # original: valid unmatched queries that land in masked slots (>= Q), i.e. left out of the loss
                moved += int((a[k, Q:] < Q).sum())
                total_unmatched += Q - L
        print(f"{stage.split('/')[-1]}: {len(files)} events | prepare_on_device full permutation identical: {n_prep}/{len(files)} | "
              f"trim same query for every particle: {n_match}/{4 * len(files)} layer-matrices | original code left "
              f"{moved} of {total_unmatched} unmatched valid queries ({100 * moved / max(total_unmatched, 1):.2f}%) in masked slots")
