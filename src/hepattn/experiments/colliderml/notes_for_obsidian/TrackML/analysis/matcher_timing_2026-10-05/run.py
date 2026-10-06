# One timing run: worktree code (via the sitecustomize shim), weights from INIT_WEIGHTS, per-step and per-matcher-call times to TIMING_OUT.
import json, os, sys, time
import torch

if __name__ == "__main__":
    sys.path.insert(0, "/shared/projects/hepattn-matcher/src/hepattn/experiments/trackml")
    import run_tracking
    from hepattn.models import matcher as m
    assert "hepattn-matcher" in m.__file__, m.__file__

    step_ends, match_times = [], []
    orig_forward = m.Matcher.forward

    def timed_forward(self, *a, **k):
        t = time.perf_counter()
        out = orig_forward(self, *a, **k)
        match_times.append(time.perf_counter() - t)
        return out

    m.Matcher.forward = timed_forward
    cls = run_tracking.TrackMLTracker

    def on_fit_start(self):
        state = torch.load(os.environ["INIT_WEIGHTS"], map_location="cpu", weights_only=False)["state_dict"]
        self.load_state_dict(state)
        print(f"loaded weights from {os.environ['INIT_WEIGHTS']}", flush=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step_ends.append(time.perf_counter())

    def on_train_end(self):
        json.dump({"step_ends": step_ends, "match_times": match_times}, open(os.environ["TIMING_OUT"], "w"))

    cls.on_fit_start, cls.on_train_batch_end, cls.on_train_end = on_fit_start, on_train_batch_end, on_train_end
    run_tracking.main()
