"""One timing run of the overlap branch (via the import shim): weights from INIT_WEIGHTS, per-step end times and the time
the loss spends waiting for the matching, to TIMING_OUT. With PROFILE_OUT set, also a torch.profiler trace of steps
PROFILE_FROM .. PROFILE_FROM+19 (summary tables + Chrome trace).
"""
import json
import os
import sys
import time

import torch

if __name__ == "__main__":
    sys.path.insert(0, "/shared/projects/hepattn-overlap/src/hepattn/experiments/trackml")
    import run_tracking

    from hepattn.models import matcher as m

    assert "hepattn-overlap" in m.__file__, m.__file__

    step_ends, match_times = [], []

    def timed(fn):
        def wrapper(*a, **k):
            t = time.perf_counter()
            out = fn(*a, **k)
            match_times.append(time.perf_counter() - t)
            return out

        return wrapper

    # Time spent in the step waiting for the matching: the whole call without overlap, collecting the results with it
    m.Matcher.forward = timed(m.Matcher.forward)
    m.LayerwiseMatching.results = timed(m.LayerwiseMatching.results)

    profile_out = os.environ.get("PROFILE_OUT")
    profile_from = int(os.environ.get("PROFILE_FROM", "300"))
    prof = None
    if profile_out:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=profile_from - 5, warmup=5, active=20, repeat=1),
            record_shapes=False,
            with_stack=False,
        )

    cls = run_tracking.TrackMLTracker

    def on_fit_start(self):
        state = torch.load(os.environ["INIT_WEIGHTS"], map_location="cpu", weights_only=False)["state_dict"]
        self.load_state_dict(state)
        print(f"loaded weights from {os.environ['INIT_WEIGHTS']}", flush=True)
        if prof is not None:
            prof.start()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step_ends.append(time.perf_counter())
        if prof is not None:
            prof.step()

    def on_train_end(self):
        json.dump({"step_ends": step_ends, "match_times": match_times}, open(os.environ["TIMING_OUT"], "w"))
        if prof is not None:
            prof.stop()
            prof.export_chrome_trace(profile_out + ".trace.json")
            with open(profile_out + ".txt", "w") as f:
                for key in ("self_cuda_time_total", "self_cpu_time_total"):
                    f.write(prof.key_averages().table(sort_by=key, row_limit=40) + "\n\n")

    cls.on_fit_start, cls.on_train_batch_end, cls.on_train_end = on_fit_start, on_train_batch_end, on_train_end
    run_tracking.main()
