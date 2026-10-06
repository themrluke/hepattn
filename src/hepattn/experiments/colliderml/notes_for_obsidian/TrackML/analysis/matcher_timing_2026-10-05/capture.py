# Load a checkpoint's weights, run a few training steps, and save every matcher input.
import os, sys, torch
from hepattn.models import matcher as m

OUT, N = os.environ["CAPTURE_DIR"], int(os.environ.get("CAPTURE_N", "8"))
orig, calls = m.Matcher.forward, [0]


def forward(self, costs, object_valid_mask=None, query_valid_mask=None):
    if calls[0] < N:
        torch.save({"costs": costs.detach().cpu(), "object_valid": object_valid_mask.detach().cpu(),
                    "query_valid": None if query_valid_mask is None else query_valid_mask.detach().cpu()},
                   f"{OUT}/costs_{calls[0]:03d}.pt")
        print(f"saved {calls[0]} {tuple(costs.shape)} particles={int(object_valid_mask[0].sum())} "
              f"valid_queries={int(query_valid_mask[0].sum())}", flush=True)
    calls[0] += 1
    return orig(self, costs, object_valid_mask, query_valid_mask)


if __name__ == "__main__":
    m.Matcher.forward = forward
    sys.path.insert(0, "/shared/projects/hepattn/src/hepattn/experiments/trackml")
    import run_tracking

    def on_fit_start(self):  # weights only: no optimizer/scheduler state, so any checkpoint works
        self.load_state_dict(torch.load(os.environ["INIT_WEIGHTS"], map_location="cpu", weights_only=False)["state_dict"])
        print(f"loaded weights from {os.environ['INIT_WEIGHTS']}", flush=True)

    run_tracking.TrackMLTracker.on_fit_start = on_fit_start
    run_tracking.main()
