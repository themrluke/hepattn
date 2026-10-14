"""Compare identity.py outputs: matchings, loss terms and gradient hashes, event by event."""
import sys

import torch

a, b = torch.load(sys.argv[1], weights_only=False), torch.load(sys.argv[2], weights_only=False)
assert len(a) == len(b)
n_match = n_loss = n_grad = 0
for i, (x, y) in enumerate(zip(a, b, strict=True)):
    same_match = x["matching"].keys() == y["matching"].keys() and all(torch.equal(x["matching"][k], y["matching"][k]) for k in x["matching"])
    same_loss = x["losses"] == y["losses"]
    diff_grads = [k for k in x["grads"] if x["grads"][k] != y["grads"].get(k)]
    n_match += same_match
    n_loss += same_loss
    n_grad += not diff_grads
    rel = None
    if "grad_values" in x and "grad_values" in y:
        rel = ((x["grad_values"] - y["grad_values"]).norm() / x["grad_values"].norm()).item()
    worst = max(abs(x["losses"][k] - y["losses"][k]) for k in x["losses"])
    print(f"event {i}: matching {'same' if same_match else 'DIFFERENT'}, losses {'same' if same_loss else f'differ (max {worst:.3g})'}, "
          f"grads {'same' if not diff_grads else f'{len(diff_grads)}/{len(x["grads"])} params differ'}"
          + (f", |Δgrad|/|grad| = {rel:.2e}" if rel is not None else ""))
print(f"{len(a)} events: identical matching {n_match}, identical losses {n_loss}, identical gradients {n_grad}")
