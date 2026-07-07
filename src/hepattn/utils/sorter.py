import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense


class Sorter(nn.Module):
    def __init__(self, input_sort_field: str) -> None:
        super().__init__()
        self.input_sort_field = input_sort_field
        self.input_names = None  # set by MaskFormer

    def _sort_key(self, inputs: dict[str, Tensor], input_name: str) -> Tensor:
        """Return the per-constituent value the sequence is ordered by (ascending).

        The base sorter orders by the raw value of ``{input_name}_{input_sort_field}``.
        Subclasses (e.g. :class:`LearnedSorter`) override this to order by a learned score.
        """
        return inputs[f"{input_name}_{self.input_sort_field}"]

    def sort_inputs(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        input_names = [*self.input_names, "key"]
        sort_idxs = {}

        for input_name in input_names:
            sort_idx = torch.argsort(self._sort_key(inputs, input_name), dim=-1)
            sort_idxs[input_name] = sort_idx

            for key, x in inputs.items():
                if x is None or input_name not in key:
                    continue

                # embeddings
                if key == f"{input_name}_embed":
                    sort_dim = 1
                    this_sort_idx = sort_idx.unsqueeze(-1).expand_as(x)

                # input type masks
                elif "key_is_" in key:
                    if input_name != "key":
                        continue
                    sort_dim = 1
                    this_sort_idx = sort_idx

                # normal inputs
                elif key.startswith(input_name):
                    sort_dim = 1
                    this_sort_idx = sort_idx

                else:
                    raise ValueError(f"Unexpected key {key} for input type {input_name}")

                shape_before = x.shape
                inputs[key] = torch.gather(x, sort_dim, this_sort_idx)
                assert inputs[key].shape == shape_before, f"Shape mismatch after sorting: {inputs[key].shape} != {shape_before} for key {key}"

        return inputs

    def sort_targets(self, targets: dict, sort_fields: dict[str, Tensor]) -> dict:
        for input_name in self.input_names:
            sort_idx = torch.argsort(sort_fields[f"{input_name}_{self.input_sort_field}"], dim=-1)

            for key, x in targets.items():
                if x is None or input_name not in key:
                    continue

                # sort target mask
                if x.ndim == 3:
                    sort_dim = 2
                    this_sort_idx = sort_idx
                    this_sort_idx = sort_idx.unsqueeze(1).expand_as(x)

                # sort target for input constituent
                elif x.ndim == 2:
                    sort_dim = 1
                    this_sort_idx = sort_idx
                else:
                    raise ValueError(f"Unexpected key {key} for input hit {input_name}")

                shape_before = x.shape
                targets[key] = torch.gather(x, sort_dim, this_sort_idx)
                assert targets[key].shape == shape_before, f"Shape mismatch after sorting: {targets[key].shape} != {shape_before} for key {key}"

        return targets


class LearnedSorter(Sorter):
    """Learnable serialization of constituents (OPTNet, arXiv:2605.17197, Eq. 1).

    A lightweight MLP predicts a scalar score ``s_i = sigmoid(MLP(fields_i))`` per
    constituent from its *raw* per-hit fields, and the sequence is ordered by
    ``argsort(s)``. The score head is trained by an auxiliary ordering loss
    (:class:`~hepattn.models.task.SupervisedOrderingTask`); because ``argsort`` is
    non-differentiable, no gradient flows from the downstream task into the score
    head, so the two objectives stay decoupled under one optimizer.

    Two-phase schedule (paper §3.4): for the first ``warmup_steps`` training steps the
    sequence is ordered by ``input_sort_field`` (the analogue of the paper's static
    space-filling curve -- use the tuned Hilbert / phi here) while the score head
    learns; afterwards the learned score drives the ordering. Scores are computed
    every step regardless, so the ordering loss can train the head throughout.

    The score is stashed in original (pre-sort) order under ``{name}_{score_field}``
    (and, for the merged set, ``ordering_score``) so a label-based ordering loss can
    align it with the original-order targets.

    Args:
        input_sort_field: Field used for ordering during warmup and for
            :meth:`sort_targets` (e.g. ``"phi"``, or a precomputed Hilbert index).
        score_fields: Raw per-hit fields fed to the score MLP. Defaults to Cartesian +
            ``(r, eta)`` -- continuous coordinates with no phi seam, so the head can
            learn a wrap-safe, region-adaptive order.
        hidden_dims: Hidden layer sizes of the score MLP.
        warmup_steps: Training steps ordered by ``input_sort_field`` before switching
            to the learned score (irrelevant when there is no encoder to warm up).
        score_field: Suffix under which per-input scores are stashed.
    """

    def __init__(
        self,
        input_sort_field: str = "phi",
        score_fields: tuple[str, ...] = ("x", "y", "z", "r", "eta"),
        hidden_dims: tuple[int, ...] = (64, 64),
        warmup_steps: int = 0,
        score_field: str = "sort_score",
    ) -> None:
        super().__init__(input_sort_field)
        self.score_fields = list(score_fields)
        self.warmup_steps = warmup_steps
        self.score_field = score_field
        self.score_net = Dense(len(self.score_fields), 1, hidden_layers=list(hidden_dims))
        # Checkpointed, device-correct training-step counter gating the warmup schedule.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def compute_scores(self, inputs: dict[str, Tensor], input_name: str) -> Tensor:
        """Predict and stash per-constituent scores in ``[0, 1]`` for ``input_name``.

        Called with the raw fields still in their original order, so the stashed score
        aligns element-wise with the original-order targets.
        """
        feats = torch.stack([inputs[f"{input_name}_{f}"] for f in self.score_fields], dim=-1)
        score = torch.sigmoid(self.score_net(feats.float()).squeeze(-1))
        inputs[f"{input_name}_{self.score_field}"] = score
        if input_name == "key":
            inputs["ordering_score"] = score
        return score

    def _sort_key(self, inputs: dict[str, Tensor], input_name: str) -> Tensor:
        """Order by the learned score, falling back to ``input_sort_field`` during warmup."""
        score = self.compute_scores(inputs, input_name)
        if self._step < self.warmup_steps:
            return inputs[f"{input_name}_{self.input_sort_field}"]
        return score

    def sort_inputs(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Sort as in the base class, then advance the warmup counter (training only)."""
        out = super().sort_inputs(inputs)
        if self.training:
            self._step += 1
        return out
