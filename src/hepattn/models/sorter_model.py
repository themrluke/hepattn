from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from hepattn.utils.sorter import LearnedSorter


class SorterModel(nn.Module):
    """Encoder-free model for training a :class:`~hepattn.utils.sorter.LearnedSorter`.

    Computes a per-hit sort score from the raw input fields and trains it with an
    auxiliary ordering task (e.g. :class:`~hepattn.models.task.SupervisedOrderingTask`)
    -- no encoder, decoder, matcher or windowed attention. Because the score head is
    trained purely by the ordering loss (the downstream task cannot backprop through
    ``argsort``), the sorter can be trained standalone here and the resulting order
    then frozen and fed to a hit-filter / tracking model as its ``input_sort_field``.

    Mirrors the model contract the :class:`~hepattn.models.wrapper.ModelWrapper`
    expects: ``forward`` -> outputs, ``predict``, and ``loss`` returning
    ``(outputs, targets, losses)`` keyed by layer ("final") then task then loss name.

    Args:
        sorter: The learned sorter holding the score MLP.
        tasks: Ordering tasks scored on the per-hit sort score.
        input_name: Constituent set to score (single hit type, e.g. ``"sihit"``).
        passthrough_fields: Raw per-hit fields (suffixes) the tasks read as labels
            (e.g. ``("particle_id",)`` -> ``sihit_particle_id``), copied into ``x``.
    """

    def __init__(
        self,
        sorter: LearnedSorter,
        tasks: nn.ModuleList,
        input_name: str = "sihit",
        passthrough_fields: tuple[str, ...] = ("particle_id",),
    ):
        super().__init__()
        self.sorter = sorter
        self.tasks = tasks
        self.input_name = input_name
        self.passthrough_fields = list(passthrough_fields)
        # No encoder-stage tasks; kept so ModelWrapper metric logging can introspect it.
        self.encoder_tasks = nn.ModuleList([])

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        name = self.input_name
        x: dict[str, Tensor] = {}
        # Per-hit score in original (unsorted) order, aligned with the original-order targets.
        x[f"{name}_{self.sorter.score_field}"] = self.sorter.compute_scores(inputs, name)
        x[f"{name}_valid"] = inputs[f"{name}_valid"]
        for field in self.passthrough_fields:
            x[f"{name}_{field}"] = inputs[f"{name}_{field}"]

        outputs = {"final": {}}
        for task in self.tasks:
            outputs["final"][task.name] = task(x)
        return outputs

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"final": {task.name: task.predict(outputs["final"][task.name]) for task in self.tasks}}

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        losses = {"final": {task.name: task.loss(outputs["final"][task.name], targets) for task in self.tasks}}
        return outputs, targets, losses
