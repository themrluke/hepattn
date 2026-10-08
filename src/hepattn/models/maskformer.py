from typing import Any

import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.utils.model_utils import unmerge_inputs


class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        decoder: MaskFormerDecoder,
        tasks: nn.ModuleList,
        dim: int,
        target_object: str = "particle",
        pooling: nn.Module | None = None,
        matcher: nn.Module | None = None,
        input_sort_field: str | None = None,
        sorter: nn.Module | None = None,
        unified_decoding: bool = False,
        dynamic_query_source: str = "hit",
        encoder_tasks: nn.ModuleList | None = None,
        trim_target_costs: bool = False,
    ):
        """Initializes the MaskFormer model, which is a modular transformer-style architecture designed
        for multi-task object reconstruction with attention-based decoding and optional encoder blocks.

        Args:
            input_nets: A list of input modules, each responsible for embedding a specific constituent type.
            encoder: An optional encoder module that processes merged constituent embeddings with optional sorting.
            decoder: The decoder module that handles multi-layer decoding and task integration.
            tasks: A list of task modules, each responsible for producing and processing predictions from decoder outputs.
            dim: The dimensionality of the query and key embeddings.
            target_object: The target object name which is used to mark valid/invalid objects during matching.
            pooling: An optional pooling module used to aggregate features from the input constituents.
            matcher: A module used to match predictions to targets (e.g., using the Hungarian algorithm) for loss computation.
            input_sort_field: An optional key used to sort the input constituents (e.g., for windowed attention).
            sorter: An optional sorter module used to reorder input constituents before processing.
            unified_decoding: If True, inputs remain merged for task processing instead of being unmerged after encoding.
            dynamic_query_source: Name of the input type to use as the source for dynamic query initialization (default: "hit").
            encoder_tasks: Optional list of tasks to run after the encoder (before decoder). These tasks operate on post-encoder features.
            trim_target_costs: If True, the matching costs are computed only for the real target objects (which come first),
                not for the padding up to num_queries, which the matcher never reads. Losses still use every slot.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.tasks = tasks
        self.encoder_tasks = encoder_tasks or nn.ModuleList()
        self.decoder.encoder_tasks = self.encoder_tasks
        self.pooling = pooling
        self.tasks = tasks
        self.target_object = target_object
        self.matcher = matcher
        self.unified_decoding = unified_decoding
        self.decoder.unified_decoding = unified_decoding
        self.dynamic_query_source = dynamic_query_source
        self.decoder.dynamic_query_source = dynamic_query_source
        self.trim_target_costs = trim_target_costs

        assert not (input_sort_field and sorter), "Cannot specify both input_sort_field and sorter."
        # The per-head mask is defined over token index, so hits only have useful neighbours once
        # the sequence carries geometric locality. Unsorted, every mask comes out fully dense and
        # OR amplification is a no-op that still pays for replicating q/k/v across the hashes.
        or_sort_msg = "OR amplification needs a sorted sequence: set input_sort_field or a sorter, or the masks are dense and it does nothing."
        assert self.encoder.or_n_hashes is None or input_sort_field or sorter, or_sort_msg
        self.input_sort_field = input_sort_field
        self.sorter = sorter
        if self.sorter is not None:
            self.sorter.input_names = self.input_names

        # Layerwise matching state of the current forward pass, and the decoder hook that feeds it. The hook is
        # set once, so the compiled decoder sees the same function every step and doesn't recompile.
        self._layerwise = None
        if self.matches_during_forward:
            self.decoder.layer_outputs_hook = self._layer_outputs_ready

        assert "key" not in self.input_names, "'key' input name is reserved."
        assert "query" not in self.input_names, "'query' input name is reserved."
        assert not any("_" in name for name in self.input_names), "Input names cannot contain underscores."

    @property
    def input_names(self) -> list[str]:
        return [input_net.input_name for input_net in self.input_nets]

    @property
    def matches_during_forward(self) -> bool:
        """True if forward should be given the targets, so it can start matching each decoder layer early."""
        return self.matcher is not None and getattr(self.matcher, "overlap_layers", False) and self.sorter is None

    def _cost_targets(self, targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """The targets the matching costs are computed from: with trim_target_costs, the target objects cut to the
        real ones (they come first, as the matcher assumes), so the cost matrices are (batch, pred, real) instead of
        (batch, pred, num_targets). Reading the count waits for the targets to reach the device.
        """
        if not self.trim_target_costs:
            return targets
        valid = targets[f"{self.target_object}_valid"]
        num_target = valid.shape[1]
        num_real = max(int(valid.sum(dim=1).max()), 1)
        prefix = f"{self.target_object}_"
        return {
            k: v[:, :num_real] if k.startswith(prefix) and torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == num_target else v
            for k, v in targets.items()
        }

    @torch.compiler.disable
    def _layer_outputs_ready(self, layer_name: str, outputs: dict) -> None:
        """Decoder hook: once a layer's outputs exist, compute its costs and hand them to the matcher.

        Does nothing unless forward was given targets and started a layerwise matching. Never compiled, so
        the costs are computed exactly as in loss() (which runs outside the compiled encoder and decoder).
        """
        if self._layerwise is None:
            return
        matching, layer_names, targets = self._layerwise
        query_mask = outputs.get("encoder", {}).get("query_mask")
        # Same targets as _prepare_targets_and_outputs gives the costs in loss()
        if query_mask is not None and "query_mask" not in targets:
            targets = {**targets, "query_mask": query_mask}
        cost = self._layer_cost(outputs[layer_name], targets)
        if cost is None:
            return
        matching.submit(len(layer_names), cost, targets.get("query_mask"))
        layer_names.append(layer_name)

    def forward(self, inputs: dict[str, Tensor], targets: dict[str, Tensor] | None = None) -> dict[str, dict[str, dict[str, Tensor]]]:
        """Run the model. With targets and a matcher with overlap_layers, the matching of each decoder layer
        starts as soon as its outputs exist; loss() then collects it instead of matching from scratch.

        Raises:
            ValueError: If dynamic queries are used with a sorter or without their source input.
        """
        batch_size = inputs[self.input_names[0] + "_valid"].shape[0]
        x = {"inputs": inputs}

        # Track per-input slices into the merged key tensor.
        # This is only used to keep dynamic_queries compatible with unified_decoding.
        key_slices: dict[str, slice] = {}
        key_start = 0

        # Embed the input constituents
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            n_objects = x[input_name + "_embed"].shape[-2]
            key_slices[input_name] = slice(key_start, key_start + n_objects)
            key_start += n_objects

            # These slices can be used to pick out specific
            # objects after we have merged them all together
            # Only needed when not doing unified decoding
            if not self.unified_decoding:
                device = inputs[input_name + "_valid"].device
                mask = torch.cat([torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device) for i in self.input_names], dim=-1)
                x[f"key_is_{input_name}"] = mask.unsqueeze(0).expand(batch_size, -1)

        # Merge the input constituents and the padding mask into a single set
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in self.input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in self.input_names], dim=-1)
        # Preserve a non-None version for downstream logic that expects a tensor mask.
        x["key_valid_full"] = x["key_valid"]

        # If all key_valid are true, then we can just set it to None, however,
        # if we are using flash-varlen, we have to always provide a kv_mask argument
        if batch_size == 1 and x["key_valid"].all() and self.encoder.attn_type != "flash-varlen":
            x["key_valid"] = None

        # LEGACY. TODO: remove
        if self.input_sort_field and not self.sorter:
            x[f"key_{self.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.input_sort_field] for input_name in self.input_names], dim=-1
            )

        # OR amplification needs the hit coordinates, and they have to end up in the same order
        # as the tokens. Adding them here, before the sorter, covers both cases: a sorter will
        # permute them along with everything else, and without one they stay as they are and the
        # encoder permutes them itself using x_sort_value. Two 1-D fields rather than a stacked
        # (B, N, 2) tensor, because sort_inputs gathers with a (B, N) index that 3D would break.
        if self.encoder.or_n_hashes is not None:
            for field in ("eta", "phi"):
                x[f"key_{field}"] = torch.concatenate([inputs[f"{name}_{field}"] for name in self.input_names], dim=-1)

        # Dedicated sorting step before encoder
        if self.sorter is not None:
            x[f"key_{self.sorter.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.sorter.input_sort_field] for input_name in self.input_names], dim=-1
            )
            for input_name in self.input_names:
                field = f"{input_name}_{self.sorter.input_sort_field}"
                x[field] = inputs[field]
            x = self.sorter.sort_inputs(x)

        # Pass merged input constituents through the encoder
        x_sort_value = x.get(f"key_{self.input_sort_field}") if self.sorter is None else None
        # (B, N) + (B, N) -> (B, N, 2), eta first: E2LSHOrderingGrid reads coords[..., 0] as eta.
        x_coords = torch.stack([x["key_eta"], x["key_phi"]], dim=-1) if self.encoder.or_n_hashes is not None else None
        x["key_embed"] = self.encoder(x["key_embed"], x_sort_value=x_sort_value, kv_mask=x.get("key_valid"), x_coords=x_coords)

        # Keep dynamic query initialization compatible with unified decoding by ensuring
        # source_embed/source_valid refer to *post-encoder* features.
        if self.decoder.dynamic_queries and self.unified_decoding:
            if self.sorter is not None:
                raise ValueError("dynamic_queries with unified_decoding is not supported when sorter is enabled")
            if self.dynamic_query_source not in key_slices:
                raise ValueError(f"dynamic_queries=True requires an input named '{self.dynamic_query_source}'")
            source_slice = key_slices[self.dynamic_query_source]
            x[f"{self.dynamic_query_source}_embed"] = x["key_embed"][:, source_slice, :]
            x[f"{self.dynamic_query_source}_valid"] = x["key_valid_full"][:, source_slice]

        # Unmerge the updated features back into the separate input types only if not doing unified decoding
        if not self.unified_decoding:
            x = unmerge_inputs(x, self.input_names)

        # Run encoder tasks
        outputs = {"encoder": {}}
        for task in self.encoder_tasks:
            outputs["encoder"][task.name] = task(x, outputs=outputs["encoder"])

        # Start matching each decoder layer as soon as its outputs exist (overlap_layers)
        self._layerwise = None
        if targets is not None and self.matches_during_forward:
            cost_targets = self._cost_targets(targets)
            matching = self.matcher.start_layerwise(cost_targets[f"{self.target_object}_valid"], num_slots=len(self.decoder.decoder_layers) + 1)
            if matching is not None:
                self._layerwise = (matching, [], cost_targets)

        # Pass through decoder layers
        x, decoder_outputs = self.decoder(x, self.input_names)
        outputs["encoder"].update(decoder_outputs.pop("encoder", {}))
        outputs.update(decoder_outputs)

        # Do any pooling if desired
        if self.pooling is not None:
            x_pooled = self.pooling(x[f"{self.pooling.input_name}_embed"], x[f"{self.pooling.input_name}_valid"])
            x[f"{self.pooling.output_name}_embed"] = x_pooled

        # Get the final outputs
        outputs["final"] = {}
        for task in self.tasks:
            # Pass outputs dict so tasks can read from previously executed tasks
            outputs["final"][task.name] = task(x, outputs=outputs["final"])

        if self._layerwise is not None:
            matching, layer_names, _ = self._layerwise
            self._layer_outputs_ready("final", outputs)
            outputs["_layerwise_matching"] = (matching, layer_names)
            self._layerwise = None

        # store info about the input sort field for each input type
        if self.sorter is not None:
            sort = self.sorter.input_sort_field
            sort_dict = {f"{name}_{sort}": inputs[f"{name}_{sort}"] for name in self.input_names}
            outputs["final"][sort] = sort_dict

        return outputs

    def predict(self, outputs: dict) -> dict:
        """Takes the raw model outputs and produces a set of actual inferences / predictions.
        For example will take output probabilies and apply threshold cuts to prduce boolean predictions.

        Args:
            outputs: The outputs produced by the forward pass of the model.

        Returns:
            preds: A dictionary containing the predicted values for each task.
        """
        preds: dict[str, dict[str, Any]] = {}

        # Get query_mask from encoder outputs for masking padded queries in predictions
        query_mask = outputs.get("encoder", {}).get("query_mask")

        # Compute predictions for each task in each block
        for layer_name, layer_outputs in outputs.items():
            if layer_name.startswith("_"):
                continue

            preds[layer_name] = {}

            # Handle encoder tasks
            if layer_name == "encoder":
                for task in self.encoder_tasks:
                    if task.name not in layer_outputs:
                        continue
                    preds[layer_name][task.name] = task.predict(layer_outputs[task.name])

            # Handle decoder tasks
            else:
                for task in self.tasks:
                    if task.name not in layer_outputs:
                        continue
                    preds[layer_name][task.name] = task.predict(layer_outputs[task.name], query_mask=query_mask)

        return preds

    def _prepare_targets_and_outputs(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """Prepare targets and separate encoder/decoder outputs.

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets.

        Returns:
            Tuple of (targets, encoder_outputs, decoder_outputs) where:
            - targets: Targets dict with query_mask added if present
            - encoder_outputs: Separated encoder outputs
            - decoder_outputs: Separated decoder layer outputs
        """
        # Separate encoder and decoder outputs for cleaner logic (keys starting with "_" are internal)
        encoder_outputs = {"encoder": outputs["encoder"]} if "encoder" in outputs else {}
        decoder_outputs = {k: v for k, v in outputs.items() if k != "encoder" and not k.startswith("_")}

        # Include query_mask in targets if present (for masking padded query losses)
        if "encoder" in outputs and "query_mask" in outputs["encoder"] and "query_mask" not in targets:
            targets = targets.copy()
            targets["query_mask"] = outputs["encoder"]["query_mask"]

        # Sort targets if using a sorter
        if self.sorter is not None:
            targets = self.sorter.sort_targets(targets, decoder_outputs["final"][self.sorter.input_sort_field])

        return targets, encoder_outputs, decoder_outputs

    def _compute_encoder_losses(self, encoder_outputs: dict, targets: dict) -> dict[str, dict[str, Tensor]]:
        """Compute losses for encoder tasks (no matching required).

        Args:
            encoder_outputs: Dictionary of encoder layer outputs.
            targets: The data containing the targets.

        Returns:
            Dictionary of encoder losses keyed by layer name and task name.
        """
        losses: dict[str, dict[str, Tensor]] = {}
        for layer_name, layer_outputs in encoder_outputs.items():
            losses[layer_name] = {}
            for task in self.encoder_tasks:
                if task.name not in layer_outputs:
                    continue
                losses[layer_name][task.name] = task.loss(layer_outputs[task.name], targets, layer_outputs=layer_outputs)
        return losses

    def _compute_decoder_costs(self, decoder_outputs: dict, targets: dict) -> dict[str, Tensor]:
        """Compute costs for decoder layers by aggregating task costs.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs.
            targets: The data containing the targets.

        Returns:
            Dictionary of costs keyed by layer name. Cost axes are (batch, pred, true).
        """
        cost_targets = self._cost_targets(targets)
        return {layer_name: self._layer_cost(layer_outputs, cost_targets) for layer_name, layer_outputs in decoder_outputs.items()}

    def _layer_cost(self, layer_outputs: dict, targets: dict) -> Tensor | None:
        """Cost matrix (batch, pred, true) of one decoder layer, or None if no task contributes to it."""
        layer_costs = None

        # Get the cost contribution from each of the decoder tasks
        for task in self.tasks:
            # Skip tasks that do not contribute intermediate losses
            if task.name not in layer_outputs:
                continue

            # Compute costs
            task_costs = task.cost(layer_outputs[task.name], targets)

            # Add the cost on to our running cost total, otherwise initialise a running cost matrix
            for cost in task_costs.values():
                if layer_costs is None:
                    layer_costs = cost
                else:
                    layer_costs += cost

        # Added to allow completely turning off inter layer loss
        # Possibly redundant as completely switching them off performs worse
        if layer_costs is not None:
            layer_costs = layer_costs.detach()

        return layer_costs

    def _match_and_permute_outputs(self, decoder_outputs: dict, costs: dict[str, Tensor], targets: dict) -> None:
        """Perform optimal matching and permute decoder outputs accordingly.

        After permutation, outputs are aligned with target order, so the original
        particle_valid mask can be used directly by all tasks for loss computation.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs (will be modified in-place).
            costs: Dictionary of costs keyed by layer name.
            targets: The data containing the targets.
        """
        # Stack all layer costs into a single 4D tensor for parallel matching.
        # Some decoder layers may have no active loss tasks (cost=None); skip those.
        valid_layer_costs = [(name, layer_cost) for name, layer_cost in costs.items() if torch.is_tensor(layer_cost)]
        num_layers = len(valid_layer_costs)

        if num_layers > 0:
            layer_names = [name for name, _ in valid_layer_costs]
            # Stack costs: [num_layers, batch, num_pred, num_target]
            stacked_costs = torch.stack([layer_cost for _, layer_cost in valid_layer_costs], dim=0)
            batch_size = stacked_costs.shape[1]
            num_pred = stacked_costs.shape[2]
            num_target = stacked_costs.shape[3]

            # Reshape to [num_layers * batch, num_pred, num_target] to use layers as additional batch dim
            stacked_costs = stacked_costs.reshape(num_layers * batch_size, num_pred, num_target)

            # Expand validity mask to match stacked batch dimension: [num_layers * batch, num_target]
            # (num_target is the number of real targets with trim_target_costs; they come first)
            target_valid = targets[f"{self.target_object}_valid"][:, :num_target]
            stacked_target_valid = target_valid.unsqueeze(0).expand(num_layers, -1, -1).reshape(num_layers * batch_size, -1)

            # Get query_mask if present (for masking padded queries in matching)
            query_mask = targets.get("query_mask")
            stacked_query_valid = None
            if query_mask is not None:
                stacked_query_valid = query_mask.unsqueeze(0).expand(num_layers, -1, -1).reshape(num_layers * batch_size, -1)

            # Get the indices that can permute the predictions to yield their optimal matching
            # Output shape: [num_layers * batch, num_pred]
            stacked_pred_idxs = self.matcher(stacked_costs, stacked_target_valid, stacked_query_valid)

            # Reshape back to [num_layers, batch, num_pred]
            stacked_pred_idxs = stacked_pred_idxs.view(num_layers, batch_size, num_pred)
            self._permute_outputs(decoder_outputs, layer_names, list(stacked_pred_idxs))

    def _permute_outputs(self, decoder_outputs: dict, layer_names: list[str], layer_pred_idxs: list[Tensor]) -> None:
        """Permute each layer's outputs by its matching, shape [batch, num_pred], so output[i] belongs to target[i]."""
        for layer_name, pred_idxs in zip(layer_names, layer_pred_idxs, strict=True):
            # Create batch indices for indexing
            batch_idxs_expanded = torch.arange(pred_idxs.shape[0], device=pred_idxs.device).unsqueeze(1)

            for task in self.tasks:
                if not task.should_permute_outputs(layer_name, decoder_outputs[layer_name]):
                    continue

                for output_name in task.outputs:
                    output_tensor = decoder_outputs[layer_name][task.name][output_name]
                    decoder_outputs[layer_name][task.name][output_name] = output_tensor[batch_idxs_expanded, pred_idxs]

    def _compute_decoder_losses(self, decoder_outputs: dict, targets: dict) -> dict[str, dict[str, Tensor]]:
        """Compute final losses for decoder tasks using permuted outputs.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs (already permuted).
            targets: The targets dict to use for loss computation.

        Returns:
            Dictionary of decoder losses keyed by layer name and task name.
        """
        losses: dict[str, dict[str, Tensor]] = {}

        for layer_name, layer_outputs in decoder_outputs.items():
            losses[layer_name] = {}

            for task in self.tasks:
                if task.name not in layer_outputs:
                    continue

                task_losses = task.loss(layer_outputs[task.name], targets, layer_outputs=layer_outputs)
                losses[layer_name][task.name] = task_losses

        return losses

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """Computes the loss between the forward pass of the model and the data / targets.

        This method performs Hungarian matching to align predictions with targets before computing
        losses. The matching works as follows:

        1. **Cost Matrix**: A cost matrix of shape [batch, num_pred, num_target] is computed by summing
           task-specific costs (e.g., BCE for classification, L1 for regression).

        2. **Hungarian Matching**: The matcher solves the linear assignment problem on the transposed
           cost matrix [batch, num_target, num_pred] and returns `pred_idxs` of shape [batch, num_pred]
           where `pred_idxs[i]` = which prediction slot should be placed at target position `i`.

        3. **Output Permutation**: Outputs are gathered using `output[batch_idxs, pred_idxs]`, which
           reorders predictions so that `output[i]` corresponds to `target[i]`. After this permutation,
           the original `particle_valid` mask can be used directly to filter matched pairs.

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets.

        Returns:
            Tuple of (outputs, targets, losses) where:
            - outputs: The outputs dict.
            - targets: The targets dict.
            - losses: A dictionary containing the computed losses for each task.
        """
        # Prepare targets and separate encoder/decoder outputs
        targets, encoder_outputs, decoder_outputs = self._prepare_targets_and_outputs(outputs, targets)

        # Compute encoder losses (no matching required)
        losses = self._compute_encoder_losses(encoder_outputs, targets)

        layerwise = outputs.pop("_layerwise_matching", None)
        if layerwise is not None:
            # The forward pass already started matching each layer (overlap_layers): collect it
            matching, layer_names = layerwise
            self._permute_outputs(decoder_outputs, layer_names, matching.results())
        else:
            # Compute costs for decoder layers
            costs = self._compute_decoder_costs(decoder_outputs, targets)

            # Perform matching and permute decoder outputs to align with target order
            # After this, output[i] corresponds to target[i] for all i
            self._match_and_permute_outputs(decoder_outputs, costs, targets)

        # Compute final decoder losses using permuted outputs
        decoder_losses = self._compute_decoder_losses(decoder_outputs, targets)
        losses.update(decoder_losses)

        return outputs, targets, losses
