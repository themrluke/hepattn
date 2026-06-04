from pathlib import Path

import h5py
import numpy as np
from lightning import Callback, LightningModule, Trainer
from torch import Tensor

from hepattn.utils.tensor_utils import tensor_to_numpy


def _numpy_sample(value, idx):
    return tensor_to_numpy(value[idx])


def _lookup_nested(mapping, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _paper_track_iou_values(final_preds, idx):
    track_iou = _lookup_nested(final_preds, "track_iou", "query_iou")
    if track_iou is None:
        return None
    return _numpy_sample(track_iou, idx).astype(np.float32)


def _paper_track_valid_prob_values(final_preds, idx):
    valid_prob = _lookup_nested(final_preds, "track_valid", "track_valid_prob")
    if valid_prob is None:
        valid_mask = _lookup_nested(final_preds, "track_valid", "track_valid")
        if valid_mask is None:
            return None
        return _numpy_sample(valid_mask, idx).astype(np.float32)
    return _numpy_sample(valid_prob, idx).astype(np.float32)


def _paper_class_scores(final_preds, idx):
    valid_prob_np = _paper_track_valid_prob_values(final_preds, idx)
    if valid_prob_np is None:
        return None
    return np.stack([valid_prob_np, 1.0 - valid_prob_np], axis=-1)


def _paper_mask_values(final_outputs, final_preds, idx):
    mask_logits = _lookup_nested(final_outputs, "track_hit_valid", "track_hit_logit")
    if mask_logits is not None:
        return _numpy_sample(mask_logits, idx).astype(np.float32)

    mask_values = _lookup_nested(final_preds, "track_hit_valid", "track_hit_valid")
    if mask_values is None:
        return None
    return _numpy_sample(mask_values, idx).astype(np.float32)


def _paper_regression_values(final_preds, idx, num_queries):
    regression = {}
    track_regr = _lookup_nested(final_preds, "track_regr")
    if isinstance(track_regr, dict):
        for key, value in track_regr.items():
            old_key = key.removeprefix("track_")
            regression[old_key] = _numpy_sample(value, idx).astype(np.float32)

    has_cartesian = all(key in regression for key in ("px", "py", "pz"))
    if not has_cartesian and all(key in regression for key in ("pt", "eta", "phi")):
        pt = regression["pt"]
        eta = regression["eta"]
        phi = regression["phi"]
        regression["px"] = (pt * np.cos(phi)).astype(np.float32)
        regression["py"] = (pt * np.sin(phi)).astype(np.float32)
        regression["pz"] = (pt * np.sinh(eta)).astype(np.float32)
        has_cartesian = True

    if not has_cartesian:
        missing = np.full((num_queries,), np.nan, dtype=np.float32)
        regression["px"] = missing
        regression["py"] = missing.copy()
        regression["pz"] = missing.copy()
        return regression, True

    return regression, False


class PredictionWriter(Callback):
    def __init__(
        self,
        write_inputs: bool,
        write_outputs: bool,
        write_preds: bool,
        write_targets: bool,
        write_losses: bool,
        write_layers: list[str] | None = None,
        write_paper_compatible_test: bool = False,
        paper_track_valid_threshold: float = 0.5,
        paper_iou_threshold: float = 0.0,
    ):
        super().__init__()
        _ = (
            write_inputs,
            write_outputs,
            write_preds,
            write_targets,
            write_losses,
            write_layers,
            write_paper_compatible_test,
            paper_track_valid_threshold,
            paper_iou_threshold,
        )
        self.file = None
        self.event_index = 0
        self.missing_regression_warned = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
        self.trainer = trainer
        self.file = h5py.File(self.output_path, "w")

    @property
    def output_path(self) -> Path:
        return Path(self.trainer.ckpt_dir / f"{self.trainer.ckpt_name}__test.h5")

    def on_test_batch_end(self, trainer, pl_module, test_step_outputs, batch, batch_idx):
        _inputs, targets = batch
        sorted_targets = None

        if len(test_step_outputs) == 4:
            outputs, preds, _losses, sorted_targets = test_step_outputs
        else:
            outputs, preds, _losses = test_step_outputs

        targets_to_write = sorted_targets if sorted_targets is not None else targets

        if "sample_id" in targets_to_write:
            sample_ids = targets_to_write["sample_id"]
            for idx, sample_id in enumerate(sample_ids):
                self.write_sample(sample_id, targets_to_write, outputs, preds, idx)
        else:
            self.write_sample(batch_idx, targets_to_write, outputs, preds, 0)

    def write_sample(self, sample_id, targets, outputs, preds, idx):
        if isinstance(sample_id, Tensor):
            sample_id = sample_id.item()

        sample_group = self.file.create_group(f"event_{self.event_index}")
        sample_group.attrs["sample_id"] = sample_id
        self.event_index += 1

        required_target_keys = (
            "paper_truth_particle_id",
            "paper_truth_hit_id",
            "paper_truth_weight",
            "paper_hit_particle_id",
            "paper_hit_id",
            "paper_hit_weight",
            "paper_all_particle_id",
            "paper_all_particle_pt",
            "paper_all_particle_eta",
            "paper_all_particle_phi",
            "paper_all_particle_vz",
            "paper_all_particle_n_hits",
        )
        missing = [key for key in required_target_keys if key not in targets]
        if missing:
            msg = f"Paper-format TrackML output requires test targets {missing}."
            raise ValueError(msg)

        final_preds = preds.get("final", {})
        final_outputs = outputs.get("final", {})

        class_scores = _paper_class_scores(final_preds, idx)
        if class_scores is None:
            raise ValueError("Paper-compatible output requires `preds/final/track_valid` or `track_valid_prob`.")
        valid_prob = _paper_track_valid_prob_values(final_preds, idx)
        pred_iou = _paper_track_iou_values(final_preds, idx)

        masks = _paper_mask_values(final_outputs, final_preds, idx)
        if masks is None:
            raise ValueError("Paper-compatible output requires `track_hit_valid` predictions.")

        regression, used_placeholder_regression = _paper_regression_values(final_preds, idx, num_queries=masks.shape[0])
        if used_placeholder_regression and not self.missing_regression_warned:
            print(
                "PredictionWriter: no track regression outputs were found; paper-compatible "
                "file is writing NaN `px/py/pz` placeholders. Old eval fake-rate "
                "or efficiency numbers that depend on predicted track kinematics will not "
                "be directly comparable for this checkpoint."
            )
            self.missing_regression_warned = True

        truth_group = sample_group.create_group("truth")
        hits_group = sample_group.create_group("hits")
        parts_group = sample_group.create_group("parts")
        preds_group = sample_group.create_group("preds")
        regression_group = preds_group.create_group("regression")

        self.create_dataset(truth_group, "particle_id", _numpy_sample(targets["paper_truth_particle_id"], idx))
        self.create_dataset(truth_group, "hit_id", _numpy_sample(targets["paper_truth_hit_id"], idx))
        self.create_dataset(truth_group, "weight", _numpy_sample(targets["paper_truth_weight"], idx))

        self.create_dataset(hits_group, "pids", _numpy_sample(targets["paper_hit_particle_id"], idx))
        self.create_dataset(hits_group, "hids", _numpy_sample(targets["paper_hit_id"], idx))
        self.create_dataset(hits_group, "weight", _numpy_sample(targets["paper_hit_weight"], idx))

        self.create_dataset(parts_group, "pids", _numpy_sample(targets["paper_all_particle_id"], idx))
        self.create_dataset(parts_group, "pts", _numpy_sample(targets["paper_all_particle_pt"], idx))
        self.create_dataset(parts_group, "etas", _numpy_sample(targets["paper_all_particle_eta"], idx))
        self.create_dataset(parts_group, "phis", _numpy_sample(targets["paper_all_particle_phi"], idx))
        self.create_dataset(parts_group, "vzs", _numpy_sample(targets["paper_all_particle_vz"], idx))
        self.create_dataset(parts_group, "n_hits", _numpy_sample(targets["paper_all_particle_n_hits"], idx))

        self.create_dataset(preds_group, "class_preds", class_scores)
        if valid_prob is not None:
            self.create_dataset(preds_group, "track_valid_prob", valid_prob)
        self.create_dataset(preds_group, "masks", masks)
        if pred_iou is not None:
            self.create_dataset(preds_group, "query_iou", pred_iou)
        for name, value in regression.items():
            self.create_dataset(regression_group, name, value)

    def create_dataset(self, group, name, value, squeeze=False):
        if isinstance(value, np.ndarray):
            value_np = value
        else:
            value_np = tensor_to_numpy(value)
        if squeeze:
            value_np = np.squeeze(value_np)

        dataset_kwargs = {}
        if np.ndim(value_np) != 0:
            dataset_kwargs["compression"] = "lzf"

        group.create_dataset(name, data=value_np, **dataset_kwargs)

    def teardown(self, trainer, module, stage):
        if stage == "test" and self.file is not None:
            self.file.close()
            print("-" * 80)
            print("Created paper-compatible output file", self.output_path)
            print("-" * 80)
