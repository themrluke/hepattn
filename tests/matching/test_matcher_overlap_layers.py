import numpy as np
import pytest
import torch

from hepattn.models import matcher as matcher_module
from hepattn.models.matcher import SOLVERS, Matcher

# The original Matcher (git ef88a68) and the random inputs are shared with the prepare_on_device tests
from test_matcher_prepare_on_device import DEVICES, PROCESS, SIZES, make_inputs, original_matcher  # noqa: F401

NUM_LAYERS = 4


def make_layer_inputs(rng: np.random.Generator, batch_size: int, num_pred: int, num_target: int, device: str):
    """One cost matrix per layer, all against the same targets and queries, as in MaskFormer."""
    costs, object_valid, query_valid = make_inputs(rng, batch_size, num_pred, num_target, device)
    padded = ~object_valid.unsqueeze(1).expand(-1, num_pred, -1)
    layer_costs = [costs]
    for _ in range(NUM_LAYERS - 1):
        layer = torch.from_numpy(rng.random((batch_size, num_pred, num_target), dtype=np.float32) * 10).to(device)
        layer[padded] = float("nan")  # padded targets as in the first layer
        layer_costs.append(layer)
    return layer_costs, object_valid, query_valid


def stacked(layer_costs, object_valid, query_valid):
    """What MaskFormer._match_and_permute_outputs hands the matcher: the layers stacked along the batch."""
    costs = torch.stack(layer_costs).flatten(0, 1)
    return costs, object_valid.repeat(NUM_LAYERS, 1), query_valid.repeat(NUM_LAYERS, 1)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_layerwise_identical_to_original(original_matcher, solver, device, dtype, batch_size):  # noqa: F811
    rng = np.random.default_rng(10)
    original = original_matcher(default_solver=solver, **PROCESS)
    prepared = Matcher(default_solver=solver, **PROCESS, prepare_on_device=True)
    overlap = Matcher(default_solver=solver, **PROCESS, prepare_on_device=True, overlap_layers=True)
    for num_pred, num_target in SIZES:
        layer_costs, object_valid, query_valid = make_layer_inputs(rng, batch_size, num_pred, num_target, device)
        layer_costs = [c.to(dtype) for c in layer_costs]
        expected = original(*stacked(layer_costs, object_valid, query_valid)).view(NUM_LAYERS, batch_size, num_pred)

        matching = overlap.start_layerwise(object_valid, num_slots=NUM_LAYERS)
        for slot, costs in enumerate(layer_costs):
            matching.submit(slot, costs, query_valid)
        got = matching.results()

        assert len(got) == NUM_LAYERS
        for layer in range(NUM_LAYERS):
            assert got[layer].dtype == expected.dtype
            assert torch.equal(got[layer], expected[layer])  # the full permutation, unmatched queries included
        assert torch.equal(torch.stack(got).flatten(0, 1), prepared(*stacked(layer_costs, object_valid, query_valid)))


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_layerwise_trim_identical_to_stacked_trim(solver, device, batch_size):
    rng = np.random.default_rng(14)
    kwargs = {"default_solver": solver, **PROCESS, "prepare_on_device": True, "trim_padded_queries": True}
    stacked_trim, overlap_trim = Matcher(**kwargs), Matcher(**kwargs, overlap_layers=True)
    for num_pred, num_target in SIZES:
        layer_costs, object_valid, query_valid = make_layer_inputs(rng, batch_size, num_pred, num_target, device)
        expected = stacked_trim(*stacked(layer_costs, object_valid, query_valid)).view(NUM_LAYERS, batch_size, num_pred)
        matching = overlap_trim.start_layerwise(object_valid, num_slots=NUM_LAYERS)
        for slot, costs in enumerate(layer_costs):
            matching.submit(slot, costs, query_valid)
        for layer, got in enumerate(matching.results()):
            assert torch.equal(got, expected[layer])


@pytest.mark.parametrize("device", DEVICES)
def test_layerwise_trim_falls_back_and_keeps_enough_columns(original_matcher, device):  # noqa: F811
    rng = np.random.default_rng(15)
    kwargs = {"default_solver": "lap1015_late" if "lap1015_late" in SOLVERS else "scipy", **PROCESS}
    overlap_trim = Matcher(**kwargs, prepare_on_device=True, trim_padded_queries=True, overlap_layers=True)
    stacked_trim = Matcher(**kwargs, prepare_on_device=True, trim_padded_queries=True)
    layer_costs, object_valid, query_valid = make_layer_inputs(rng, 1, 100, 100, device)
    cases = {
        "valid queries last: can't trim": (object_valid, query_valid.flip(-1)),
        "fewer valid queries than targets": (torch.arange(100, device=device) < 60, torch.arange(100, device=device) < 40),
    }
    for object_valid, query_valid in cases.values():
        object_valid, query_valid = object_valid.reshape(1, -1), query_valid.reshape(1, -1)
        costs = [c.nan_to_num(1.0) for c in layer_costs]
        matching = overlap_trim.start_layerwise(object_valid, num_slots=NUM_LAYERS)
        for slot, c in enumerate(costs):
            matching.submit(slot, c, query_valid)
        assert torch.equal(torch.cat(matching.results()), stacked_trim(*stacked(costs, object_valid, query_valid)))


@pytest.mark.parametrize("device", DEVICES)
def test_layerwise_without_query_mask_and_fewer_slots(original_matcher, device):  # noqa: F811
    rng = np.random.default_rng(11)
    layer_costs, object_valid, _ = make_layer_inputs(rng, 1, 90, 90, device)
    matching = Matcher(default_solver="scipy", **PROCESS, prepare_on_device=True, overlap_layers=True).start_layerwise(object_valid, num_slots=6)
    for slot, costs in enumerate(layer_costs):  # 4 of the 6 reserved slots, like a layer without intermediate tasks
        matching.submit(slot, costs)
    got = torch.cat(matching.results())
    expected = original_matcher(default_solver="scipy", **PROCESS)(torch.cat(layer_costs), object_valid.repeat(NUM_LAYERS, 1))
    assert torch.equal(got, expected)


def test_start_layerwise_leaves_adaptive_checks_to_forward():
    kwargs = {**PROCESS, "default_solver": "scipy", "adaptive_solver": True, "adaptive_check_interval": 3}
    matcher = Matcher(**kwargs, prepare_on_device=True, overlap_layers=True)
    rng = np.random.default_rng(12)
    costs, object_valid, query_valid = make_inputs(rng, 2, 40, 40, "cpu", pad_value=1e4)
    used_forward = []
    for _ in range(7):
        matching = matcher.start_layerwise(object_valid, num_slots=1)
        if matching is None:
            matcher(costs, object_valid, query_valid)
        else:
            matching.submit(0, costs, query_valid)
            matching.results()
        used_forward.append(matching is None)
    # Same steps as forward alone would check the solvers on: 0, 3, 6
    assert used_forward == [True, False, False, True, False, False, True]
    assert matcher.step == 7


def test_start_layerwise_is_off_by_default():
    matcher = Matcher(default_solver="scipy", **PROCESS, prepare_on_device=True)
    assert matcher.start_layerwise(torch.ones(1, 10, dtype=torch.bool), num_slots=4) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_layerwise_pins_the_buffer_once():
    rng = np.random.default_rng(13)
    matcher = Matcher(default_solver="scipy", **PROCESS, prepare_on_device=True, overlap_layers=True)
    addresses = []
    for _ in range(2):
        layer_costs, object_valid, query_valid = make_layer_inputs(rng, 1, 70, 70, "cuda")
        matching = matcher.start_layerwise(object_valid, num_slots=NUM_LAYERS)
        for slot, costs in enumerate(layer_costs):
            matching.submit(slot, costs, query_valid)
        matching.results()
        addresses.append(matcher_module._PINNED_ADDRESS)
    assert addresses[0] is not None
    assert addresses[0] == addresses[1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prepare_on_device": False}, "overlap_layers needs"),
        ({"prepare_on_device": True, "parallel_backend": "thread"}, "overlap_layers needs"),
    ],
)
def test_overlap_layers_needs_prepare_on_device(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Matcher(default_solver="scipy", **{**PROCESS, **kwargs}, overlap_layers=True)
