"""scipy and lap1015_late find the same optimal pairs; they only differ in the order of the leftover queries."""

import numpy as np
import pytest
import torch
from test_matcher_prepare_on_device import DEVICES, PROCESS, SIZES, make_inputs

from hepattn.models.matcher import SOLVERS, Matcher

pytestmark = pytest.mark.skipif("lap1015_late" not in SOLVERS, reason="needs lap1015")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trim", [False, True])
def test_scipy_and_lap1015_give_the_same_pairs_and_leftovers(device, trim):
    rng = np.random.default_rng(20)
    kwargs = {**PROCESS, "prepare_on_device": True, "trim_padded_queries": trim}
    scipy_matcher = Matcher(default_solver="scipy", **kwargs)
    lap_matcher = Matcher(default_solver="lap1015_late", **kwargs)
    for num_pred, num_target in SIZES:
        costs, object_valid, query_valid = make_inputs(rng, 4, num_pred, num_target, device)
        a, b = scipy_matcher(costs, object_valid, query_valid), lap_matcher(costs, object_valid, query_valid)
        for k in range(len(a)):
            n_targets, n_queries = int(object_valid[k].sum()), int(query_valid[k].sum())
            # Same query for every target (random costs: no ties)
            assert torch.equal(a[k, :n_targets], b[k, :n_targets])
            # scipy always lists the leftovers in index order: valid queries, then padded ones
            leftovers = sorted(set(range(num_pred)) - set(a[k, :n_targets].tolist()))
            assert a[k, n_targets:].tolist() == leftovers
            if trim:
                # With trimming lap1015 also puts exactly the unmatched valid queries in the valid slots
                assert set(b[k, n_targets:n_queries].tolist()) == set(a[k, n_targets:n_queries].tolist())
                assert torch.equal(b[k, n_queries:], a[k, n_queries:])
