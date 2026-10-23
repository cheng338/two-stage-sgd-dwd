"""Boundary witnesses for the established binary64 retained-count contract."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from two_stage_sgd_dwd._core import select_near_boundary


@pytest.mark.parametrize(
    "fraction, expected_count",
    [
        (0.29, 28),
        (np.nextafter(0.29, 0.0), 28),
        (np.nextafter(0.29, 1.0), 29),
        (0.30, 30),
    ],
)
def test_binary64_floor_boundary_preserves_rank_and_random_draws(fraction, expected_count):
    scores = np.arange(100, dtype=np.float64)
    original = scores.copy()
    rng = np.random.default_rng(2900)
    reference = np.random.default_rng(2900)
    reference.random(100)

    selected = select_near_boundary(scores, fraction, rng)

    assert_array_equal(selected, np.arange(expected_count))
    assert_array_equal(scores, original)
    assert_array_equal(rng.random(20), reference.random(20))
