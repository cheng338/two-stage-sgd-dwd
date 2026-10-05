"""Adaptive candidates retain the exact finite parameter values supplied."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from two_stage_sgd_dwd import TwoStageDWDClassifier

_PATH = Path(__file__).resolve().parents[1] / 'examples' / '_adaptive_grid.py'
_SPEC = importlib.util.spec_from_file_location('exact_grid_subject', _PATH)
_GRID = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GRID)


def parameters(**changes):
    return {'kernel': 'rbf', 'lambd': 1., 'gamma': 1.,
            'retention_fraction': .3, **changes}


def receipt(params, score=.5):
    return {'params': params, 'mean_score': score, 'eligible': True}


@pytest.mark.parametrize('name,value', [
    ('lambd', 1.), ('gamma', 1.), ('retention_fraction', .3),
    ('lambd', math.ulp(0.)), ('gamma', math.ulp(0.)),
    ('lambd', math.nextafter(sys.float_info.max, 0.)),
    ('gamma', math.nextafter(sys.float_info.max, 0.)),
])
def test_adjacent_coordinates_keep_separate_keys_and_initial_proposals(name, value):
    adjacent = math.nextafter(value, math.inf)
    first, second = parameters(**{name: value}), parameters(**{name: adjacent})
    assert _GRID.candidate_key(first) != _GRID.candidate_key(second)
    grid = {key: [item] for key, item in first.items()}
    grid[name] = [value, adjacent]
    result = _GRID.next_search_stage([], grid, _GRID.AdaptiveSearchConfig())
    assert len(result['candidates']) == 2
    assert {p[name].hex() for p in result['candidates']} == {value.hex(), adjacent.hex()}
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert restored == result
    assert [_GRID.candidate_key(p) for p in restored['candidates']] == [
        _GRID.candidate_key(p) for p in result['candidates']]
    rows = [receipt(p) for p in restored['candidates']]
    assert len(_GRID._policy_results(rows, _GRID.AdaptiveSearchConfig())['params']) == 2


@pytest.mark.parametrize('value', [1, np.int32(1), np.int64(1), np.float32(.3),
                                  np.float64(.3), math.ulp(0.), sys.float_info.max])
def test_numeric_types_normalize_to_float_without_decimal_rounding(value):
    params = parameters(lambd=value)
    expected = float(value)
    assert _GRID.candidate_key(params)[0] == expected.hex()
    grid = {key: [item] for key, item in params.items()}
    result = _GRID.next_search_stage([], grid, _GRID.AdaptiveSearchConfig())
    normalized = result['candidates'][0]['lambd']
    assert type(normalized) is float and normalized.hex() == expected.hex()
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_preserved_retention_boundary_changes_actual_selected_count():
    fractions = [.3, math.nextafter(.3, math.inf)]
    grid = {key: [item] for key, item in parameters().items()}
    grid['retention_fraction'] = fractions
    result = _GRID.next_search_stage([], grid, _GRID.AdaptiveSearchConfig())
    candidates = result['candidates']
    assert [p['retention_fraction'] for p in candidates] == fractions
    X = np.arange(-5, 5, dtype=float).reshape(10, 1)
    y = np.where(X[:, 0] < 0, -1, 1)
    models = [TwoStageDWDClassifier(**p, max_epochs=1, stopping='fixed',
                                   random_state=13).fit(X, y) for p in candidates]
    assert [model.n_retained_ for model in models] == [3, 4]
    assert [len(model.selected_indices_) for model in models] == [3, 4]


def test_exact_keys_do_not_change_numeric_tie_break_order():
    # Hexadecimal strings have different lexical and numeric ordering here.
    rows = [receipt(parameters(lambd=lambd, gamma=gamma))
            for lambd in (1., 2.) for gamma in (1., 2.)]
    ordered = _GRID.ordered_eligible(rows)
    assert [(row['params']['lambd'], row['params']['gamma']) for row in ordered] == [
        (2., 1.), (2., 2.), (1., 1.), (1., 2.)]
    initial = {'kernel': ['rbf'], 'lambd': [1., 2.], 'gamma': [1., 2.],
               'retention_fraction': [.3]}
    answer = _GRID.next_search_stage(rows, initial, _GRID.AdaptiveSearchConfig())
    assert answer['state']['best_parameters'] == parameters(lambd=2., gamma=1.)


def test_adjacent_midpoints_exhaust_without_repeated_proposals():
    middle = math.nextafter(1., math.inf)
    values = [1., middle, math.nextafter(middle, math.inf)]
    initial = {'kernel': ['rbf'], 'lambd': values, 'gamma': values,
               'retention_fraction': [.3]}
    rows = [receipt(p, .9 if p['lambd'] == middle and p['gamma'] == middle else .5)
            for p in _GRID._parameter_grid(initial)]
    assert _GRID.propose_refinement(rows, [.3], top_k=1) == []
    config = _GRID.AdaptiveSearchConfig(top_k=1)
    answer = _GRID.next_search_stage(rows, initial, config)
    assert answer['review']['state'] == 'needs_review'
    assert 'parameter_limit' in answer['review']['stopping_reason']
    for _ in range(3):
        assert answer['candidates'] == []
        answer = _GRID.next_search_stage(rows, initial, config,
                                        json.loads(json.dumps(answer['state'])))
        assert answer['review']['state'] == 'needs_review'


@pytest.mark.parametrize('legacy', [None, 1])
def test_nonempty_legacy_policy_states_reject_without_mutating_inputs(legacy):
    initial = {key: [item] for key, item in parameters().items()}
    config = _GRID.AdaptiveSearchConfig()
    fresh = _GRID.next_search_stage([], initial, config, {})
    state = copy.deepcopy(fresh['state'])
    if legacy is None:
        state.pop('schema_version')
    else:
        state['schema_version'] = legacy
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='schema'):
        _GRID.next_search_stage([], initial, config, state)
    assert state == before
