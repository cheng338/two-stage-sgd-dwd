"""Timing summaries distinguish completed work from recovered measurements."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / 'examples' / '_mnist_timing.py'
_SPEC = importlib.util.spec_from_file_location('mnist_timing_test_subject', _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize = _MODULE.summarize_candidate_timing
COMPONENTS = ('linear_seconds_charged', 'selection_seconds',
              'kernel_seconds', 'score_seconds')


def record(cut=0):
    folds = []
    for index in range(3):
        folds.append(dict(fold=index, eligible=True, accuracy=.8, status='trained',
                          linear_seconds_charged=float(index + 1),
                          linear_seconds_reused=float(index + 1),
                          selection_seconds=.25, kernel_seconds=4., score_seconds=.5))
    return dict(folds=folds, replayed_fold_indices=list(range(cut)),
                elapsed_seconds_not_a_benchmark=12.5)


@pytest.mark.parametrize('cut', [0, 1, 2, 3])
def test_recovery_partition_preserves_records_and_component_totals(cut):
    original = record(cut)
    before = deepcopy(original)
    actual = summarize(original)
    assert original == before
    assert actual['scope'] == 'candidate_completion_evaluation'
    assert actual['completion_elapsed_seconds_not_a_benchmark'] == 12.5
    for name, expected in (('newly_evaluated_folds', list(range(cut, 3))),
                           ('replayed_folds', list(range(cut)))):
        group = actual[name]
        assert group['fold_indices'] == expected
        assert group['component_seconds'] == {
            field: sum(original['folds'][index][field] for index in expected)
            for field in COMPONENTS}
    for field in COMPONENTS:
        assert (actual['newly_evaluated_folds']['component_seconds'][field] +
                actual['replayed_folds']['component_seconds'][field] ==
                sum(fold[field] for fold in original['folds']))
    json.dumps(actual, allow_nan=False)
    actual['replayed_folds']['fold_indices'].append(999)
    actual['newly_evaluated_folds']['component_seconds']['kernel_seconds'] = 0
    assert original == before


def test_screening_cache_reference_is_not_charged_again():
    value = record(1)
    for fold in value['folds']:
        fold.update(linear_seconds_charged=0., linear_seconds_reused=1e100,
                    linear_cache_hit=True)
    actual = summarize(value)
    assert all(actual[name]['component_seconds']['linear_seconds_charged'] == 0
               for name in ('newly_evaluated_folds', 'replayed_folds'))


def test_selection_failure_still_has_completed_screening_and_selection_costs():
    value = record(1)
    value['folds'][0].update(eligible=False, status='single_class_selection',
                            accuracy=None, kernel_seconds=0., score_seconds=0.)
    value['folds'][2].update(eligible=False, status='single_class_selection',
                            accuracy=None, kernel_seconds=0., score_seconds=0.)
    actual = summarize(value)
    assert actual['replayed_folds']['component_seconds'] == {
        'linear_seconds_charged': 1., 'selection_seconds': .25,
        'kernel_seconds': 0., 'score_seconds': 0.}
    assert actual['newly_evaluated_folds']['component_seconds'] == {
        'linear_seconds_charged': 5., 'selection_seconds': .5,
        'kernel_seconds': 4., 'score_seconds': .5}


def test_inherited_control_has_no_fabricated_elapsed_time():
    value = record(2)
    value.pop('elapsed_seconds_not_a_benchmark')
    assert summarize(value)['completion_elapsed_seconds_not_a_benchmark'] is None


def test_all_replayed_is_zero_new_component_cost_not_zero_elapsed():
    actual = summarize(record(3))
    assert actual['newly_evaluated_folds']['fold_indices'] == []
    assert set(actual['newly_evaluated_folds']['component_seconds'].values()) == {0.}
    assert actual['completion_elapsed_seconds_not_a_benchmark'] == 12.5


def test_a_later_read_does_not_relabel_historical_completion_as_current_work():
    original = record()
    first = summarize(original)
    later = summarize(json.loads(json.dumps(original)))
    assert first == later
    assert first['scope'] == 'candidate_completion_evaluation'


def test_missing_legacy_replay_evidence_is_not_assumed_fresh():
    value = record()
    del value['replayed_fold_indices']
    with pytest.raises(ValueError):
        summarize(value)


@pytest.mark.parametrize('indices', [[1], [0, 0], [0, 2], [False], [0.],
                                    [-1], [0, 1, 2, 3], '0', None])
def test_invalid_replayed_prefix_is_rejected(indices):
    value = record()
    value['replayed_fold_indices'] = indices
    with pytest.raises(ValueError):
        summarize(value)


@pytest.mark.parametrize('indices', [[0, 0, 2], [1, 0, 2], [0, 2, 3],
                                    [0, False, 2], [0, 1., 2]])
def test_missing_duplicate_or_reordered_fold_identity_is_rejected(indices):
    value = record()
    for fold, index in zip(value['folds'], indices):
        fold['fold'] = index
    with pytest.raises(ValueError):
        summarize(value)


@pytest.mark.parametrize('component', COMPONENTS)
@pytest.mark.parametrize('invalid', [-1., float('nan'), float('inf'), True, '1', None])
def test_invalid_component_measurement_is_rejected(component, invalid):
    value = record(1)
    value['folds'][0][component] = invalid
    with pytest.raises(ValueError):
        summarize(value)


@pytest.mark.parametrize('component', COMPONENTS)
def test_missing_measurement_is_not_treated_as_zero(component):
    value = record()
    del value['folds'][2][component]
    with pytest.raises(ValueError):
        summarize(value)


@pytest.mark.parametrize('invalid', [-1., float('nan'), float('inf'), True, '1', None])
def test_present_but_invalid_elapsed_is_rejected(invalid):
    value = record()
    value['elapsed_seconds_not_a_benchmark'] = invalid
    with pytest.raises(ValueError):
        summarize(value)


def test_large_finite_components_remain_finite():
    value = record(1)
    for fold in value['folds']:
        fold['kernel_seconds'] = 1e300
    actual = summarize(value)
    assert actual['newly_evaluated_folds']['component_seconds']['kernel_seconds'] == 2e300
    assert actual['replayed_folds']['component_seconds']['kernel_seconds'] == 1e300
    json.dumps(actual, allow_nan=False)


def test_unrepresentable_sum_is_not_exported_as_infinity():
    value = record()
    for fold in value['folds']:
        fold['kernel_seconds'] = 1e308
    with pytest.raises(ValueError):
        summarize(value)


@pytest.mark.parametrize('value', [None, [], {'folds': [], 'replayed_fold_indices': []},
                                   {'folds': [None], 'replayed_fold_indices': []}])
def test_invalid_record_structure_is_rejected(value):
    with pytest.raises(ValueError):
        summarize(value)
