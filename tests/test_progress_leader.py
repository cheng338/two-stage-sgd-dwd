"""Reporting leaders preserve exact ranking, durable recovery and selection."""
from contextlib import ExitStack
import copy
from fractions import Fraction
from itertools import permutations
import math
from unittest.mock import patch

import pytest

from test_tune_mnist import _RUNNER, Scores, arguments, read, records, synthetic


def count_record(identifier, numerator=3, denominator=10, *, eligible=True, **changes):
    params = dict(kernel='rbf', lambd=1., gamma=1., retention_fraction=.5) | changes
    folds = [dict(fold=i, eligible=eligible, status='trained' if eligible else 'single_class_selection',
                  accuracy=numerator / denominator if eligible else None) for i in range(2)]
    if eligible:
        for fold in folds:
            fold.update(n_correct=numerator, n_validation=denominator)
    return dict(record_id=identifier, params=params, folds=folds, eligible=eligible,
                mean_score=numerator / denominator if eligible else None)


def full_best(rows):
    ranked = _RUNNER.ordered_eligible(rows, expected_folds=2)
    return ranked[0] if ranked else None


def test_every_prefix_and_resume_position_matches_stable_exact_full_ranking():
    first = count_record('first')
    same = count_record('same-key')
    better = count_record('subulp-better', 300000000000000001, 10**18)
    assert first['mean_score'] == better['mean_score']
    assert _RUNNER.record_score(first) < _RUNNER.record_score(better)
    rows = [first, same, better, count_record('invalid', eligible=False)]
    for ordering in permutations(rows):
        ordering = list(ordering)
        before = copy.deepcopy(ordering)
        expected = [full_best(ordering[:stop]) for stop in range(1, len(ordering) + 1)]
        for prefix in range(len(ordering) + 1):
            leader = _RUNNER._ProgressLeader(ordering[:prefix], expected_folds=2)
            assert leader.best is (expected[prefix - 1] if prefix else None)
            for row, best in zip(ordering[prefix:], expected[prefix:]):
                assert leader.add(row) is best
        assert ordering == before


def test_parameter_ties_use_the_same_retention_lambda_gamma_order():
    rows = [count_record('ordinary'), count_record('retention', retention_fraction=.25),
            count_record('lambda', retention_fraction=.25, lambd=4.),
            count_record('gamma', retention_fraction=.25, lambd=4., gamma=.25)]
    for ordering in permutations(rows):
        leader = _RUNNER._ProgressLeader([], expected_folds=2)
        for stop, row in enumerate(ordering, 1):
            assert leader.add(row) is full_best(ordering[:stop])


def test_empty_failed_partial_and_custom_scalar_records_match_full_ranking():
    leader = _RUNNER._ProgressLeader([], expected_folds=2)
    assert leader.best is None
    failed = count_record('failed', eligible=False)
    partial = count_record('partial')
    partial['folds'].pop()
    scalar = count_record('scalar')
    for fold in scalar['folds']:
        del fold['n_correct'], fold['n_validation']
    mixed = count_record('mixed-counts')
    del mixed['folds'][0]['n_correct'], mixed['folds'][0]['n_validation']
    rows = [failed, partial, scalar, mixed]
    for stop, row in enumerate(rows, 1):
        assert leader.add(row) is full_best(rows[:stop])
    assert leader.best is scalar


@pytest.mark.parametrize('change', [dict(eligible='yes'), dict(params={}), dict(params=None)])
def test_malformed_new_records_raise_without_replacing_the_existing_leader(change):
    first = count_record('first')
    invalid = count_record('bad') | change
    leader = _RUNNER._ProgressLeader([first], expected_folds=2)
    with pytest.raises((TypeError, ValueError)):
        full_best([first, invalid])
    with pytest.raises((TypeError, ValueError)):
        leader.add(invalid)
    assert leader.best is first


class _RecomputedProgressLeader:
    """Test reference for the former per-append full ranking."""
    def __init__(self, completed, *, expected_folds):
        self._records = list(completed)
        self._expected_folds = expected_folds
        ranked = _RUNNER.ordered_eligible(self._records, expected_folds=expected_folds)
        self.best = ranked[0] if ranked else None

    def add(self, record):
        self._records.append(record)
        ranked = _RUNNER.ordered_eligible(self._records, expected_folds=self._expected_folds)
        self.best = ranked[0] if ranked else None
        return self.best


class ExactScores(Scores):
    def evaluate(self, params, *, completed_folds=None, on_fold_completed=None):
        self.calls.append(copy.deepcopy(params))
        key = 'control' if params['gamma'] == 'scale' else (params['lambd'], params['gamma'])
        if key == self.fail_once:
            self.fail_once = None
            raise self.fail_error
        eligible = self.mode != 'invalid' or key == 'control'
        if key == 'control':
            value = Fraction(99, 100)
        elif self.mode == 'ties':
            value = Fraction(3, 10)
        elif self.mode == 'subulp':
            value = Fraction(300000000000000000 + int(100 * params['lambd'] + 10 * params['gamma']), 10**18)
        else:
            score = max(.05, .9 - .04 * (math.log2(params['lambd'])**2 + math.log2(params['gamma'])**2))
            value = Fraction(score)
        record = count_record('synthetic', value.numerator, value.denominator,
                              eligible=eligible, **params)
        del record['record_id']
        record['execution_identity'] = copy.deepcopy(self.execution_identity)
        prefix = copy.deepcopy(completed_folds or [])
        record['folds'][:len(prefix)] = prefix
        if on_fold_completed is not None:
            for fold in record['folds'][len(prefix):]:
                on_fold_completed(copy.deepcopy(fold))
        return record


def normalized(value):
    # Timings and hashes derived from those timings necessarily differ between
    # sequential invocations; each run validates its own receipts independently.
    omitted = {'elapsed_seconds_not_a_benchmark', '_receipt', 'records_sha256', 'evidence_sha256'}
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def run_case(path, mode, interruption, accepted, leader_class):
    args = arguments(path, max_candidates=200 if accepted else 9)
    fake = ExactScores(mode=mode, fail_once=interruption)
    statuses, rank_calls = [], []
    write_json, rank = _RUNNER.write_json, _RUNNER.ordered_eligible

    def capture(path, value):
        if path.name == 'status.json' and value.get('state') == 'running':
            committed = records(args.output_dir)
            ranked = rank(committed, expected_folds=args.folds)
            expected = float(_RUNNER.record_score(ranked[0])) if ranked else None
            assert value['best_mean_accuracy'] == expected
            assert value['completed_candidates'] == len(committed)
            statuses.append(copy.deepcopy(value))
        return write_json(path, value)

    def count_rank(*args, **kwargs):
        rank_calls.append(1)
        return rank(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(synthetic(fake))
        stack.enter_context(patch.object(_RUNNER, '_ProgressLeader', leader_class))
        stack.enter_context(patch.object(_RUNNER, 'write_json', side_effect=capture))
        stack.enter_context(patch.object(_RUNNER, 'ordered_eligible', side_effect=count_rank))
        prefix = []
        if interruption is not None:
            with pytest.raises(RuntimeError, match='synthetic interruption'):
                _RUNNER.run_search(args)
            prefix = records(args.output_dir)
            args.resume = True
        selection = _RUNNER.run_search(args)
        if accepted:
            assert _RUNNER.load_qualified_selection(args.output_dir)['params'] == selection['params']
        final = records(args.output_dir)
        assert final[:len(prefix)] == prefix
        calls = copy.deepcopy(fake.calls)
        args.resume = True
        assert _RUNNER.run_search(args) == selection
        assert fake.calls == calls
    return dict(selection=normalized(selection), records=normalized(final), statuses=statuses,
                stages=read(args.output_dir / 'stages.json'), plan=read(args.output_dir / 'plan_state.json'),
                calls=calls, committed_prefix=len(prefix), rank_calls=len(rank_calls))


@pytest.mark.parametrize('mode, interruption, accepted', [
    ('peak', None, True), ('ties', None, False), ('subulp', None, False),
    ('invalid', None, False), ('peak', (1., 1.), False), ('peak', 'control', False),
])
def test_actual_runner_preserves_reporting_receipts_policy_and_resume(tmp_path, mode, interruption, accepted):
    reference = run_case(tmp_path / 'reference', mode, interruption, accepted, _RecomputedProgressLeader)
    candidate = run_case(tmp_path / 'candidate', mode, interruption, accepted, _RUNNER._ProgressLeader)
    assert candidate['rank_calls'] < reference['rank_calls']
    del reference['rank_calls'], candidate['rank_calls']
    assert candidate == reference
