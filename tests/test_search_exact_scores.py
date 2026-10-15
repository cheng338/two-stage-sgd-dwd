"""Accuracy evidence and finite-arithmetic search regressions; no fitting."""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import importlib.util
from itertools import permutations, product
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1] / 'examples' / '_adaptive_grid.py'
_SPEC = importlib.util.spec_from_file_location('exact_search_test_subject', _PATH)
search = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(search)


def params(lambd=4., gamma=4., retention=.5):
    return dict(kernel='rbf', lambd=lambd, gamma=gamma, retention_fraction=retention)


def record(parameters, numerators=(1, 2, 3, 4, 5)):
    folds = [dict(fold=i, eligible=True, status='trained', n_correct=digit * (i + 1),
                  n_validation=10 * (i + 1), accuracy=digit / 10.)
             for i, digit in enumerate(numerators)]
    return dict(params=parameters, folds=folds, eligible=True,
                mean_score=float(np.mean([fold['accuracy'] for fold in folds])))


def perturbed_equal_records():
    rows = [record(params(), digits) for digits in permutations((1, 2, 3, 4, 5))]
    return min(rows, key=lambda row: row['mean_score']), max(rows, key=lambda row: row['mean_score'])


def test_exact_unweighted_mean_and_all_tie_breaks():
    low, high = perturbed_equal_records()
    assert low['mean_score'] < high['mean_score']
    assert search.record_score(low) == search.record_score(high) == Fraction(3, 10)
    # Unequal validation sizes must not change the statistic to pooled accuracy.
    assert Fraction(sum(f['n_correct'] for f in low['folds']),
                    sum(f['n_validation'] for f in low['folds'])) != Fraction(3, 10)
    high['params'] = params(retention=1.)
    low['params'] = params(lambd=8., gamma=2.)
    other = copy.deepcopy(low)
    other['params'] = params(lambd=8., gamma=4.)
    weaker = copy.deepcopy(low)
    weaker['params'] = params(lambd=4., gamma=1.)
    rows = [high, weaker, other, low]
    assert search.ordered_eligible(rows, expected_folds=5) == [low, other, weaker, high]
    results = search._policy_results(rows, search.AdaptiveSearchConfig(expected_folds=5))
    assert search.ranked_valid_indices(results) == [3, 2, 1, 0]
    assert search.boundary_axes([low, high], tie_tol=0.)['lambd'] == ['lower', 'upper']


@pytest.mark.parametrize('bad', [True, -1, 11, 1.0, None])
def test_invalid_correct_counts_rejected(bad):
    row = record(params())
    row['folds'][0]['n_correct'] = bad
    assert not search.valid_record(row, expected_folds=5)
    with pytest.raises(ValueError):
        search.record_score(row)


@pytest.mark.parametrize('bad', [True, 0, -1, 10., None])
def test_invalid_sample_counts_rejected(bad):
    row = record(params())
    row['folds'][0]['n_validation'] = bad
    assert not search.valid_record(row, expected_folds=5)


def test_partial_counts_or_inconsistent_accuracy_are_not_valid_evidence():
    for mutation in ('one_count', 'one_fold', 'accuracy', 'mean'):
        row = record(params())
        if mutation == 'one_count':
            row['folds'][0].pop('n_correct')
        elif mutation == 'one_fold':
            row['folds'][0].pop('n_correct')
            row['folds'][0].pop('n_validation')
        elif mutation == 'accuracy':
            row['folds'][0]['accuracy'] = .2
        else:
            row['mean_score'] = .9
        assert not search.valid_record(row, expected_folds=5)


def test_scalar_only_custom_scores_keep_original_behavior():
    rows = [dict(params=params(lambd=lambd), eligible=True, mean_score=score)
            for lambd, score in ((8., .3), (4., math.nextafter(.3, 1.)))]
    assert search.ordered_eligible(rows)[0] is rows[1]
    assert search.score_token(search.record_score(rows[1])) == {'score': rows[1]['mean_score']}


@pytest.mark.parametrize('real_improvement', [False, True])
def test_roundtrip_state_distinguishes_equal_counts_from_improvement(real_improvement):
    low, high = perturbed_equal_records()
    initial = dict(lambd=[2., 4., 8.], gamma=[2., 4., 8.], retention_fraction=[.5])
    config = search.AdaptiveSearchConfig(top_k=1, expected_folds=5)
    lo_digits = [round(f['accuracy'] * 10) for f in low['folds']]
    hi_digits = [round(f['accuracy'] * 10) for f in high['folds']]
    rows = [record(params(l, g), lo_digits if (l, g) == (4., 4.) else [1] * 5)
            for l, g in product(initial['lambd'], initial['gamma'])]
    proposal = search.next_search_stage(rows, initial, config)
    assert proposal['phase'] == 'local'
    peak = next(p for p in proposal['candidates'] if p['lambd'] > 4. and p['gamma'] == 4.)
    for p in proposal['candidates']:
        added = record(p, hi_digits if p == peak else [1] * 5)
        if real_improvement and p == peak:
            size = 10**18
            added['folds'] = [dict(fold=i, eligible=True, status='trained', n_correct=3*size//10+1,
                                    n_validation=size, accuracy=.3) for i in range(5)]
            added['mean_score'] = .3
        rows.append(added)
    resumed = json.loads(json.dumps(proposal['state']))
    answer = search.next_search_stage(rows, initial, config, resumed)
    assert answer['state']['stable_rounds'] == (0 if real_improvement else 1)
    assert answer['state']['best_mean_score'] == .3
    token = answer['state']['best_score']
    assert (Fraction(token['numerator'], token['denominator']) > Fraction(3, 10)) == real_improvement
    assert json.loads(json.dumps(answer)) == answer


def test_old_policy_schema_requires_fresh_state():
    with pytest.raises(ValueError, match='schema'):
        search.next_search_stage([], dict(lambd=[1.], gamma=[1.], retention_fraction=[.3]),
                                 search.AdaptiveSearchConfig(), {'schema_version': 2})


def float_offset(value, count):
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    return struct.unpack('>d', struct.pack('>Q', bits + count))[0]


@pytest.mark.parametrize('start', [1e-300, 1e-8, 1e308])
def test_close_coordinate_repairs_match_independent_high_precision_reference(start):
    repaired_midpoints = repaired_boundaries = 0
    for offset, gap in product(range(32), (2, 3, 4, 8, 16)):
        lower, upper = float_offset(start, offset), float_offset(start, offset + gap)
        ordinary = math.sqrt(lower) * math.sqrt(upper)
        with localcontext() as ctx:
            ctx.prec = 250
            expected = float((Decimal.from_float(lower) * Decimal.from_float(upper)).sqrt())
        actual = search._midpoint(lower, upper)
        assert actual == (ordinary if lower < ordinary < upper else expected)
        assert lower < actual < upper
        assert actual in search._local_midpoints([lower, upper], lower.hex())
        repaired_midpoints += not lower < ordinary < upper
        for side in ('low', 'high'):
            edge, neighbor = (lower, upper) if side == 'low' else (upper, lower)
            step = abs(math.log(edge) - math.log(neighbor))
            ordinary = math.exp(math.log(edge) + (-step if side == 'low' else step))
            outward = ordinary < edge if side == 'low' else ordinary > edge
            exact_edge, exact_neighbor = Fraction(edge), Fraction(neighbor)
            factor = min(exact_neighbor / exact_edge if side == 'low'
                         else exact_edge / exact_neighbor, Fraction(10))
            expected = float(exact_edge / factor if side == 'low' else exact_edge * factor)
            assert search._outside([lower, upper], side) == (ordinary if outward else expected)
            repaired_boundaries += not outward
    assert repaired_boundaries
    if start != 1e308:
        assert repaired_midpoints


@pytest.mark.parametrize('start', [math.ulp(0.), 1e-300, 1e-8, 1., 1e308])
def test_adjacent_float_interval_remains_unrefinable(start):
    assert search._midpoint(start, math.nextafter(start, math.inf)) is None


@pytest.mark.parametrize('value,side', [(math.ulp(0.), 'low'), (sys.float_info.max, 'high')])
def test_actual_float_limits_remain_explicit(value, side):
    with pytest.raises(ValueError, match='representable'):
        search._outside([value], side)
    with pytest.raises(ValueError, match='represent'):
        search._outward([value], 'lower' if side == 'low' else 'upper', 'gamma')


def test_fallback_keeps_neighboring_parameter_identities_distinct():
    first = params(gamma=1e-8)
    second = dict(first, gamma=math.nextafter(first['gamma'], math.inf))
    assert search.candidate_key(first) != search.candidate_key(second)
    assert search.parameter_hash(first) != search.parameter_hash(second)
