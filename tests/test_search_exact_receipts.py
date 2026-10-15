"""Exact score evidence is preserved through runner exports and recovery."""
from copy import deepcopy
from fractions import Fraction
import json

import pytest

from test_tune_mnist import _RUNNER, Scores, arguments, read, records, synthetic
from test_receipt_integrity import refresh_outer_hashes, reject_unchanged


class CountScores(Scores):
    def evaluate(self, params, *, completed_folds=None, on_fold_completed=None):
        result = super().evaluate(params, completed_folds=completed_folds)
        for fold in result['folds']:
            if fold['eligible']:
                correct, size = fold['accuracy'].as_integer_ratio()
                fold.update(n_correct=correct, n_validation=size)
        if on_fold_completed is not None:
            for fold in result['folds'][len(completed_folds or []):]:
                on_fold_completed(deepcopy(fold))
        return result


def test_exact_count_export_and_resume_use_same_winner_without_refitting(tmp_path):
    args, fake = arguments(tmp_path), CountScores(fail_once=(4., 4.))
    with synthetic(fake):
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        committed = records(args.output_dir)
        prior_calls = list(fake.calls)
        args.resume = True
        selection = _RUNNER.run_search(args)
        assert selection['state'] == 'accepted'
        assert selection['cv_score'] == _RUNNER.score_token(Fraction(.9))
        assert selection['mean_cv_accuracy'] == .9
        assert records(args.output_dir)[:len(committed)] == committed
        # The interrupted candidate was not committed; each committed candidate
        # is absent from the calls performed after recovery.
        new_calls = fake.calls[len(prior_calls):]
        assert not any(row['params'] in new_calls for row in committed)
        calls = list(fake.calls)
        locked = _RUNNER.load_qualified_selection(args.output_dir)
        assert locked['params'] == selection['params']
        assert fake.calls == calls
        config = read(args.output_dir / 'config.json')
        assert config['schema_version'] == 9
        assert '_adaptive_grid.py' in config['implementation_sha256']
        assert '_mnist_cv.py' in config['implementation_sha256']
        assert config['accuracy_comparison'] == 'exact_equal_weight_fold_fractions_when_counts_available'


@pytest.mark.parametrize('field,value', [('n_correct', True), ('n_correct', -1),
    ('n_correct', 3), ('n_validation', 0), ('n_validation', True), ('accuracy', .75)])
def test_saved_fold_count_mismatch_rejected_before_reuse(field, value):
    fold = dict(fold=0, eligible=True, status='trained', accuracy=.5, n_correct=1, n_validation=2)
    fold[field] = value
    with pytest.raises(ValueError, match='accuracy|counts'):
        _RUNNER._validate_fold_prefix([fold], 2)


@pytest.mark.parametrize('mutation', ['score', 'fraction', 'winner'])
def test_export_cannot_qualify_incorrect_winner_even_with_refreshed_outer_hashes(tmp_path, mutation):
    args, fake = arguments(tmp_path), CountScores()
    with synthetic(fake):
        _RUNNER.run_search(args)
        selection = read(args.output_dir / 'selection.json')
        if mutation == 'score':
            selection['mean_cv_accuracy'] = .8
        elif mutation == 'fraction':
            selection['cv_score'] = {'numerator': 1, 'denominator': 2}
        else:
            replacement = next(row for row in records(args.output_dir)
                               if row['params'] != selection['params'])
            selection['params'] = replacement['params']
            locked = read(args.output_dir / 'qualified-selection.json')
            locked['params'] = replacement['params']
            (args.output_dir / 'qualified-selection.json').write_text(json.dumps(locked))
        (args.output_dir / 'selection.json').write_text(json.dumps(selection))
        refresh_outer_hashes(args.output_dir)
        reject_unchanged(args, fake, route='qualified', match='disagrees')
