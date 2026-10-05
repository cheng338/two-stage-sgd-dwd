"""Run ownership and candidate evidence checks use the public CV entry points."""
import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from test_tune_mnist import (_RUNNER, Scores, arguments, read, records, synthetic)


def _output_bytes(directory):
    # Windows excludes reads through a second handle while the lease is held.
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob('*')
            if path.is_file() and path.name != '.run.lock'}


def rewrite_records(directory, rows, *, refresh_hashes=False):
    path = directory / 'records.jsonl'
    path.write_text(''.join(json.dumps(row, allow_nan=True) + '\n' for row in rows),
                    encoding='utf-8')
    if not refresh_hashes:
        return
    # Valid hashes cannot make malformed candidate evidence trustworthy.
    selection_path = directory / 'selection.json'
    selection = read(selection_path)
    selection['records_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    selection_path.write_text(json.dumps(selection), encoding='utf-8')
    locked_path = directory / 'qualified-selection.json'
    locked = read(locked_path)
    for name in ('records.jsonl', 'selection.json'):
        locked['evidence_sha256'][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    locked_path.write_text(json.dumps(locked), encoding='utf-8')


def corrupt(row, kind):
    value = copy.deepcopy(row)
    if kind == 'not_mapping':
        return []
    if kind == 'missing_params':
        del value['params']
    elif kind == 'invalid_lambda':
        value['params']['lambd'] = 0.
    elif kind == 'invalid_kernel':
        value['params']['kernel'] = 'linear'
    elif kind == 'nonboolean_eligible':
        value['eligible'] = 1
    elif kind == 'nonfinite_mean':
        value['mean_score'] = float('inf')
    elif kind == 'mean_outside_accuracy_range':
        value['mean_score'] = 1.1
    elif kind == 'missing_fold':
        value['folds'].pop()
    elif kind == 'duplicate_fold':
        value['folds'][1]['fold'] = value['folds'][0]['fold']
    elif kind == 'fold_id_outside_range':
        value['folds'][1]['fold'] = len(value['folds'])
    elif kind == 'nonmapping_fold':
        value['folds'][0] = []
    elif kind == 'nonfinite_accuracy':
        value['folds'][0]['accuracy'] = float('nan')
    elif kind == 'accuracy_outside_range':
        value['folds'][0]['accuracy'] = -1.
    elif kind == 'fold_eligible_not_boolean':
        value['folds'][0]['eligible'] = 1
    elif kind == 'trained_status_disagreement':
        value['folds'][0]['status'] = 'single_class_selection'
    elif kind == 'mean_disagrees_with_folds':
        value['mean_score'] /= 2.
    elif kind == 'execution_identity_disagreement':
        value['execution_identity']['package_version'] = 'different'
    else:
        raise AssertionError(kind)
    return value


MALFORMED = (
    'not_mapping', 'missing_params', 'invalid_lambda', 'invalid_kernel',
    'nonboolean_eligible', 'nonfinite_mean', 'mean_outside_accuracy_range',
    'missing_fold', 'duplicate_fold', 'fold_id_outside_range',
    'nonmapping_fold', 'nonfinite_accuracy', 'accuracy_outside_range',
    'fold_eligible_not_boolean', 'trained_status_disagreement',
    'mean_disagrees_with_folds', 'execution_identity_disagreement',
)


def test_lease_excludes_reentry_and_releases_without_deleting_lock(tmp_path):
    path = tmp_path / '.run.lock'
    path.touch()  # An empty lock file left behind by a prior interrupted owner.
    with _RUNNER._RunLease(path):
        assert path.exists()
        with pytest.raises(_RUNNER.SearchBusyError):
            with _RUNNER._RunLease(path.parent / '.' / path.name):
                pytest.fail('A second owner acquired the same output lock')
    assert path.exists()
    with _RUNNER._RunLease(path):
        assert path.exists()


@pytest.mark.parametrize('failure', [RuntimeError('lease error'), KeyboardInterrupt(), SystemExit(3)])
def test_lease_released_on_normal_exceptions_and_interrupts(tmp_path, failure):
    path = tmp_path / '.run.lock'
    with pytest.raises(type(failure)):
        with _RUNNER._RunLease(path):
            raise failure
    with _RUNNER._RunLease(path):
        assert path.exists()


def test_qualified_loader_rejects_a_current_owner_without_writing(tmp_path):
    args, fake = arguments(tmp_path), Scores()
    with synthetic(fake):
        _RUNNER.run_search(args)
    before = _output_bytes(args.output_dir)
    with _RUNNER._RunLease(args.output_dir / '.run.lock'):
        with pytest.raises(_RUNNER.SearchBusyError):
            _RUNNER.load_qualified_selection(args.output_dir)
    assert _output_bytes(args.output_dir) == before


def test_second_coordinator_same_output_rejected_without_evaluation_or_writes(tmp_path):
    args, fake = arguments(tmp_path), Scores()
    evaluate = fake.evaluate
    checked = False

    def attempt_reentry(params):
        nonlocal checked
        if not checked:
            checked = True
            before = _output_bytes(args.output_dir)
            calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
            nested = copy.copy(args)
            nested.resume = True
            with pytest.raises(_RUNNER.SearchBusyError, match='busy or inaccessible'):
                _RUNNER.run_search(nested)
            assert fake.calls == calls and len(fake.constructed) == constructors
            assert _output_bytes(args.output_dir) == before
        return evaluate(params)

    with synthetic(fake), patch.object(fake, 'evaluate', side_effect=attempt_reentry):
        result = _RUNNER.run_search(args)
    assert checked and result['state'] == 'accepted'
    assert len(records(args.output_dir)) == len(fake.search_calls)


def test_independent_output_directories_can_run_while_first_is_owned(tmp_path):
    args, fake = arguments(tmp_path / 'first'), Scores()
    other_args, other = arguments(tmp_path / 'second'), Scores()
    evaluate = fake.evaluate
    nested_results = []

    def independent_run(params):
        if not nested_results:
            with synthetic(other):
                nested_results.append(_RUNNER.run_search(other_args))
        return evaluate(params)

    with synthetic(fake), patch.object(fake, 'evaluate', side_effect=independent_run):
        result = _RUNNER.run_search(args)
    assert result['state'] == nested_results[0]['state'] == 'accepted'
    assert len(fake.search_calls) == len(other.search_calls)
    assert len(records(args.output_dir)) == len(records(other_args.output_dir))


@pytest.mark.parametrize('failure', [RuntimeError('interrupted fixture'), KeyboardInterrupt(), SystemExit(7)])
def test_aborted_owner_releases_output_for_resume_and_completed_read(tmp_path, failure):
    args = arguments(tmp_path)
    fake = Scores(fail_once=(1., 1.), fail_error=failure)
    with synthetic(fake):
        with pytest.raises(type(failure)):
            _RUNNER.run_search(args)
        committed = records(args.output_dir)
        args.resume = True
        result = _RUNNER.run_search(args)
        assert result['state'] == 'accepted'
        assert records(args.output_dir)[:len(committed)] == committed
        calls = copy.deepcopy(fake.calls)
        assert _RUNNER.run_search(args) == result
        assert fake.calls == calls


@pytest.mark.parametrize('route', ['partial_resume', 'completed_resume', 'qualified_loader'])
def test_duplicate_candidates_reject_before_resume_or_qualified_shortcuts(tmp_path, route):
    args = arguments(tmp_path)
    fake = Scores(fail_once=(1., 1.) if route == 'partial_resume' else None)
    with synthetic(fake):
        if route == 'partial_resume':
            with pytest.raises(RuntimeError, match='synthetic interruption'):
                _RUNNER.run_search(args)
        else:
            _RUNNER.run_search(args)
        rows = records(args.output_dir)
        rewrite_records(args.output_dir, rows + [copy.deepcopy(rows[0])],
                        refresh_hashes=route != 'partial_resume')
        before = _output_bytes(args.output_dir)
        calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
        with pytest.raises(ValueError, match='[Dd]uplicate'):
            if route == 'qualified_loader':
                _RUNNER.load_qualified_selection(args.output_dir)
            else:
                args.resume = True
                _RUNNER.run_search(args)
        assert fake.calls == calls and len(fake.constructed) == constructors
        assert _output_bytes(args.output_dir) == before


@pytest.mark.parametrize('route', ['partial_resume', 'completed_resume', 'qualified_loader'])
@pytest.mark.parametrize('kind', MALFORMED)
def test_malformed_records_rejected_even_when_saved_hashes_match(tmp_path, route, kind):
    args = arguments(tmp_path)
    fake = Scores(fail_once=(1., 1.) if route == 'partial_resume' else None)
    with synthetic(fake):
        if route == 'partial_resume':
            with pytest.raises(RuntimeError, match='synthetic interruption'):
                _RUNNER.run_search(args)
        else:
            _RUNNER.run_search(args)
        rows = records(args.output_dir)
        rows[0] = corrupt(rows[0], kind)
        rewrite_records(args.output_dir, rows, refresh_hashes=route != 'partial_resume')
        before = _output_bytes(args.output_dir)
        calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
        with pytest.raises(ValueError):
            if route == 'qualified_loader':
                _RUNNER.load_qualified_selection(args.output_dir)
            else:
                args.resume = True
                _RUNNER.run_search(args)
        assert fake.calls == calls and len(fake.constructed) == constructors
        assert _output_bytes(args.output_dir) == before


@pytest.mark.parametrize('kind', MALFORMED)
def test_invalid_new_evaluator_record_rejected_before_candidate_commit(tmp_path, kind):
    args, fake = arguments(tmp_path), Scores()
    evaluate = fake.evaluate

    def malformed_candidate(params):
        result = evaluate(params)
        return result if params['gamma'] == 'scale' else corrupt(result, kind)

    with synthetic(fake), patch.object(fake, 'evaluate', side_effect=malformed_candidate):
        with pytest.raises(ValueError):
            _RUNNER.run_search(args)
    assert records(args.output_dir) == []
    assert read(args.output_dir / 'status.json')['state'] == 'failed'
    assert read(args.output_dir / 'failure.json')['error_type'] == 'ValueError'
    assert len(fake.search_calls) == 1


def test_ordinary_ineligible_fold_receipts_still_finish_as_needs_review(tmp_path):
    args, fake = arguments(tmp_path), Scores(mode='invalid')
    with synthetic(fake):
        result = _RUNNER.run_search(args)
        args.resume = True
        assert _RUNNER.run_search(args) == result
    assert result['state'] == 'needs_review'
    assert all(not row['eligible'] and row['mean_score'] is None
               for row in records(args.output_dir))
    assert all(fold['status'] == 'single_class_selection'
               for row in records(args.output_dir) for fold in row['folds'])


def test_mixed_trained_and_ineligible_folds_remain_legitimate_evidence(tmp_path):
    args, fake = arguments(tmp_path), Scores(mode='invalid')
    evaluate = fake.evaluate

    def mixed(params):
        result = evaluate(params)
        if params['gamma'] != 'scale':
            result['folds'][0].update(eligible=True, accuracy=.5, status='trained')
        return result

    with synthetic(fake), patch.object(fake, 'evaluate', side_effect=mixed):
        result = _RUNNER.run_search(args)
        args.resume = True
        assert _RUNNER.run_search(args) == result
    assert result['state'] == 'needs_review'
    assert all(not row['eligible'] and row['mean_score'] is None
               and row['folds'][0]['eligible'] and not row['folds'][1]['eligible']
               for row in records(args.output_dir))


@pytest.mark.parametrize('corruption', ['duplicate', 'malformed'])
def test_invalid_prior_evidence_is_not_repaired_as_an_interrupted_tail(tmp_path, corruption):
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.))
    with synthetic(fake):
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        rows = records(args.output_dir)
        if corruption == 'duplicate':
            rows.append(copy.deepcopy(rows[0]))
        else:
            rows[0] = corrupt(rows[0], 'missing_fold')
        rewrite_records(args.output_dir, rows)
        path = args.output_dir / 'records.jsonl'
        path.write_bytes(path.read_bytes() + b'{"params":')
        before = _output_bytes(args.output_dir)
        calls = copy.deepcopy(fake.calls)
        args.resume = True
        with pytest.raises(ValueError):
            _RUNNER.run_search(args)
        assert fake.calls == calls and _output_bytes(args.output_dir) == before
        assert not (args.output_dir / 'records-interrupted.jsonl').exists()


@pytest.mark.parametrize('route', ['completed_resume', 'qualified_loader'])
@pytest.mark.parametrize('kind', ['empty', 'all_ineligible'])
def test_accepted_selection_requires_some_eligible_candidate_evidence(tmp_path, route, kind):
    args, fake = arguments(tmp_path), Scores()
    with synthetic(fake):
        _RUNNER.run_search(args)
        rows = records(args.output_dir) if kind == 'all_ineligible' else []
        for row in rows:
            row.update(eligible=False, mean_score=None)
            for fold in row['folds']:
                fold.update(eligible=False, accuracy=None, status='single_class_selection')
        rewrite_records(args.output_dir, rows, refresh_hashes=True)
        before = _output_bytes(args.output_dir)
        calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
        with pytest.raises(ValueError):
            if route == 'qualified_loader':
                _RUNNER.load_qualified_selection(args.output_dir)
            else:
                args.resume = True
                _RUNNER.run_search(args)
        assert fake.calls == calls and len(fake.constructed) == constructors
        assert _output_bytes(args.output_dir) == before
