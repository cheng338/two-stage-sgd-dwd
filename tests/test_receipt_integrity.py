"""Committed CV receipts detect changed evidence before any reuse or repair."""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from test_tune_mnist import _RUNNER, Scores, arguments, read, records, synthetic


class DetailedScores(Scores):
    """Synthetic receipts include the same kinds of auxiliary fit evidence."""
    def evaluate(self, params):
        result = super().evaluate(params)
        for fold in result['folds']:
            fold.update(selected_ids_sha256='a' * 64, linear_seconds_charged=.125,
                        kernel_seconds=.5, kernel_updates=12)
        return result


def output_bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob('*') if path.is_file() and path.name != '.run.lock'}


def write_rows(directory, rows):
    (directory / 'records.jsonl').write_text(
        ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')


def refresh_outer_hashes(directory):
    # This exercises the earlier per-receipt guard even with current export hashes.
    selection_path = directory / 'selection.json'
    selection = read(selection_path)
    selection['records_sha256'] = hashlib.sha256((directory / 'records.jsonl').read_bytes()).hexdigest()
    selection_path.write_text(json.dumps(selection), encoding='utf-8')
    locked_path = directory / 'qualified-selection.json'
    locked = read(locked_path)
    for name in locked['evidence_sha256']:
        locked['evidence_sha256'][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    locked_path.write_text(json.dumps(locked), encoding='utf-8')


@contextmanager
def started(path, *, completed=False, **options):
    args = arguments(path, **options)
    fake = DetailedScores(fail_once=None if completed else (4., 4.))
    with synthetic(fake):
        if completed:
            _RUNNER.run_search(args)
        else:
            with pytest.raises(RuntimeError, match='synthetic interruption'):
                _RUNNER.run_search(args)
        yield args, fake


def reject_unchanged(args, fake, route='resume', match=None):
    before = output_bytes(args.output_dir)
    calls, constructions = copy.deepcopy(fake.calls), len(fake.constructed)
    with pytest.raises(ValueError, match=match):
        if route == 'qualified':
            _RUNNER.load_qualified_selection(args.output_dir)
        else:
            args.resume = True
            _RUNNER.run_search(args)
    assert fake.calls == calls and len(fake.constructed) == constructions
    assert output_bytes(args.output_dir) == before


def run_hash(directory):
    return _RUNNER._run_sha256(read(directory / 'config.json'), read(directory / 'splits.json'))


def test_clean_receipts_bind_all_fields_and_resume_without_repeating_committed_candidates(tmp_path):
    with started(tmp_path) as (args, fake):
        saved = records(args.output_dir)
        assert len(saved) == 8
        root = run_hash(args.output_dir)
        previous = root
        for index, row in enumerate(saved):
            receipt = row['_receipt']
            assert receipt['schema_version'] == 1
            assert receipt['run_sha256'] == root
            assert receipt['sequence'] == index
            assert receipt['previous_sha256'] == previous
            assert receipt['candidate_key'] == list(_RUNNER.candidate_key(row['params']))
            content = copy.deepcopy(row)
            del content['_receipt']['sha256']
            # Independent encoding checks the persisted contract, not the writer helper.
            raw = json.dumps(content, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode('utf-8')
            assert receipt['sha256'] == hashlib.sha256(raw).hexdigest()
            previous = receipt['sha256']
        args.resume = True
        result = _RUNNER.run_search(args)
        assert result['state'] == 'accepted' and result['mean_cv_accuracy'] == .9
        assert records(args.output_dir)[:len(saved)] == saved
        calls, constructions = copy.deepcopy(fake.calls), len(fake.constructed)
        assert _RUNNER.run_search(args) == result
        assert _RUNNER.load_qualified_selection(args.output_dir)['params'] == result['params']
        assert fake.calls == calls and len(fake.constructed) == constructions


@pytest.mark.parametrize('kind', ['coherent_score', 'adjacent_parameter', 'elapsed_time',
                                 'selected_ids', 'fold_time', 'fold_updates', 'stage'])
def test_semantically_consistent_record_edit_rejected_before_resume(tmp_path, kind):
    with started(tmp_path) as (args, fake):
        rows = records(args.output_dir)
        row = rows[4]
        if kind == 'coherent_score':
            row['mean_score'] = .99
            for fold in row['folds']:
                fold['accuracy'] = .99
        elif kind == 'adjacent_parameter':
            row['params']['lambd'] = math.nextafter(row['params']['lambd'], math.inf)
        elif kind == 'elapsed_time':
            row['elapsed_seconds_not_a_benchmark'] += 1.
        elif kind == 'selected_ids':
            row['folds'][0]['selected_ids_sha256'] = 'b' * 64
        elif kind == 'fold_time':
            row['folds'][0]['kernel_seconds'] += 1.
        elif kind == 'fold_updates':
            row['folds'][0]['kernel_updates'] += 1
        else:
            row['stage'] = 'different-stage'
        write_rows(args.output_dir, rows)
        reject_unchanged(args, fake, match='[Rr]eceipt|[Cc]hecksum')


@pytest.mark.parametrize('route', ['resume', 'qualified'])
def test_completed_results_still_require_original_candidate_receipts(tmp_path, route):
    with started(tmp_path, completed=True) as (args, fake):
        rows = records(args.output_dir)
        rows[0]['folds'][0]['kernel_seconds'] += 1.
        write_rows(args.output_dir, rows)
        refresh_outer_hashes(args.output_dir)
        reject_unchanged(args, fake, route, match='[Rr]eceipt|[Cc]hecksum')


@pytest.mark.parametrize('kind', ['missing', 'bad_hash', 'wrong_hash', 'extra_field',
                                 'sequence_bool', 'sequence_float', 'schema_bool',
                                 'wrong_schema', 'candidate_key', 'previous_hash'])
def test_malformed_or_inconsistent_receipt_rejected(tmp_path, kind):
    with started(tmp_path) as (args, fake):
        rows = records(args.output_dir)
        receipt = rows[0]['_receipt']
        if kind == 'missing':
            del rows[0]['_receipt']
        elif kind == 'bad_hash':
            receipt['sha256'] = 'not-a-sha256'
        elif kind == 'wrong_hash':
            receipt['sha256'] = 'f' * 64
        elif kind == 'extra_field':
            receipt['unknown'] = 1
        elif kind == 'sequence_bool':
            receipt['sequence'] = False
        elif kind == 'sequence_float':
            receipt['sequence'] = 0.
        elif kind == 'schema_bool':
            receipt['schema_version'] = True
        elif kind == 'wrong_schema':
            receipt['schema_version'] = 99
        elif kind == 'candidate_key':
            receipt['candidate_key'][0] = math.nextafter(rows[0]['params']['lambd'], math.inf).hex()
        else:
            receipt['previous_sha256'] = 'e' * 64
        write_rows(args.output_dir, rows)
        reject_unchanged(args, fake, match='[Rr]eceipt|[Cc]hecksum')


@pytest.mark.parametrize('kind', ['reordered', 'missing_interior', 'foreign_config', 'foreign_splits'])
def test_records_cannot_be_moved_between_positions_or_run_contexts(tmp_path, kind):
    with started(tmp_path / 'target') as (args, fake):
        rows = records(args.output_dir)
        if kind == 'reordered':
            rows[1], rows[2] = rows[2], rows[1]
        elif kind == 'missing_interior':
            del rows[2]
        else:
            options = {'epochs': args.epochs + 1} if kind == 'foreign_config' else {'random_state': args.random_state + 1}
            with started(tmp_path / 'foreign', **options) as (other_args, _):
                rows = records(other_args.output_dir)
                assert run_hash(other_args.output_dir) != run_hash(args.output_dir)
        write_rows(args.output_dir, rows)
        reject_unchanged(args, fake, match='[Rr]eceipt|[Cc]hecksum')


@pytest.mark.parametrize('kind', ['record', 'fold', 'receipt'])
def test_duplicate_json_keys_are_rejected_even_when_values_agree(tmp_path, kind):
    with started(tmp_path) as (args, fake):
        path = args.output_dir / 'records.jsonl'
        lines = path.read_text(encoding='utf-8').splitlines()
        value = json.loads(lines[0])
        needle = {'record': '"mean_score": ', 'fold': '"accuracy": ', 'receipt': '"sequence": '}[kind]
        number = value['mean_score'] if kind != 'receipt' else 0
        duplicated = needle + json.dumps(number) + ', ' + needle
        assert needle in lines[0]
        lines[0] = lines[0].replace(needle, duplicated, 1)
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        reject_unchanged(args, fake, match='[Dd]uplicate')


@pytest.mark.parametrize('filename,key', [('config.json', 'epochs'), ('splits.json', 'observation_ids')])
@pytest.mark.parametrize('route', ['resume', 'qualified'])
def test_duplicate_run_identity_fields_are_rejected_before_reuse(tmp_path, filename, key, route):
    with started(tmp_path, completed=route == 'qualified') as (args, fake):
        path = args.output_dir / filename
        content = read(path)
        repeated = json.dumps(key) + ': ' + json.dumps(content[key]) + ', '
        raw = json.dumps(content)
        path.write_text('{' + repeated + raw[1:] + '\n', encoding='utf-8')
        if route == 'qualified':
            refresh_outer_hashes(args.output_dir)
        reject_unchanged(args, fake, route, match='[Dd]uplicate')


@pytest.mark.parametrize('tail', ['complete_wrong_checksum', 'complete_missing_receipt', 'malformed_newline'])
def test_complete_or_newline_terminated_bad_tail_is_not_silently_recovered(tmp_path, tail):
    with started(tmp_path) as (args, fake):
        rows = records(args.output_dir)
        if tail == 'complete_wrong_checksum':
            rows[-1]['_receipt']['sha256'] = 'f' * 64
        elif tail == 'complete_missing_receipt':
            del rows[-1]['_receipt']
        path = args.output_dir / 'records.jsonl'
        payload = '\n'.join(json.dumps(row) for row in rows).encode('utf-8')
        if tail == 'malformed_newline':
            payload += b'\n{"params":\n'
        path.write_bytes(payload)
        reject_unchanged(args, fake)
        assert not (args.output_dir / 'records-interrupted.jsonl').exists()


def test_bad_earlier_receipt_prevents_any_partial_tail_repair(tmp_path):
    with started(tmp_path) as (args, fake):
        rows = records(args.output_dir)
        rows[0]['folds'][0]['kernel_seconds'] += 1.
        write_rows(args.output_dir, rows)
        path = args.output_dir / 'records.jsonl'
        path.write_bytes(path.read_bytes() + b'{"params":')
        reject_unchanged(args, fake, match='[Rr]eceipt|[Cc]hecksum')
        assert not (args.output_dir / 'records-interrupted.jsonl').exists()


def test_complete_valid_receipt_without_newline_is_kept_and_separated_before_append(tmp_path):
    with started(tmp_path) as (args, fake):
        path = args.output_dir / 'records.jsonl'
        saved = records(args.output_dir)
        path.write_bytes(path.read_bytes().rstrip(b'\r\n'))
        args.resume = True
        assert _RUNNER.run_search(args)['state'] == 'accepted'
        assert records(args.output_dir)[:len(saved)] == saved
        assert not (args.output_dir / 'records-interrupted.jsonl').exists()


def test_actual_partial_append_preserves_backup_and_retries_only_uncommitted_candidate(tmp_path):
    args, fake = arguments(tmp_path), DetailedScores()
    real_open = Path.open
    writes = []
    failure = KeyboardInterrupt()

    class InterruptedAppend:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *details):
            return self.stream.__exit__(*details)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, value):
            writes.append(value)
            if len(writes) == 2:
                self.stream.write(value[:len(value) // 2])
                self.stream.flush()
                raise failure
            return self.stream.write(value)

    def interrupted_open(path, mode='r', *positional, **keywords):
        stream = real_open(path, mode, *positional, **keywords)
        if path.name == 'records.jsonl' and mode == 'a':
            return InterruptedAppend(stream)
        return stream

    with synthetic(fake):
        with patch.object(Path, 'open', interrupted_open), pytest.raises(KeyboardInterrupt) as caught:
            _RUNNER.run_search(args)
        assert caught.value is failure
        raw = (args.output_dir / 'records.jsonl').read_bytes()
        committed = json.loads(raw.splitlines()[0])
        first_key = _RUNNER.candidate_key(committed['params'])
        assert len(raw.splitlines()) == 2
        args.resume = True
        assert _RUNNER.run_search(args)['state'] == 'accepted'
        assert (args.output_dir / 'records-interrupted.jsonl').read_bytes() == raw
        assert records(args.output_dir)[0] == committed
        assert sum(_RUNNER.candidate_key(item) == first_key for item in fake.search_calls) == 1


@pytest.mark.parametrize('route', ['resume', 'qualified'])
def test_previous_schema_requires_fresh_output_without_modifying_evidence(tmp_path, route):
    with started(tmp_path, completed=route == 'qualified') as (args, fake):
        path = args.output_dir / 'config.json'
        config = read(path)
        assert config['schema_version'] == 7
        config['schema_version'] = 6
        path.write_text(json.dumps(config), encoding='utf-8')
        if route == 'qualified':
            refresh_outer_hashes(args.output_dir)
        reject_unchanged(args, fake, route, match='fresh output directory')


def test_valid_checksum_does_not_excuse_semantically_inconsistent_fold_mean(tmp_path):
    with started(tmp_path) as (args, fake):
        rows = records(args.output_dir)
        rows[0]['mean_score'] = .99
        previous = root = run_hash(args.output_dir)
        for index, row in enumerate(rows):
            row.pop('_receipt')
            rows[index] = _RUNNER._seal_record(row, run_sha256=root, sequence=index,
                                              previous_sha256=previous)
            previous = rows[index]['_receipt']['sha256']
        write_rows(args.output_dir, rows)
        reject_unchanged(args, fake, match='mean score disagrees')
