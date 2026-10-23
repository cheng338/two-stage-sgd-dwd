"""Bounded existing exception evidence survives durable failure records."""
import json
from unittest.mock import patch

import pytest

from test_tune_mnist import _RUNNER, Scores, arguments, records, synthetic


@pytest.mark.parametrize('failure', [RuntimeError('fit failed'), KeyboardInterrupt(), SystemExit(3)])
def test_notes_status_cause_survive_failure_receipt_and_resume(tmp_path, failure):
    failure.add_note('fold=1; kernel stage; 24 retained rows')
    failure.status = 'numerical_failure'
    failure.__cause__ = ValueError('kernel construction witness')
    args = arguments(tmp_path)
    fake = Scores(fail_once=(1., 1.), fail_error=failure)
    with synthetic(fake):
        with pytest.raises(type(failure)) as caught:
            _RUNNER.run_search(args)
        assert caught.value is failure
        evidence = json.loads((args.output_dir / 'failure.json').read_text())
        assert evidence['exception_notes'] == failure.__notes__
        assert evidence['error_status'] == 'numerical_failure'
        assert evidence['exception_causes'] == [{'error_type': 'ValueError', 'message': 'kernel construction witness'}]
        before = records(args.output_dir)
        args.resume = True
        assert _RUNNER.run_search(args)['state'] == 'accepted'
        assert records(args.output_dir)[:len(before)] == before


def test_bounds_unknown_notes_and_cause_cycles():
    failure = RuntimeError('original')
    failure.__notes__ = [object(), 'x' * 2000, *['more'] * 9]
    failure.status = 's' * 400
    failure.__cause__ = ValueError('v' * 2000)
    failure.__cause__.__cause__ = failure
    facts = _RUNNER._exception_context(failure)
    assert len(facts['exception_notes']) == 7
    assert len(facts['exception_notes'][0]) == 1024
    assert facts['exception_notes_truncated'] is True
    assert len(facts['error_status']) == 256 and facts['error_status_truncated']
    assert len(facts['exception_causes']) == 1
    assert facts['exception_causes'][0]['message_truncated']
    assert facts['exception_causes_truncated']
    json.dumps(facts, allow_nan=False)


def test_long_cause_chain_and_hostile_message_are_bounded():
    class BadMessage(ValueError):
        def __str__(self):
            raise ValueError('cannot render this cause')
    failure = RuntimeError('original')
    node = failure
    for index in range(7):
        node.__cause__ = BadMessage() if index == 0 else ValueError(str(index))
        node = node.__cause__
    facts = _RUNNER._exception_context(failure)
    assert len(facts['exception_causes']) == 4
    assert facts['exception_causes'][0]['message_unavailable']
    assert facts['exception_causes_truncated']


def test_plain_string_conversion_does_not_call_note_subclass_override():
    class StringSubclass(str):
        def __str__(self):
            raise AssertionError('unexpected custom string conversion')
        def __getitem__(self, key):
            raise AssertionError('unexpected custom slicing')
    failure = RuntimeError('original')
    failure.__notes__ = [StringSubclass('plain content')]
    assert _RUNNER._exception_context(failure)['exception_notes'] == ['plain content']


def test_hostile_optional_attributes_do_not_mask_original_failure():
    class Hostile(RuntimeError):
        def __getattribute__(self, name):
            if name in ('__notes__', 'status'):
                raise ValueError('optional metadata inaccessible')
            return super().__getattribute__(name)
    assert _RUNNER._exception_context(Hostile('original')) == {}


@pytest.mark.parametrize('attribute', ['__notes__', 'status', '__cause__'])
@pytest.mark.parametrize('interrupt', [KeyboardInterrupt(), SystemExit(7)])
def test_new_interrupt_in_diagnostic_attribute_is_not_suppressed(interrupt, attribute):
    class Interrupting(RuntimeError):
        def __getattribute__(self, name):
            if name == attribute:
                raise interrupt
            return super().__getattribute__(name)
    with pytest.raises(type(interrupt)) as caught:
        _RUNNER._exception_context(Interrupting('original'))
    assert caught.value is interrupt


@pytest.mark.parametrize('notes', [object(), iter(['unused']), {'note': 'unused'}, 5, None])
def test_arbitrary_note_containers_are_not_executed(notes):
    failure = RuntimeError('original')
    failure.__notes__ = notes
    assert _RUNNER._exception_context(failure) == {}


def test_nested_cause_attribute_failure_preserves_already_collected_context():
    class InaccessibleCause(ValueError):
        def __getattribute__(self, name):
            if name == '__cause__':
                raise ValueError('optional nested cause inaccessible')
            return super().__getattribute__(name)
    failure = RuntimeError('original')
    failure.__cause__ = InaccessibleCause('first cause')
    facts = _RUNNER._exception_context(failure)
    assert facts['exception_causes'] == [{'error_type': 'InaccessibleCause', 'message': 'first cause'}]


def test_failed_diagnostic_write_still_preserves_original_exception(tmp_path):
    failure = RuntimeError('fit failed')
    failure.add_note('important existing note')
    args = arguments(tmp_path)
    fake = Scores(fail_once=(1., 1.), fail_error=failure)
    original = _RUNNER.write_json
    def deny(path, value):
        if path.name in ('failure.json', 'status.json') and 'error_type' in value:
            raise PermissionError('read-only output')
        return original(path, value)
    with synthetic(fake), patch.object(_RUNNER, 'write_json', side_effect=deny):
        with pytest.raises(RuntimeError) as caught:
            _RUNNER.run_search(args)
        assert caught.value is failure
    assert records(args.output_dir)
