"""Unknown diagnostic objects must not run code merely to test their type."""
import json

import pytest

from test_tune_mnist import _RUNNER, Scores, arguments, records, synthetic


def unknown_note(events, action):
    class UnknownNote:
        @property
        def __class__(self):
            events.append('__class__')
            if action == 'pretend-str':
                return str
            if action == 'pretend-object':
                return object
            if action == 'ordinary-error':
                raise ValueError('unexpected note type hook')
            if action == 'interrupt':
                raise KeyboardInterrupt('unexpected note type hook')
            raise SystemExit(17)

        def __str__(self):
            events.append('__str__')
            raise AssertionError('unknown note conversion')

        def __repr__(self):
            events.append('__repr__')
            raise AssertionError('unknown note representation')

    return UnknownNote()


@pytest.mark.parametrize('action', ['pretend-str', 'pretend-object', 'ordinary-error', 'interrupt', 'exit'])
@pytest.mark.parametrize('container', [list, tuple])
def test_unknown_note_is_skipped_without_type_or_conversion_hooks(action, container):
    events = []
    note = unknown_note(events, action)
    failure = RuntimeError('original')
    notes = container(['before', note, 'after'])
    cause = ValueError('existing cause')
    failure.__notes__, failure.__cause__, failure.status = notes, cause, 'failed_fit'
    facts = _RUNNER._exception_context(failure)
    assert events == []
    assert facts == {'exception_notes': ['before', 'after'], 'error_status': 'failed_fit',
                     'exception_causes': [{'error_type': 'ValueError', 'message': 'existing cause'}]}
    assert failure.__notes__ is notes and failure.__cause__ is cause
    assert failure.args == ('original',) and failure.status == 'failed_fit'
    assert failure.__dict__ == {'__notes__': notes, 'status': 'failed_fit'}
    json.dumps(facts, allow_nan=False)


@pytest.mark.parametrize('text', ['', 'plain content', 'x' * 1025])
def test_actual_string_subclass_uses_base_string_operations(text):
    events = []

    class HostileMeta(type):
        def __instancecheck__(cls, other):
            events.append('meta_instancecheck')
            raise AssertionError('unexpected type protocol')

        def __subclasscheck__(cls, other):
            events.append('meta_subclasscheck')
            raise AssertionError('unexpected type protocol')

    class ActualString(str, metaclass=HostileMeta):
        @property
        def __class__(self):
            events.append('__class__')
            raise KeyboardInterrupt('unexpected string type hook')

        def __str__(self):
            events.append('__str__')
            raise SystemExit(9)

        def __len__(self):
            events.append('__len__')
            raise ValueError('unexpected string length hook')

        def __getitem__(self, key):
            events.append('__getitem__')
            raise ValueError('unexpected string slicing hook')

    failure = RuntimeError('original')
    failure.__notes__ = [ActualString(text)]
    facts = _RUNNER._exception_context(failure)
    assert events == []
    assert facts['exception_notes'] == [text[:1024]]
    assert type(facts['exception_notes'][0]) is str
    assert facts.get('exception_notes_truncated', False) == (len(text) > 1024)


@pytest.mark.parametrize('base', [list, tuple])
def test_unknown_container_subclasses_are_skipped_without_operations(base):
    events = []

    class UnknownContainer(base):
        @property
        def __class__(self):
            events.append('__class__')
            return list

        def __len__(self):
            events.append('__len__')
            raise KeyboardInterrupt('unexpected container length hook')

        def __iter__(self):
            events.append('__iter__')
            raise SystemExit(1)

        def __getitem__(self, key):
            events.append('__getitem__')
            raise ValueError('unexpected container slicing hook')

    failure = RuntimeError('original')
    notes = UnknownContainer(['uninspected'])
    failure.__notes__ = notes
    assert _RUNNER._exception_context(failure) == {}
    assert failure.__notes__ is notes and events == []


@pytest.mark.parametrize('action', ['pretend-str', 'ordinary-error', 'interrupt', 'exit'])
def test_skipped_note_cannot_replace_failure_or_prevent_durable_context_and_resume(tmp_path, action):
    events = []
    failure = RuntimeError('original fit failure')
    failure.__notes__ = ['known witness', unknown_note(events, action), 'second witness']
    failure.status = 'numerical_failure'
    cause = ValueError('original cause')
    failure.__cause__ = cause
    args = arguments(tmp_path)
    fake = Scores(fail_once=(1., 1.), fail_error=failure)
    with synthetic(fake):
        with pytest.raises(RuntimeError) as caught:
            _RUNNER.run_search(args)
        assert caught.value is failure and events == []
        evidence = json.loads((args.output_dir / 'failure.json').read_text())
        assert evidence['exception_notes'] == ['known witness', 'second witness']
        assert evidence['error_status'] == failure.status
        assert evidence['exception_causes'] == [{'error_type': 'ValueError', 'message': 'original cause'}]
        before = records(args.output_dir)
        args.resume = True
        assert _RUNNER.run_search(args)['state'] == 'accepted'
        assert records(args.output_dir)[:len(before)] == before
    assert failure.__cause__ is cause and events == []
