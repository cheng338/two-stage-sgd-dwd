"""Native build contracts distinguish CPython 3.10 from the 3.11 stable ABI."""
from collections import namedtuple
from contextlib import contextmanager
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

setuptools = pytest.importorskip('setuptools')
# Import build commands before simulating another interpreter version.
from setuptools.command.build_ext import build_ext


_SOURCE = Path(__file__).resolve().parents[1]
_Version = namedtuple('Version', 'major minor micro releaselevel serial')


@contextmanager
def configuration(minor, portable=False):
    version = _Version(3, minor, 12, 'final', 0)
    with patch.object(sys, 'version_info', version), \
            patch.dict(os.environ, {'SGD_BUILD_ACCEL': '0' if portable else '1'}), \
            patch.object(setuptools, 'setup') as setup:
        namespace = runpy.run_path(str(_SOURCE / 'setup.py'))
        setup.assert_called_once()
        yield namespace, setup.call_args.kwargs


@pytest.mark.parametrize('minor', [10, 11, 12])
@pytest.mark.parametrize('portable', [False, True])
def test_build_selects_correct_abi_and_keeps_portable_variant(minor, portable):
    with configuration(minor, portable) as (_, settings):
        if portable:
            assert settings['ext_modules'] == []
            assert settings['options'] == {}
            return
        extension, = settings['ext_modules']
        assert extension.name == 'two_stage_sgd_dwd._sgd_accel'
        assert extension.optional
        if minor == 10:
            assert not extension.py_limited_api
            assert extension.define_macros == []
            assert settings['options'] == {}
        else:
            assert extension.py_limited_api
            assert extension.define_macros == [('Py_LIMITED_API', '0x030B0000')]
            assert settings['options'] == {'bdist_wheel': {'py_limited_api': 'cp311'}}


@pytest.mark.parametrize('minor,library,limited', [
    (10, 'python310.lib', False), (11, 'python3.lib', True), (12, 'python3.lib', True),
])
def test_windows_zig_linker_matches_the_selected_abi(tmp_path, minor, library, limited):
    target = tmp_path / '_sgd_accel.pyd'
    with configuration(minor) as (namespace, settings):
        extension, = settings['ext_modules']
        # This contract check inspects the command; actual wheel tests compile it.
        command = namespace['StrictBuildExt'](setuptools.Distribution())
        command.extensions = [extension]
        with patch.dict(os.environ, {'SGD_BUILD_ZIG': 'compiler-under-test'}), \
                patch.object(sys, 'platform', 'win32'), \
                patch.object(namespace['sysconfig'], 'get_platform', return_value='win-amd64'), \
                patch.object(command, 'get_ext_fullpath', return_value=str(target)), \
                patch.object(namespace['subprocess'], 'run') as run:
            command.build_extensions()
        run.assert_called_once()
        arguments, = run.call_args.args
        assert run.call_args.kwargs == {'check': True}
        assert arguments[0:2] == ['compiler-under-test', 'cc']
        assert str(Path(sys.base_prefix) / 'libs' / library) in arguments
        assert ('-DPy_LIMITED_API=0x030B0000' in arguments) == limited
        assert all(flag in arguments for flag in ('-ffp-contract=off', '-fno-fast-math',
                                                  '-fno-associative-math', '-march=x86_64'))
        assert not target.exists()
