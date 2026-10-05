"""Portable identity of the loaded numerical runtime used by the CV example.

This is a reproducibility guard for an immutable environment, not an attestation
of already loaded machine code. File hashes are reused while stat stamps agree.
"""
from __future__ import annotations

import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
# Load the optional optimizer's numerical dependency before thread limits are
# established, without importing or requiring the optimizer extension itself.
import scipy.linalg.cython_blas
import sklearn
from threadpoolctl import threadpool_info


# Some BLAS libraries load OpenMP when their thread count is first queried.
# Do this before callers establish limits, so the context also sees that pool.
threadpool_info()


_HASH_CACHE = {}
_PACKAGES = {'numpy': np, 'scipy': scipy, 'sklearn': sklearn}


def _stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _file_sha256(path):
    """Stream once per file version; reject a file changed while being read."""
    path = Path(path).absolute()
    stamp = _stamp(path)
    cached = _HASH_CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if _stamp(path) != stamp:
        raise RuntimeError('Numerical runtime file changed while being identified: ' + path.name)
    result = digest.hexdigest()
    _HASH_CACHE[path] = (stamp, result)
    return result


def _loaded_binaries():
    """Find loaded scientific extensions, including later optional imports."""
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.split('.', 1)[0] not in _PACKAGES:
            continue
        origin = getattr(module, '__file__', None)
        if origin is not None and any(str(origin).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
            result[name] = Path(origin)
    return result


def runtime_identity():
    """Record binary bytes, library implementations and current thread counts.

    Only portable filenames and module names are persisted. Equal environments
    can live at different paths. Loaded BLAS/OpenMP libraries are enumerated on
    every check; no assumption ties a DLL load to Python's module count.
    """
    packages = {}
    for name, module in _PACKAGES.items():
        path = Path(module.__file__)
        packages[name] = {'version': str(module.__version__),
                          'filename': path.name, 'sha256': _file_sha256(path)}
    binaries = [{'module': name, 'filename': path.name, 'sha256': _file_sha256(path)}
                for name, path in sorted(_loaded_binaries().items())]
    pools = []
    for pool in threadpool_info():
        path = Path(pool['filepath'])
        record = {key: value for key, value in pool.items() if key != 'filepath'}
        record.update(filename=path.name, sha256=_file_sha256(path))
        pools.append(record)
    if not any(pool.get('user_api') == 'blas' for pool in pools):
        raise RuntimeError('The numerical runtime has no identifiable loaded BLAS library')
    pools.sort(key=lambda record: json.dumps(record, sort_keys=True, allow_nan=False))
    return {'schema_version': 1,
            'python': platform.python_version(),
            'implementation': platform.python_implementation(),
            'system': platform.system(), 'machine': platform.machine(),
            'packages': packages, 'binaries': binaries, 'threadpools': pools}
