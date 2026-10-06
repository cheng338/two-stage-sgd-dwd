"""Check an installed wheel from an isolated virtual environment."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pkgutil
import sys
import sysconfig
import xml.etree.ElementTree as ET


PACKAGE = "two_stage_sgd_dwd"
DISTRIBUTION = "two-stage-sgd-dwd"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check(flavor, junit=None):
    require(sys.flags.isolated == 1, "Run this check with Python -I.")
    require(sysconfig.get_platform() == "win-amd64", "This check targets Windows AMD64 wheels.")
    prefix = Path(sys.prefix).resolve()
    require(prefix != Path(sys.base_prefix).resolve(), "An isolated venv is required.")
    roots = {Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")}
    require(all(path.is_relative_to(prefix) for path in roots),
            "The venv's site-packages must be inside its prefix.")

    def installed(path):
        resolved = Path(path).resolve()
        require(any(resolved.is_relative_to(root) for root in roots),
                f"Import or metadata outside venv site-packages: {resolved}")
        return resolved

    distribution = importlib.metadata.distribution(DISTRIBUTION)
    files = list(distribution.files or ())
    require(files, "Installed distribution has no file inventory.")
    metadata_files = [entry for entry in files if entry.name == "METADATA"
                      and entry.parent.name.endswith(".dist-info")]
    require(len(metadata_files) == 1, "Expected one installed distribution METADATA file.")
    metadata_path = installed(distribution.locate_file(metadata_files[0]))
    wheel_path = installed(distribution.locate_file(metadata_files[0].parent / "WHEEL"))
    tags = [line[5:] for line in wheel_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("Tag: ")]
    expected_tag = ("py3-none-any" if flavor == "portable" else
                    "cp310-cp310-win_amd64" if sys.version_info[:2] == (3, 10) else
                    "cp311-abi3-win_amd64")
    require(tags == [expected_tag], f"Unexpected installed wheel tags: {tags}")
    package = importlib.import_module(PACKAGE)
    package_path = installed(package.__file__).parent
    require(package.__version__ == distribution.version,
            "Package and installed distribution versions differ.")
    require(all(installed(path) == package_path for path in package.__path__),
            "Unexpected package search path.")
    package_files = [entry for entry in files if entry.parts[0] == PACKAGE]
    require(package_files, "Installed distribution contains no package files.")
    record_hashes = {}
    for entry in package_files:
        path = installed(distribution.locate_file(entry))
        require(path.is_relative_to(package_path),
                f"Distribution package file outside imported package: {entry}")
        if entry.hash is not None:
            digest = hashlib.new(entry.hash.mode, path.read_bytes()).digest()
            actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
            require(actual == entry.hash.value, f"Installed file differs from RECORD: {entry}")
            record_hashes[str(entry)] = {"algorithm": entry.hash.mode, "value": actual}

    native_name = PACKAGE + "._sgd_accel"
    native_spec = importlib.util.find_spec(native_name)
    native_files = [entry for entry in package_files
                    if entry.name.startswith("_sgd_accel")
                    and entry.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))]
    if flavor == "native":
        require(native_spec is not None and len(native_files) == 1,
                "Native wheel must contain one importable compiled accelerator.")
        require(isinstance(native_spec.loader, importlib.machinery.ExtensionFileLoader),
                "Native accelerator must use an extension loader.")
        require(installed(native_spec.origin) == installed(distribution.locate_file(native_files[0])),
                "Native import differs from the distribution's compiled file.")
    else:
        require(native_spec is None and not native_files,
                "Portable wheel must not contain or import the native accelerator.")

    # Load every package submodule, not just the top-level import.
    for item in pkgutil.walk_packages(package.__path__, PACKAGE + "."):
        importlib.import_module(item.name)

    def module_paths():
        result = {}
        for name, module in sorted(sys.modules.items()):
            if name != PACKAGE and not name.startswith(PACKAGE + "."):
                continue
            path = installed(module.__file__)
            require(path.is_relative_to(package_path), f"Mixed package origins: {name}")
            result[name] = {"path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return result

    module_paths()
    import numpy as np
    from threadpoolctl import threadpool_info

    X = np.array([[-1.5, -.4], [-1., -.8], [-.6, -.2], [-.2, -.4],
                  [.2, .4], [.6, .2], [1., .8], [1.5, .4]], dtype=np.float64)
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    query = np.vstack((X, [[0., 0.], [.3, -.2]]))
    definitions = (
        (package.LinearSGDDWD, {}),
        (package.KernelSGDDWD, {"lambd": .1, "gamma": .5}),
        (package.TwoStageDWDClassifier,
         {"lambd": .1, "gamma": .5, "retention_fraction": 1.}),
    )
    fitted = []
    for estimator, extra in definitions:
        def create(backend):
            return estimator(max_epochs=2, batch_size=1, stopping="fixed",
                             random_state=17, optimizer_backend=backend, **extra)

        model = create("native" if flavor == "native" else "auto").fit(X, y)
        stages = ([model.linear_estimator_, model.kernel_estimator_]
                  if isinstance(model, package.TwoStageDWDClassifier) else [model])
        expected = "native" if flavor == "native" else "python"
        reason = "native_supported" if flavor == "native" else "extension_unavailable"
        require(model.optimizer_backend_ == expected, "Unexpected fitted backend.")
        for stage in stages:
            require(stage.optimizer_backend_ == expected
                    and stage.optimizer_backend_reason_ == reason,
                    "A stage did not use the required backend or fallback.")
            require(stage.n_updates_ == 2 * len(y), "Tiny fit did not complete its update budget.")
        scores, labels = model.decision_function(query), model.predict(query)
        require(np.isfinite(scores).all() and scores.shape == (len(query),),
                "Tiny fitted model produced invalid scores.")
        require(labels.shape == scores.shape and np.isin(labels, y).all(),
                "Tiny fitted model produced invalid labels.")
        if flavor == "portable":
            try:
                create("native").fit(X, y)
            except ImportError as error:
                require("optional compiled" in str(error), "Unexpected native-import failure.")
            else:
                raise RuntimeError("Portable installation accepted forced native fitting.")
        fitted.append({"estimator": estimator.__name__, "backend": expected,
                       "stage_reasons": [stage.optimizer_backend_reason_ for stage in stages],
                       "stage_updates": [int(stage.n_updates_) for stage in stages],
                       "scores": scores.tolist(), "labels": labels.tolist()})

    suite = None
    if junit is not None:
        cases = list(ET.parse(junit).getroot().iter("testcase"))
        require(cases, "JUnit report has no test cases.")
        failed = [case for case in cases if case.find("failure") is not None
                  or case.find("error") is not None]
        skipped = [case.find("skipped") for case in cases if case.find("skipped") is not None]
        # JUnit text also contains source paths; classify the skip message only.
        reasons = [item.get("message", "") for item in skipped]
        require(not failed, "The installed-wheel suite contains failures or errors.")
        require(len(skipped) < len(cases), "The installed-wheel suite skipped every test case.")
        if flavor == "native":
            require(not any(("native" in reason.lower() or "compiled" in reason.lower())
                            and ("unavailable" in reason.lower() or "legacy" in reason.lower())
                            for reason in reasons),
                    "Native tests were skipped because compiled support was unavailable.")
        suite = {"path": str(junit.resolve()),
                 "sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
                 "test_cases": len(cases), "skipped": len(skipped), "skip_reasons": reasons}

    return {"state": "passed", "flavor": flavor, "python": sys.version,
            "executable": sys.executable, "prefix": str(prefix),
            "site_packages": sorted(map(str, roots)), "metadata": str(metadata_path),
            "version": distribution.version, "wheel_tags": tags,
            "verified_record_hashes": record_hashes,
            "modules": module_paths(), "fits": fitted, "suite": suite,
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl",
                              "pytest", "setuptools", "wheel")},
            "threadpools": threadpool_info()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", choices=("native", "portable"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    report = {"state": "failed", "flavor": args.flavor}
    try:
        report = check(args.flavor, args.junit)
    except Exception as error:
        report.update(error_type=type(error).__name__, message=str(error))
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
