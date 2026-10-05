"""Build the optional strict-arithmetic accelerator or the portable package."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class StrictBuildExt(build_ext):
    def build_extensions(self):
        zig = os.environ.get("SGD_BUILD_ZIG")
        flags = ["-O3", "-g0", "-std=c11", "-ffp-contract=off", "-fno-fast-math",
                 "-fno-associative-math", "-fno-unsafe-math-optimizations",
                 "-fno-finite-math-only"]
        if zig:
            # Explicit build-time option; importing/installing a wheel never
            # downloads a compiler or invokes a build tool.
            if sys.platform != "win32":
                raise RuntimeError("SGD_BUILD_ZIG currently supports Windows builds.")
            if sysconfig.get_platform().lower() != "win-amd64":
                raise RuntimeError("SGD_BUILD_ZIG requires a Windows AMD64 Python build.")
            for extension in self.extensions:
                target = Path(self.get_ext_fullpath(extension.name)).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                # A win_amd64 wheel must also run on baseline x86-64 hardware.
                library = ("python3.lib" if extension.py_limited_api else
                           f"python{sys.version_info.major}{sys.version_info.minor}.lib")
                command = [zig, "cc", *flags, "-march=x86_64", "-s", "-shared",
                           *[f"-D{name}={value}" for name, value in extension.define_macros],
                           "-I" + sysconfig.get_path("include"),
                           str(Path(sys.base_prefix) / "libs" / library),
                           *extension.sources, "-o", str(target)]
                subprocess.run(command, check=True)
                for stem in (target.stem, "sgd_accel"):
                    for suffix in (".pdb", ".lib", ".exp"):
                        product = target.parent / (stem + suffix)
                        if product.is_file():
                            product.unlink()
            return
        if self.compiler.compiler_type == "msvc":
            flags = ["/O2", "/std:c11", "/fp:strict"]
        for extension in self.extensions:
            extension.extra_compile_args = flags
        super().build_extensions()


build_accel = os.environ.get("SGD_BUILD_ACCEL", "1")
if build_accel not in ("0", "1"):
    raise RuntimeError("SGD_BUILD_ACCEL must be 0 (portable) or 1 (optional native).")
portable = build_accel == "0"
# Py_buffer is part of the stable ABI starting with Python 3.11. Python 3.10
# uses its own full-API extension instead of claiming an incompatible abi3 tag.
limited_api = sys.version_info >= (3, 11)
extensions = [] if portable else [Extension(
    "two_stage_sgd_dwd._sgd_accel",
    sources=["src/two_stage_sgd_dwd/sgd_accel.c"],
    define_macros=[("Py_LIMITED_API", "0x030B0000")] if limited_api else [],
    py_limited_api=limited_api,
    libraries=[] if sys.platform == "win32" else ["m"],
    optional=True,
)]
setup(ext_modules=extensions, cmdclass={"build_ext": StrictBuildExt},
      options={"bdist_wheel": {"py_limited_api": "cp311"}} if limited_api and not portable else {})
