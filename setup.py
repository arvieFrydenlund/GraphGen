from setuptools import setup, Extension
import os
import platform
import shutil
import subprocess
import pybind11


def _boost_include_dir() -> str:
    """Locate Boost headers so setup.py's build_ext sees the same path as install.sh."""
    override = os.environ.get("BOOST_INCLUDE_DIR")
    if override:
        return override
    if platform.system() == "Darwin" and shutil.which("brew"):
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "boost"], text=True
            ).strip()
            candidate = os.path.join(prefix, "include")
            if os.path.isdir(os.path.join(candidate, "boost")):
                return candidate
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return "/usr/include"


extra_compile_args = ["-std=c++20", "-O3", "-DNDEBUG"]
extra_link_args = []
if platform.system() == "Darwin":
    # pybind11 modules on macOS resolve Python symbols at import time.
    extra_link_args.append("-undefined")
    extra_link_args.append("dynamic_lookup")

ext_modules = [
    Extension(
        "generator",  # Module name
        ["generator.cpp"],  # Source files
        include_dirs=[pybind11.get_include(), _boost_include_dir()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="generator",
    version="0.1",
    ext_modules=ext_modules,
    # Only the C++ extension is packaged; no Python packages or top-level
    # modules. Setting these explicitly stops setuptools from auto-discovering
    # `old_code/`, `cool_graphs/`, etc. as flat-layout packages.
    packages=[],
    py_modules=[],
)