"""Setup."""

import os
from setuptools import setup

pkgname = "yapympi"
pkgdesc = "Yet Another Python MPI Library"

with open("README.rst", "r") as fh:
    long_description = fh.read()

classifiers = """
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: POSIX :: Linux
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Scientific/Engineering
Topic :: System :: Distributed Computing
""".strip().split("\n")

os.environ.setdefault("CC", "mpicc")

setup(
    name=pkgname,
    pkgdesc=pkgdesc,

    author="Parantapa Bhattacharya",
    author_email="pb+pypi@parantapa.net",

    long_description=long_description,
    long_description_content_type="text/x-rst",

    packages=[pkgname],
    package_dir={'': 'src'},

    use_scm_version=True,

    setup_requires=["setuptools_scm", "cffi>=1.0.0"],
    cffi_modules=["src/%s/cmpi_build.py:ffibuilder" % pkgname],
    install_requires=["cffi>=1.0.0"],

    url="http://github.com/parantapa/%s" % pkgname,
    classifiers=classifiers
)
