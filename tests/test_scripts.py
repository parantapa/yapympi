"""Test the MPI scripts."""

from pathlib import Path
from subprocess import run

CURDIR = Path(__file__).parent

def mpirun(script, nranks):
    print("Testing '%s' with %d ranks" % (script, nranks))

    script = str(CURDIR / "scripts" / script)
    cmd = ["mpiexec", "-n", str(nranks), "python", script]
    run(cmd, check=True)

def test_hello1():
    mpirun("hello1.py", 1)
    mpirun("hello1.py", 2)

def test_hello2():
    mpirun("hello1.py", 2)
