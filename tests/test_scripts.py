"""Test the MPI scripts."""

from pathlib import Path
from subprocess import run

CURDIR = Path(__file__).parent

def mpirun(script, n):
    print("Testing '%s' with %d ranks" % (script, n))

    script = str(CURDIR / "scripts" / script)
    cmd = ["mpiexec", "-n", str(n), "python", script]
    run(cmd, check=True)

def test_hello():
    mpirun("hello.py", 1)
    mpirun("hello.py", 2)
