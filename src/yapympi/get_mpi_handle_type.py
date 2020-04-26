"""Get the type to use for MPI handle types."""

import os
from subprocess import run
from tempfile import NamedTemporaryFile


def get_mpi_handle_type(dir=None):
    """Get the type to use for MPI handle types.

    This method writes out a simple C code to determine the type
    needed to represent MPI_COMM_WORLD for cffi.

    Parameters
    ----------
    dir : str
        The directory in which the temporary C file
        and the executeable file is to be created.

    Returns
    -------
    type : str
        "int" or "pointer"
    """
    c_compiler = os.environ.get("CC", "mpicc")

    c_code = """
#include <stdio.h>
#include <stdint.h>
#include <mpi.h>

#define typename(x) _Generic((x), \
	int8_t: "int", int16_t: "int", int32_t: "int", int64_t: "int", \
	uint8_t: "int", uint16_t: "int", uint32_t: "int", uint64_t: "int", \
	default: "pointer" \
)

int main(int argc, char *argv[])
{
	puts(typename(MPI_COMM_WORLD));
	return 0;
}
""".strip()

    c_file = NamedTemporaryFile(
        mode="wt", encoding="utf-8", suffix=".c", delete=False, dir=dir
    )
    try:
        c_file.write(c_code)
        c_file.close()

        e_file = NamedTemporaryFile(delete=False, dir=dir)
        try:
            e_file.close()

            cmd = [c_compiler, "-o", e_file.name, c_file.name]
            run(cmd, shell=False, check=True)

            os.chmod(e_file.name, 0o700)

            cmd = [e_file.name]
            cp = run(cmd, shell=False, check=True, capture_output=True)

            return cp.stdout.decode("utf-8").strip()
        finally:
            os.remove(e_file.name)
    finally:
        os.remove(c_file.name)


if __name__ == "__main__":
    print(get_mpi_handle_type())
