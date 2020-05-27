"""A CFFI module for accessing MPI functions."""

import os
from subprocess import run
from tempfile import NamedTemporaryFile

from cffi import FFI

FFIBUILDER = FFI()

COMM_WORLD_TYPE_C_CODE = """
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


def get_mpi_handle_type(tmpdir=None):
    """Get the type to use for MPI handle types.

    This method writes out a simple C code to determine the type
    needed to represent MPI_COMM_WORLD for cffi.

    Parameters
    ----------
    tmpdir : str
        The directory in which the temporary C file
        and the executeable file is to be created.

    Returns
    -------
    type : str
        "int" or "pointer"
    """
    c_compiler = os.environ.get("CC", "mpicc")

    c_file = NamedTemporaryFile(
        mode="wt", encoding="utf-8", suffix=".c", delete=False, dir=tmpdir
    )
    try:
        c_file.write(COMM_WORLD_TYPE_C_CODE)
        c_file.close()

        e_file = NamedTemporaryFile(delete=False, dir=tmpdir)
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


# Do not execute get_mpi_handle_type(), which has "side effects"
# Unless this script has been called as main.
if __name__ == "__main__":
    MPI_HANDLE_TYPE = get_mpi_handle_type()
else:
    # We use default MPICH's handle type as default
    MPI_HANDLE_TYPE = "int"

if MPI_HANDLE_TYPE == "int":
    FFIBUILDER.cdef(
        """
        typedef int... MPI_Comm;
        typedef int... MPI_Datatype;
        typedef int... MPI_Request;
        typedef int... MPI_Errhandler;
    """
    )
else:  # MPI_HANDLE_TYPE == "pointer":
    FFIBUILDER.cdef(
        """
        typedef ... *MPI_Comm;
        typedef ... *MPI_Datatype;
        typedef ... *MPI_Request;
        typedef ... *MPI_Errhandler;
    """
    )

FFIBUILDER.cdef(
    """
    typedef struct {
        int MPI_SOURCE;
        int MPI_TAG;
        int MPI_ERROR;
        ...;
    } MPI_Status;

    const MPI_Comm MPI_COMM_WORLD;
    const MPI_Datatype MPI_BYTE;
    MPI_Status *const MPI_STATUS_IGNORE;
    const MPI_Errhandler MPI_ERRORS_RETURN;
    const MPI_Errhandler MPI_ERRORS_ARE_FATAL;

    const int MPI_ANY_SOURCE;
    const int MPI_ANY_TAG;
    const int MPI_MAX_PROCESSOR_NAME;
    const int MPI_MAX_ERROR_STRING;
    const int MPI_SUCCESS;
    const int MPI_ERR_IN_STATUS;

    int MPI_Error_string(int errorcode, char *string, int *resultlen);
    int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler);

    int MPI_Init(int *argc, char ***argv);
    int MPI_Finalize(void);
    int MPI_Abort(MPI_Comm comm, int errorcode);

    int MPI_Comm_rank(MPI_Comm comm, int *rank);
    int MPI_Comm_size(MPI_Comm comm, int *size);
    int MPI_Get_processor_name(char *name, int *resultlen);

    int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);

    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
    int MPI_Barrier(MPI_Comm comm);

    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
    int MPI_Wait(MPI_Request *request, MPI_Status *status);
    int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
    int MPI_Cancel(MPI_Request * request);

    int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status);
    int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
    int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);

    int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status);
    int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
    int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]);
"""
)

FFIBUILDER.set_source("yapympi.cmpi", "#include <mpi.h>", libraries=["mpi"])

if __name__ == "__main__":
    FFIBUILDER.compile(verbose=True)
