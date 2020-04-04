"""A CFFI module for accessing MPI functions."""

import os
from cffi import FFI

ffibuilder = FFI()

MPIDIST = os.environ["MPIDIST"].upper()

if MPIDIST == "MPICH":
    ffibuilder.cdef("""
        typedef int... MPI_Comm;
        typedef int... MPI_Datatype;
        typedef int... MPI_Request;
        typedef int... MPI_Errhandler;
    """)
elif MPIDIST == "OPENMPI":
    ffibuilder.cdef("""
        typedef ... *MPI_Comm;
        typedef ... *MPI_Datatype;
        typedef ... *MPI_Request;
        typedef ... *MPI_Errhandler;
    """)
else:
    raise ValueError("Unknown version: %s" % MPIDIST)

ffibuilder.cdef("""
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

    int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status);
    int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
    int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);

    int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status);
    int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
    int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]);

    int MPI_Cancel(MPI_Request * request);
""")

ffibuilder.set_source(
    "yapympi.cmpi",
    "#include <mpi.h>",
    libraries=["mpi"]
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
