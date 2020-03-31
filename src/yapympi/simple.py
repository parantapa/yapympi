"""A simple MPI interface."""

from ._rawmpi import ffi, lib

class Status:
    """MPI status object wrapper."""

    def __init__(self, status, datatype):
        self.status = status
        self.datatype = datatype

    @property
    def source(self):
        return self.status.MPI_SOURCE

    @property
    def tag(self):
        return self.status.MPI_TAG

    @property
    def error(self):
        return self.status.MPI_ERROR

    @property
    def count(self):
        count = ffi.new("int *")
        lib.MPI_Get_count(self.status, self.datatype, count)
        return count[0]


def init():
    """Initialize the MPI execution environment."""
    lib.MPI_Init(ffi.NULL, ffi.NULL)

def finalize():
    """Terminate the MPI execution environment."""
    lib.MPI_Finalize()

def abort(comm=lib.MPI_COMM_WORLD, errorcode=1):
    """Immediately terminate the MPI execution environment."""
    lib.MPI_Abort(comm, errorcode)

def comm_rank(comm=lib.MPI_COMM_WORLD):
    """Determine the rank of the calling process in the communicator."""
    rank = ffi.new("int *")
    lib.MPI_Comm_rank(comm, rank)
    return rank[0]

def comm_size(comm=lib.MPI_COMM_WORLD):
    """Determine the size of the group associated with a communicator."""
    size = ffi.new("int *")
    lib.MPI_Comm_size(comm, size)
    return size[0]

def get_processor_name():
    """Get the name of the processor."""
    name = ffi.new("char [%d]" % lib.MPI_MAX_PROCESSOR_NAME)
    resultlen = ffi.new("int *")
    lib.MPI_Get_processor_name(name, resultlen)
    ret = ffi.string(name, resultlen[0])
    return ret

def send(buf, dest, tag, comm=lib.MPI_COMM_WORLD):
    """Perform a blocking send."""
    if isinstance(buf, (str, bytes)):
        cbuf = ffi.new("char []", buf)
    else:
        cbuf = ffi.from_buffer("char []", buf)
    count = len(cbuf)
    datatype = lib.MPI_BYTE

    lib.MPI_Send(cbuf, count, datatype, dest, tag, comm)

def recv(buf, source=lib.MPI_ANY_SOURCE, tag=lib.MPI_ANY_TAG, comm=lib.MPI_COMM_WORLD):
    """Perform a blocking receive for a message."""
    cbuf = ffi.from_buffer(buf, require_writable=True)
    count = len(cbuf)
    datatype = lib.MPI_BYTE
    status = ffi.new("MPI_Status *")
    lib.MPI_Recv(cbuf, count, datatype, source, tag, comm, status)
    return Status(status, datatype)
