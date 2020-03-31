"""A Simple MPI interface."""

from .mpi import ffi, lib


class MPIError(RuntimeError):
    """MPI Runtime error.

    Attributes
    ----------
    errorcode : int
        A MPI error code
    """

    def __init__(self, errorcode):
        super().__init__(errorcode)
        self.errorcode = errorcode

    def __repr__(self):
        return "MPIError(%d)" % self.errorcode

    def __str__(self):
        error_str = error_string(self.errorcode)
        return "MPI Error: %d: %s" % (self.errorcode, error_str)


class MPIStatus:
    """MPI Status object.

    Attributes
    ----------
    source : int
        Rank of source
    tag : int
        Message tag
    error : int
        MPI error code
    count : int
        Number received elements
    """

    def __init__(self, status):
        self.status = status

    def __repr__(self):
        fmt = "MPIStatus(source=%d, tag=%d, error=%d, count=%d)"
        return fmt % (self.source, self.tag, self.error, self.count)

    @property
    def source(self):
        """Return the source."""
        return self.status.MPI_SOURCE

    @property
    def tag(self):
        """Return the tag."""
        return self.status.MPI_TAG

    @property
    def error(self):
        """Return the error."""
        return self.status.MPI_ERROR

    @property
    def count(self):
        """Return the count."""
        cnt = ffi.new("int *")
        datatype = lib.MPI_BYTE
        ret = lib.MPI_Get_count(self.status, datatype, cnt)
        check_error(ret)
        return cnt[0]

    def check_error(self):
        """Raise MPIError if error is present."""
        if self.error != lib.MPI_SUCCESS:
            raise MPIError(self.error)


def error_string(errorcode):
    """Return a string for a given error code.

    Parameters
    ----------
    errorcode : int
        A MPI error code

    Returns
    -------
    error_str : str
        Text that corresponds to the errorcode
    """
    string = ffi.new("char []", lib.MPI_MAX_ERROR_STRING)
    resultlen = ffi.new("int *")
    ret = lib.MPI_Error_string(errorcode, string, resultlen)
    if ret != lib.MPI_SUCCESS:
        return "Unknown MPI Error: %d" % errorcode
    error_str = ffi.string(string, resultlen[0])
    error_str = error_str.decode("ascii")
    return error_str


def check_error(retcode):
    """If retcode is not MPI_SUCCESS raise an error.

    Parameters
    ----------
    retcode : int
        Return code of an MPI function

    Raises
    ------
    MPIError
        If retcode is not MPI_SUCCESS
    """
    if retcode != lib.MPI_SUCCESS:
        raise MPIError(retcode)


def init():
    """Initialize the MPI execution environment."""
    ret = lib.MPI_Init(ffi.NULL, ffi.NULL)
    check_error(ret)


def finalize():
    """Terminate the MPI execution environment."""
    ret = lib.MPI_Finalize()
    check_error(ret)


def abort(comm=lib.MPI_COMM_WORLD, errorcode=1):
    """Immediately terminate the MPI execution environment.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator of tasks to abort
    errorcode : int
        Error code to return to invoking environment
    """
    ret = lib.MPI_Abort(comm, errorcode)
    check_error(ret)


def comm_set_errhandler(comm=lib.MPI_COMM_WORLD, errhandler=lib.MPI_ERRORS_RETURN):
    """Set the error handler for a communicator.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator (handle)
    errhandler : MPI_Errhandler
        New error handler for communicator (handle)
    """
    ret = lib.MPI_Comm_set_errhandler(comm, errhandler)
    check_error(ret)


def comm_rank(comm=lib.MPI_COMM_WORLD):
    """Determine the rank of the calling process in the communicator.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator (handle)

    Returns
    -------
    rank : int
        Rank of the calling process in the group of comm
    """
    rank = ffi.new("int *")
    ret = lib.MPI_Comm_rank(comm, rank)
    check_error(ret)
    return rank[0]


def comm_size(comm=lib.MPI_COMM_WORLD):
    """Determine the size of the group associated with a communicator.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator (handle)

    Returns
    -------
    size : int
        Number of processes in the group of comm
    """
    size = ffi.new("int *")
    ret = lib.MPI_Comm_size(comm, size)
    check_error(ret)
    return size[0]


def get_processor_name():
    """Get the name of the processor.

    Returns
    -------
    proc_name : str
        Name of the processor
    """
    name = ffi.new("char []", lib.MPI_MAX_PROCESSOR_NAME)
    resultlen = ffi.new("int *")
    ret = lib.MPI_Get_processor_name(name, resultlen)
    check_error(ret)
    proc_name = ffi.string(name, resultlen[0])
    proc_name = proc_name.decode("ascii")
    return proc_name


def send(buf, dest, tag, comm=lib.MPI_COMM_WORLD):
    """Perform a blocking send.

    Parameters
    ----------
    buf : str or bytes or any object supporting buffer interface
        The send buffer
    dest : int
        Rank of destination
    tag : int
        Message tag
    comm : MPI_Comm
        Communicator (handle)
    """
    if isinstance(buf, (str, bytes)):
        cbuf = ffi.new("char []", buf)
    else:
        cbuf = ffi.from_buffer("char []", buf)
    count = len(cbuf)
    datatype = lib.MPI_BYTE

    ret = lib.MPI_Send(cbuf, count, datatype, dest, tag, comm)
    check_error(ret)


def recv(buf, source=lib.MPI_ANY_SOURCE, tag=lib.MPI_ANY_TAG, comm=lib.MPI_COMM_WORLD):
    """Perform a blocking receive for a message.

    Parameters
    ----------
    buf : a writable object supporting buffer interface
        The receive buffer
    source : int
        Rank of source
    tag : int
        Message tag
    comm : MPI_Comm
        Communicator (handle)

    Returns
    -------
    status : MPIStatus
        Status object
    """
    cbuf = ffi.from_buffer(buf, require_writable=True)
    count = len(cbuf)
    datatype = lib.MPI_BYTE
    status = ffi.new("MPI_Status *")
    ret = lib.MPI_Recv(cbuf, count, datatype, source, tag, comm, status)
    check_error(ret)
    return MPIStatus(status)


def barrier(comm=lib.MPI_COMM_WORLD):
    """Blocks until all processes in the communicator have reached this routine.

    comm : MPI_Comm
        Communicator (handle)
    """
    ret = lib.MPI_Barrier(comm)
    check_error(ret)
