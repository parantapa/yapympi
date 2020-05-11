"""A Simple MPI interface."""

from .cmpi import ffi, lib


class MPIErrorBase(RuntimeError):
    """Base class of MPI Error objects."""


class MPIError(MPIErrorBase):
    """MPI runtime error.

    Attributes
    ----------
    errcode : int
        A MPI error code
    errstr : str
        String representation of the error code
    """

    def __init__(self, errcode):
        self.errcode = errcode
        self.errorstr = error_string(self.errcode)

    def __str__(self):
        return "MPI Error: %d: %s" % (self.errcode, self.errorstr)


class MPIStatusErrors(MPIErrorBase):
    """MPI runtime error in status.

    Attributes
    ----------
    erridxs : list of int
        List of indices where error occured
    errcodes : list of int
        List of MPI error codes
    errstrs : list of str
        List of string representations of the error codes
    """

    def __init__(self, erridxs, errcodes):
        self.erridxs = erridxs
        self.errcodes = errcodes
        self.errstrs = [error_string(c) for c in self.errcodes]

    def __str__(self):
        x = zip(self.erridxs, self.errcodes, self.errstrs)
        x = ["Index: %d; Error: %d: %s" % row for row in x]
        x = "\n".join(x)

        return "MPI Status Errors: %d errors\n%s" % (len(self.errcodes), x)


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
    string = ffi.new("char[]", lib.MPI_MAX_ERROR_STRING)
    resultlen = ffi.new("int*")
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


def check_error_in_status(retcode, statuses):
    """Check possible errors in status.

    Parameters
    ----------
    retcode : int
        Return code of an MPI function
    statuses : MPI_Status[]
        Array of MPI_Status objects.

    Raises
    ------
    MPIStatusErrors
        If there are errors in status
    MPIError
        For any other errors
    """
    if retcode == lib.MPI_SUCCESS:
        return

    if retcode == lib.MPI_ERR_IN_STATUS:
        erridxs, errcodes = [], []
        for idx in range(len(statuses)):
            if statuses[idx].MPI_ERROR != lib.MPI_SUCCESS:
                erridxs.append(idx)
                errcodes.append(statuses[idx].MPI_ERROR)
        raise MPIStatusErrors(erridxs, errcodes)

    raise MPIError(retcode)

def list_to_array(ctype, objs):
    """Create an array of ctype[] form a list of ctype*.

    Parameters
    ----------
    ctype : str
        C type of the data.
    objs : list of ctype*
        A python list of ctype* objects

    Returns
    -------
    ctype[]
        A contiguous array of ctype objects
    """
    arr = ffi.new("%s[]" % ctype, len(objs))
    for i, o in enumerate(objs):
        arr[i] = o[0]
    return arr

def get_count(status, datatype=lib.MPI_BYTE):
    """Get the number of "top level" elements.

    Parameters
    ----------
    status : MPI_Status*
        Return status of receive operation
    datatype : MPI_Datatype
        Datatype of each receive buffer element

    Returns
    -------
    count : int
        Number of received bytes.
    """
    cnt = ffi.new("int*")
    ret = lib.MPI_Get_count(status, datatype, cnt)
    check_error(ret)
    return cnt[0]


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


def comm_set_errhandler(comm=lib.MPI_COMM_WORLD, errhandler=lib.MPI_ERRORS_ARE_FATAL):
    """Set the error handler for a communicator.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator
    errhandler : MPI_Errhandler
        New error handler for communicator
    """
    ret = lib.MPI_Comm_set_errhandler(comm, errhandler)
    check_error(ret)


def comm_set_fatal_errhandler(comm=lib.MPI_COMM_WORLD):
    """Set the fatal error handler for a communicator.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator (handle)
    """
    errhandler = lib.MPI_ERRORS_ARE_FATAL
    ret = lib.MPI_Comm_set_errhandler(comm, errhandler)
    check_error(ret)


def comm_set_nonfatal_errhandler(comm=lib.MPI_COMM_WORLD):
    """Set the non fatal error handler for a communicator.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator (handle)
    """
    errhandler = lib.MPI_ERRORS_RETURN
    ret = lib.MPI_Comm_set_errhandler(comm, errhandler)
    check_error(ret)


def comm_rank(comm=lib.MPI_COMM_WORLD):
    """Determine the rank of the calling process in the communicator.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator

    Returns
    -------
    rank : int
        Rank of the calling process in the group of comm
    """
    rank = ffi.new("int*")
    ret = lib.MPI_Comm_rank(comm, rank)
    check_error(ret)
    return rank[0]


def comm_size(comm=lib.MPI_COMM_WORLD):
    """Determine the size of the group associated with a communicator.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator

    Returns
    -------
    size : int
        Number of processes in the group of comm
    """
    size = ffi.new("int*")
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
    name = ffi.new("char[]", lib.MPI_MAX_PROCESSOR_NAME)
    resultlen = ffi.new("int*")
    ret = lib.MPI_Get_processor_name(name, resultlen)
    check_error(ret)
    proc_name = ffi.string(name, resultlen[0])
    proc_name = proc_name.decode("ascii")
    return proc_name


def send(buf, dest, tag, comm=lib.MPI_COMM_WORLD, datatype=lib.MPI_BYTE):
    """Perform a blocking send.

    Parameters
    ----------
    buf : bytes or any object supporting buffer interface
        The send buffer
    dest : int
        Rank of destination
    tag : int
        Message tag
    comm : MPI_Comm
        Communicator
    datatype : MPI_Datatype
        Datatype of each send buffer element
    """
    if isinstance(buf, bytes):
        cbuf = ffi.new("char[]", buf)
    else:
        cbuf = ffi.from_buffer("char[]", buf)
    count = len(cbuf)

    ret = lib.MPI_Send(cbuf, count, datatype, dest, tag, comm)
    check_error(ret)


def recv(
    buf,
    source=lib.MPI_ANY_SOURCE,
    tag=lib.MPI_ANY_TAG,
    comm=lib.MPI_COMM_WORLD,
    datatype=lib.MPI_BYTE,
    status=None,
):
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
        Communicator
    datatype : MPI_Datatype
        Datatype of each receive buffer element
    status : MPI_Status*
        Status object
        If status is None a new status object is created.

    Returns
    -------
    status : MPI_Status*
        Status object
    """
    cbuf = ffi.from_buffer("char[]", buf, require_writable=True)
    count = len(cbuf)
    if status is None:
        status = ffi.new("MPI_Status*")
    ret = lib.MPI_Recv(cbuf, count, datatype, source, tag, comm, status)
    check_error(ret)
    return status


def barrier(comm=lib.MPI_COMM_WORLD):
    """Blocks until all processes in the communicator have reached this routine.

    Parameters
    ----------
    comm : MPI_Comm
        Communicator
    """
    ret = lib.MPI_Barrier(comm)
    check_error(ret)


def isend(buf, dest, tag, comm=lib.MPI_COMM_WORLD, datatype=lib.MPI_BYTE, request=None):
    """Begin a nonblocking send.

    Parameters
    ----------
    buf : bytes or any object supporting buffer interface
        The send buffer
    dest : int
        Rank of destination
    tag : int
        Message tag
    comm : MPI_Comm
        Communicator
    datatype : MPI_Datatype
        Datatype of each send buffer element
    request : MPI_Request*
        Communication request
        If request is None a new request object is created.

    Returns
    -------
    request : MPI_Request*
        Communication request
    """
    if isinstance(buf, bytes):
        cbuf = ffi.new("char[]", buf)
    else:
        cbuf = ffi.from_buffer("char[]", buf)
    count = len(cbuf)
    if request is None:
        request = ffi.new("MPI_Request*")

    ret = lib.MPI_Isend(cbuf, count, datatype, dest, tag, comm, request)
    check_error(ret)

    return request


def irecv(
    buf,
    source=lib.MPI_ANY_SOURCE,
    tag=lib.MPI_ANY_TAG,
    comm=lib.MPI_COMM_WORLD,
    datatype=lib.MPI_BYTE,
    request=None,
):
    """Begin a nonblocking receive.

    Parameters
    ----------
    buf : a writable object supporting buffer interface
        The receive buffer
    source : int
        Rank of source
    tag : int
        Message tag
    comm : MPI_Comm
        Communicator
    datatype : MPI_Datatype
        Datatype of each receive buffer element
    request : MPI_Request*
        Communication request
        If request is None a new request object is created.

    Returns
    -------
    request : MPI_Request*
        Communication request
    """
    cbuf = ffi.from_buffer("char[]", buf, require_writable=True)
    count = len(cbuf)
    if request is None:
        request = ffi.new("MPI_Request*")
    ret = lib.MPI_Irecv(cbuf, count, datatype, source, tag, comm, request)
    check_error(ret)
    return request


def wait(request, status=None):
    """Wait for an MPI request to complete.

    Parameters
    ----------
    request : MPI_Request*
        Communication request
    status : MPI_Status*
        Status object
        If status is None a new status object is created.

    Returns
    -------
    status : MPI_Status*
        Status object
    """
    if status is None:
        status = ffi.new("MPI_Status*")
    ret = lib.MPI_Wait(request, status)
    check_error(ret)
    return status


def test(request, status=None):
    """Test for the completion of a request.

    Parameters
    ----------
    request: MPI_Request*
        Communication request
    status : MPI_Status*
        Status object
        If status is None a new status object is created.

    Returns
    -------
    flag : bool
        True if operation completed
    status : MPI_Status*
        Status object
    """
    flag = ffi.new("int*")
    if status is None:
        status = ffi.new("MPI_Status*")
    ret = lib.MPI_Test(request, flag, status)
    check_error(ret)
    return flag[0], status


def cancel(request):
    """Cancel a communication request.

    Parameters
    ----------
    request : MPI_Request*
        Communication request
    """
    ret = lib.MPI_Cancel(request)
    check_error(ret)


def waitany(requests, status=None):
    """Wait for any specified MPI Request to complete.

    Parameters
    ----------
    requests : MPI_Request[]
        Array of requests
    status : MPI_Status*
        Status object
        If status is None a new status object is created.

    Returns
    -------
    indx : int
        Index of handle for operation that completed (integer).
        In the range 0 to len(requests) -1.
    status : MPI_Status*
        Status object.
    """
    indx = ffi.new("int*")
    if status is None:
        status = ffi.new("MPI_Status*")
    count = len(requests)
    ret = lib.MPI_Waitany(count, requests, indx, status)
    check_error(ret)
    return indx[0], status


def waitall(requests, statuses=None):
    """Wait for all given MPI Requests to complete.

    Parameters
    ----------
    requests : MPI_Request[]
        Array of requests
    statuses : MPI_Status[]
        Array of status objects
        If statuses is None an array of statues object will be created.

    Returns
    -------
    statuses : list of MPI_Status*
        Array of status objects
    """
    if statuses is None:
        statuses = ffi.new("MPI_Status[]", len(requests))
    else:
        assert len(requests) == len(statuses)
    count = len(requests)
    ret = lib.MPI_Waitall(count, requests, statuses)
    check_error_in_status(ret, statuses)


def waitsome(requests, statuses=None):
    """Wait for some given MPI Requests to complete.

    Parameters
    ----------
    requests : MPI_Request[]
        Array of requests
    statuses : MPI_Status[]
        Array of status objects
        If statuses is None an array of statues object will be created.

    Returns
    -------
    indices : list of int
        Array of indices of operations that completed
    statuses : list of statuses
        Array of status objects
    """
    if statuses is None:
        statuses = ffi.new("MPI_Status[]", len(requests))
    else:
        assert len(requests) == len(statuses)
    incount = len(requests)
    outcount = ffi.new("int*")
    indices = ffi.new("int[]", incount)
    ret = lib.MPI_Waitsome(incount, requests, outcount, indices, statuses)
    check_error_in_status(ret, statuses)

    indices = [indices[i] for i in range(outcount[0])]
    return indices, statuses


def testany(requests, status=None):
    """Test for completion of any previdously initiated requests.

    Parameters
    ----------
    requests : MPI_Request[]
        Array of requests
    status : MPI_Status*
        Status object
        If status is None a new status object is created.

    Returns
    -------
    flag : bool
        True if one of the operations is complete
    indx : int
        Index of handle for operation that completed (integer).
        In the range 0 to len(requests) -1.
    status : MPI_Status*
        Status object.
    """
    indx = ffi.new("int*")
    flag = ffi.new("int*")
    if status is None:
        status = ffi.new("MPI_Status*")
    count = len(requests)
    ret = lib.MPI_Testany(count, requests, indx, flag, status)
    check_error(ret)
    return bool(flag[0]), indx[0], status


def testall(requests, statuses=None):
    """Test for the completion of all previously initiated requests.

    Parameters
    ----------
    requests : MPI_Request[]
        Array of requests
    statuses : MPI_Status[]
        Array of status objects
        If statuses is None an array of statues object will be created.

    Returns
    -------
    flag : bool
        True if all requests have completed, false otherwise
    statuses : list of MPI_Status*
        Array of status objects
    """
    flag = ffi.new("int*")
    if statuses is None:
        statuses = ffi.new("MPI_Status[]", len(requests))
    else:
        assert len(requests) == len(statuses)
    count = len(requests)
    ret = lib.MPI_Testall(count, requests, flag, statuses)
    check_error_in_status(ret, statuses)

    return bool(flag[0]), statuses


def testsome(requests, statuses=None):
    """Tests for some given requests to complete.

    Parameters
    ----------
    requests : MPI_Request[]
        Array of requests
    statuses : MPI_Status[]
        Array of status objects
        If statuses is None an array of statues object will be created.

    Returns
    -------
    indices : list of int
        Array of indices of operations that completed
    statuses : list of statuses
        Array of status objects
    """
    if statuses is None:
        statuses = ffi.new("MPI_Status[]", len(requests))
    else:
        assert len(requests) == len(statuses)
    incount = len(requests)
    outcount = ffi.new("int*")
    indices = ffi.new("int[]", incount)
    ret = lib.MPI_Testsome(incount, requests, outcount, indices, statuses)
    check_error_in_status(ret, statuses)

    indices = [indices[i] for i in range(outcount[0])]
    return indices, statuses


def bcast(buf, root, comm=lib.MPI_COMM_WORLD, datatype=lib.MPI_BYTE):
    """Broadcast a message from the process with rank "root" to all other processes of the communicator.

    Parameters
    ----------
    buf : bytes or any object supporting buffer interface
        Starting address of buffer
    root : int
        Rank of broadcast root
    comm : MPI_Comm
        Communicator
    datatype : MPI_Datatype
        Data type of buffer
    """
    if root == comm_rank(comm):
        cbuf = ffi.from_buffer("char[]", buf)
    else:
        cbuf = ffi.from_buffer("char[]", buf, require_writable=True)
    count = len(cbuf)

    ret = lib.MPI_Bcast(cbuf, count, datatype, root, comm)
    check_error(ret)


def ibcast(buf, root, comm=lib.MPI_COMM_WORLD, datatype=lib.MPI_BYTE, request=None):
    """Broadcasts a message from the process with rank "root" to all other processes of the communicator in a nonblocking way.

    Parameters
    ----------
    buf : bytes or any object supporting buffer interface
        Starting address of buffer
    root : int
        Rank of broadcast root
    comm : MPI_Comm
        Communicator
    datatype : MPI_Datatype
        Data type of buffer
    request : MPI_Request*
        Communication request
        If request is None a new request object is created.
    """
    if root == comm_rank(comm):
        cbuf = ffi.from_buffer("char[]", buf)
    else:
        cbuf = ffi.from_buffer("char[]", buf, require_writable=True)
    count = len(cbuf)
    if request is None:
        request = ffi.new("MPI_Request*")

    ret = lib.MPI_Ibcast(cbuf, count, datatype, root, comm, request)
    check_error(ret)

    return request
