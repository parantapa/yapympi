"""MPI Error wrappers."""

from .cmpi import ffi, lib


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
    retcode = lib.MPI_Error_string(errorcode, string, resultlen)
    if retcode != lib.MPI_SUCCESS:
        return "Unknown MPI Error: %d" % errorcode
    error_str = ffi.string(string, resultlen[0])
    error_str = error_str.decode("ascii")
    return error_str


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
        super().__init__(errcode)

        self.errcode = errcode
        self.errorstr = error_string(self.errcode)

    def __str__(self):
        return "MPI Error: %d: %s" % (self.errcode, self.errorstr)


class MPIStatusErrors(MPIErrorBase):
    """MPI runtime error in status.

    Attributes
    ----------
    errcodes : list of int
        List of MPI error codes
    errstrs : list of str
        List of string representations of the error codes
    handles : list of handles
        List of handle objects associated with the failed requests
    """

    def __init__(self, errcodes, handles=None):
        super().__init__(errcodes, handles=None)
        if handles is None:
            handles = [None] * len(errcodes)
        else:
            assert len(handles) == len(errcodes)

        self.errcodes = errcodes
        self.errstrs = [error_string(c) for c in self.errcodes]
        self.handles = handles

    def __str__(self):
        it = zip(self.errcodes, self.errstrs, self.handles)
        it = ["Error: %d: %s\n%r" % r for r in it]
        it = "\n".join(it)
        return "MPI Status Errors: %d errors\n%s" % (len(self.errcodes), it)
