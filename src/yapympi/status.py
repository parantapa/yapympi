"""Python wrapper for MPI_Status objects."""

from .cmpi import ffi, lib
from .base import MPIError, check_error

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

    def __init__(self, status, datatype=lib.MPI_BYTE):
        """Initialize.

        Parameters
        ----------
        status : MPI_Status or MPI_Status*
            The MPI_Status object to wrap
        datatype : MPI_Datatype
            The datatype to use for computing count
        """
        self.source = status.MPI_SOURCE
        self.tag = status.MPI_TAG
        self.error = status.MPI_ERROR

        cnt = ffi.new("int*")
        if ffi.typeof(status) is ffi.typeof("MPI_Status"):
            status_p = ffi.addressof(status)
        else:
            status_p = status
        retcode = lib.MPI_Get_count(status_p, datatype, cnt)
        if retcode != lib.MPI_SUCCESS:
            raise MPIError(retcode)
        self.count = cnt[0]

    def __repr__(self):
        fmt = "MPIStatus(source=%d, tag=%d, error=%d, count=%d)"
        return fmt % (self.source, self.tag, self.error, self.count)

    def check_error(self):
        """Raise MPIError if error is present."""
        if self.error != lib.MPI_SUCCESS:
            raise MPIError(self.error)
