"""Define a wrapper class for easyily using MPI_Status objects."""

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
        self.status = status
        self.datatype = datatype

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
        cnt = ffi.new("int*")
        if ffi.typeof(self.status) is ffi.typeof("MPI_Status"):
            status_ptr = ffi.addressof(self.status)
        else:
            status_ptr = self.status
        ret = lib.MPI_Get_count(status_ptr, self.datatype, cnt)
        check_error(ret)
        return cnt[0]

    def check_error(self):
        """Raise MPIError if error is present."""
        if self.error != lib.MPI_SUCCESS:
            raise MPIError(self.error)
