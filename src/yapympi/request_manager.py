"""Async MPI Request Manager."""

from .cmpi import ffi, lib
from .status import MPIStatus
from .error import MPIError, MPIStatusErrors


class RequestManager:
    def __init__(self, capacity, comm=lib.MPI_COMM_WORLD, datatype=lib.MPI_BYTE):
        self.capacity = int(capacity)
        self.comm = comm
        self.datatype = datatype

        self.size = 0
        self.requests = ffi.new("MPI_Request[]", self.capacity)
        self.handles = []

        self.indices = ffi.new("int[]", self.capacity)
        self.outcount = ffi.new("int*")
        self.statuses = ffi.new("MPI_Status[]", self.capacity)

    def send(self, buf, dest, tag, handle=None):
        """Begin a nonblocking send.

        Parameters
        ----------
        buf : bytes or any object supporting buffer interface
            The send buffer
        dest : int
            Rank of destination
        tag : int
            Message tag
        handle : object
            Handle object to be returned when the requst is complete
        """
        if self.size == self.capacity:
            raise ValueError("Request manager has reached capacity")

        if isinstance(buf, bytes):
            cbuf = ffi.new("char[]", buf)
        else:
            cbuf = ffi.from_buffer("char[]", buf)
        count = len(cbuf)

        request = self.requests[self.size]
        request_p = ffi.addressof(request)

        retcode = lib.MPI_Isend(
            cbuf, count, self.datatype, dest, tag, self.comm, request_p
        )
        if retcode != lib.MPI_SUCCESS:
            raise MPIError(retcode)

        self.handles.append(handle)
        self.size += 1

    def recv(self, buf, source=lib.MPI_ANY_SOURCE, tag=lib.MPI_ANY_TAG, handle=None):
        """Begin a nonblocking receive.

        Parameters
        ----------
        buf : a writable object supporting buffer interface
            The receive buffer
        source : int
            Rank of source
        tag : int
            Message tag
        handle : object
            Handle object to be returned when the requst is complete
        """
        if self.size == self.capacity:
            raise ValueError("Request manager has reached capacity")

        cbuf = ffi.from_buffer("char[]", buf, require_writable=True)
        count = len(cbuf)

        request = self.requests[self.size]
        request_p = ffi.addressof(request)

        retcode = lib.MPI_Irecv(
            cbuf, count, self.datatype, source, tag, self.comm, request_p
        )
        if retcode != lib.MPI_SUCCESS:
            raise MPIError(retcode)

        self.handles.append(handle)
        self.size += 1

    def _del_request(self, idx):
        """Delete the requst at the given index."""
        if idx == self.size - 1:
            del self.handles[self.size - 1]
            self.size -= 1

        elif idx < self.size - 1:
            self.requests[idx] = self.requests[self.size - 1]
            self.handles[idx] = self.handles[self.size - 1]
            del self.handles[self.size - 1]
            self.size -= 1

        raise ValueError("Can't remove index idx=%d; size=%d" % (idx, self.size))

    def test(self):
        """Test all pending requests for completion."""
        if not self.size:
            return None, None

        lib.memset(self.indices, 0, ffi.sizeof(self.indices))
        self.outcount[0] = 0
        lib.memset(self.statuses, 0, ffi.sizeof(self.statuses))

        retcode = lib.MPI_Testsome(
            self.size, self.requests, self.outcount, self.indices, self.statuses
        )
        if retcode != lib.MPI_SUCCESS:
            if retcode == lib.MPI_ERR_IN_STATUS:
                errorcodes, handles = [], []
                for i in range(self.size):
                    if self.statuses[i].MPI_ERROR != lib.MPI_SUCCESS:
                        errorcodes.append(self.statuses[i].MPI_ERROR)
                        handles.append(self.handles[i])
                raise MPIStatusErrors(errorcodes, handles)
            else:
                raise MPIError(retcode)

        outcount = int(self.outcount[0])
        if not outcount:
            return None, None

        indices = [self.indices[i] for i in range(outcount)]

        handles, statuses = [], []
        for idx in range(indices):
            handles.append(self.handles[idx])
            statuses.append(MPIStatus(self.statuses[idx]))

        for idx in sorted(indices, reverse=True):
            self._del_request(idx)

        return handles, statuses
