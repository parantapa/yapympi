"""Test multiple non blocking send recv."""

import yapympi.base as mpi
from yapympi.status import MPIStatus

MSG = "hello".encode("utf-8")
NMSGS = 10


def main():
    mpi.init()
    mpi.barrier()

    rank = mpi.comm_rank()
    if rank == 0:
        reqs = []
        for _ in range(NMSGS):
            buf = MSG
            req = mpi.isend(buf, dest=1, tag=0)
            reqs.append(req)
        print(rank, reqs, flush=True)

        reqs_arr = mpi.list_to_array("MPI_Request", reqs)
        mpi.waitall(reqs_arr)
        print(rank, "waitall finished", flush=True)
    else:
        reqs, bufs = [], []
        for _ in range(NMSGS):
            buf = bytearray(10)
            req = mpi.irecv(buf, source=0, tag=0)
            reqs.append(req)
            bufs.append(buf)
        print(rank, reqs)

        nfinished = 0
        while nfinished < NMSGS:
            reqs_arr = mpi.list_to_array("MPI_Request", reqs)
            indices, statuses = mpi.waitsome(reqs_arr)
            for i in sorted(indices, reverse=True):
                print(i, MPIStatus(statuses[i]), bufs[i], flush=True)
                del bufs[i]
                del reqs[i]
                nfinished += 1

    mpi.barrier()
    mpi.finalize()


if __name__ == "__main__":
    main()
