"""Use blocking send recieve to send hello."""

import yapympi.base as mpi

MSG = "hello".encode("utf-8")


def main():
    mpi.init()
    try:
        mpi.barrier()

        rank = mpi.comm_rank()
        if rank == 0:
            buf = MSG
            req = mpi.isend(buf, dest=1, tag=0)
            mpi.wait(req)
        else:
            buf = bytearray(10)
            req = mpi.irecv(buf, source=0, tag=0)
            status = mpi.wait(req)
            assert mpi.get_count(status) == len(MSG)
            assert buf[:len(MSG)] == MSG
    finally:
        mpi.finalize()


if __name__ == "__main__":
    main()
