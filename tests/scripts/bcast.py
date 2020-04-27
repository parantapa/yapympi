"""Use blocking bcast to send messages."""

import yapympi.base as mpi

MSG = "hello".encode("utf-8")

def main():
    mpi.init()
    try:
        mpi.barrier()

        rank = mpi.comm_rank()
        if rank == 0:
            buf = MSG
            mpi.bcast(buf, 0)
        else:
            buf = bytearray(len(MSG))
            mpi.bcast(buf, 0)
            assert buf == MSG
    finally:
        mpi.finalize()


if __name__ == "__main__":
    main()
