"""Use blocking send recieve to send hello."""

import yapympi.base as mpi


def main():
    mpi.init()
    try:
        mpi.barrier()

        rank = mpi.comm_rank()
        if rank == 0:
            buf = "hello".encode("utf-8")
            mpi.send(buf, dest=1, tag=0)
        else:
            buf = bytearray(10)
            status = mpi.recv(buf, source=0, tag=0)
            print(status)

        print(rank, buf)
    finally:
        mpi.finalize()


if __name__ == "__main__":
    main()
