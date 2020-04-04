"""Print rank, size and processor name on all nodes."""

import yapympi.base as mpi


def main():
    mpi.init()
    try:
        print(mpi.comm_size(), mpi.comm_rank(), mpi.get_processor_name())
    finally:
        mpi.finalize()


if __name__ == "__main__":
    main()
