import sys
sys.path.insert(0, "build/lib")

from mpi4py import MPI
import pygpu_tests

if MPI.COMM_WORLD.rank == 0:
    print(f"Running test with {MPI.COMM_WORLD.size} ranks...")

pygpu_tests.init_cuda()
