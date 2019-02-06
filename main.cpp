#include <mpi.h>
#include "init_cuda.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    init_cuda();
    MPI_Finalize();
    return 0;
}
