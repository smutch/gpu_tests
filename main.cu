#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <sstream>


#define throw_on_cuda_error(cuda_code) \
{ \
_throw_on_cuda_error((cuda_code), __FILE__, __func__, __LINE__); \
}

__host__ void _throw_on_cuda_error(cudaError_t cuda_code, const char* file, const char* func, int line)
{
    std::ostringstream mesg;
    mesg << "CUDA error code " << (int)cuda_code << ": " << func << "(" << file << " +" << line << ")";
    if (cuda_code != cudaSuccess)
        throw std::runtime_error(mesg.str());
}

void init_cuda()
{
    int world_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    struct cudaDeviceProp properties;
    int device = -1;

    try {
        // get the number of devices available to this rank
        int num_devices = 0;
        throw_on_cuda_error(cudaGetDeviceCount(&num_devices));

        // do your thang to assign this rank to a device
        throw_on_cuda_error(cudaSetDevice(world_rank % num_devices));  // alternate assignment between ranks

        // do a check to make sure that we have a working assigned device
        throw_on_cuda_error(cudaFree(0));

        // Get the device assigned to this context
        throw_on_cuda_error(cudaGetDevice(&device));

        // Get the properties of the device assigned to this context
        throw_on_cuda_error(cudaGetDeviceProperties(&properties, device));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << world_rank << ": CUDA initialization completed successfully on " << properties.name << "[device=" << device << "] ..." << std::endl;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    init_cuda();
    MPI_Finalize();
    return 0;
}
