cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "icc")
set(CMAKE_CXX_COMPILER "icpc")

project(gpu_tests)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd3802")  # TODO: Check this out

find_package(CUDA REQUIRED)

# Pass options to NVCC
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_PROPAGATE_HOST_FLAGS ON)

# THE MAGIC LINES! ###########################################################
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60 --compiler-options -fPIC")
##############################################################################

################################################################################
# This ugly hack is needed to work around a bug in the fftw3.h header file
#    see the following for a discussion: https://github.com/FFTW/fftw3/issues/18
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D__INTEL_COMPILER")
################################################################################

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CUDA_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CUDA_EXE_LINKER_FLAGS}")

# Add the Meraxes lib to the project
set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
cuda_add_executable(gpu_tests main.cpp init_cuda.cu)

find_package(MPI REQUIRED)
target_link_libraries(gpu_tests PUBLIC MPI::MPI_C)
