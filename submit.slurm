#!/bin/bash
#
#SBATCH --job-name=gpu_test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:01:00
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:2
#
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=smutch@unimelb.edu.au

source $HOME/chenv.sh default
printf "\nmodules\n=======\n$(module -t list 2>&1)\n\n"
printf "python  :: ${CONDA_PREFIX}\n"
printf "mpiexec :: $(which mpiexec)\n\n"

export OMP_NUM_THREADS=1
export OMPI_MCA_mpi_warn_on_fork=0
srun build/gpu_tests
