#!/bin/tcsh
#
# This sets the total number of cores to use
#SBATCH --ntasks 16
#
# This sets how many cores to use on each node
#SBATCH --tasks-per-node=16
#
#SBATCH --exclusive
#SBATCH -J job_name
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH -t 2:00:00
#

# Need to load the same MPI module which parallel_tasks was built with
module purge
module load intel_comp/2018 intel_mpi

#
# Run the python script
#
# This runs 20 instances of hello.py, passing the job index 1-20
# as a command line parameter. The number of cpus used depends
# on the --ntasks parameter at the top of this script.
#
# The number of jobs to run is independent of the number of cpus used.
# Running more jobs or using fewer cpus will just take longer.
#
# Any instances of %d in the command get replaced with the job index.
#
mpirun ./parallel_tasks 1 20 "python ./hello.py %d"

