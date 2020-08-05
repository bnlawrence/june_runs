#!/bin/bash -l
#
#SBATCH --ntasks 28
#SBATCH -J job_name
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma7
#SBATCH -A project  #e.g. dp004
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --mail-type=END                          # notifications for job done & fail
#SBATCH --mail-user=your.email@your.host.ac.uk

module purge
#load the modules used to build your program.
module load gnu_comp/7.3/0
module load openmpi/3.0.1


# Run the program 32 times (on 28 cores).
mpirun -np 28 ./parallel_tasks 0 31 "python ./hello.py %d"