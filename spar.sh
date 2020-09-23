#!/bin/bash -l
#SBATCH --ntasks 32
#SBATCH -J june_mpi
#SBATCH -o mpi.out
#SBATCH -e mpi.err
#SBATCH -p cosma6
#SBATCH -A dp004
#SBATCH -t 01:00:00

module purge
module load python/3.6.5
module load gnu_comp/7.3.0
module load openmpi/3.0.1
module load gnu-parallel

echo "job submitted!"
echo "Running $SLURM_NTASKS"

# --delay .2 prevents overloading the controlling node
# -j is the number of tasks parallel runs so we set it to $SLURM_NTASKS
# --joblog makes parallel create a log of tasks that it has already run
# -u to ungroup stdout so that is printed immediately
parallel="parallel -u --delay .2 -j $SLURM_NTASKS --joblog logs/runtask.log"

# this runs the parallel command we want
# in this case, we are running a script named runtask
# parallel uses ::: to separate options. Here {0..99} is a shell expansion
# so parallel will run the command passing the numbers 0 through 99
# via argument {1}
$parallel "mpirun -np 2 python3 -u ./run_parallel_simulation.py {1}" ::: {0..15}

