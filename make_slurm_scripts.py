import numpy as np
import os

from optparse import OptionParser
from pathlib import Path

parser = OptionParser()
# Run information
parser.add_option('--iteration',dest='iteration',default=1)
parser.add_option('-N','--N_runs',dest='N_runs',action='store',default=250,type='int')
parser.add_option('--script',dest='script',action='store',default='june_simulation.py')
parser.add_option('--region',dest='region',action='store',default=None)

# Cosma info
parser.add_option('-p','--partition',dest='partition',default='cosma')
parser.add_option('-A','--project',dest='project',default='durham')
parser.add_option('--N_cores',dest='N_cores',default=15)
parser.add_option('--n_per_node',dest='n_per_node',action='store',default=4,type='int')
parser.add_option('-J', dest='is_jasmin', action='store', default=0, type='int')

# extra info for slurm, etc.
parser.add_option('--slurm_name',dest='slurm_name',action='store',default=None)
parser.add_option('--job_name',dest='job_name',action='store',default='june')
parser.add_option('--stdout',dest='stdout_file',action='store',default='stdout')

# submit several
parser.add_option('--submit_several_name',dest='submit_several_name',default=None)

# extra options for JUNE simulation
parser.add_option('--world_dir',dest='world_dir',action='store',default=None)
parser.add_option('--results_dir',dest='results_dir',action='store',default=None)

(options,args) = parser.parse_args()

N_runs = int(options.N_runs)

n_per_node = options.n_per_node


index_list = [d for d in np.arange(0,N_runs,n_per_node).astype(int)]

if index_list[-1] != N_runs:
    index_list.append(N_runs)

# print(index_list)

iteration = options.iteration
print(f'\nset \033[35miteration {iteration}\033[0m')

region = options.region
if region is None:
    region = 'london'
print(f'region is \033[35m{region}\033[0m')

if (options.slurm_name is None):
    if options.region is None:
        options.slurm_name = 'june_submit'
    else:
        options.slurm_name = region


work_dir = Path( os.getcwd() ) #'/cosma5/data/durham/dc-sedg2/covid/june_runs'
stdout_dir = work_dir / 'stdout'
script_dir = Path('submit_scripts')

simulation_script = work_dir / options.script
parallel_tasks = work_dir / script_dir / 'parallel_tasks'

if os.path.exists(stdout_dir) is False:
    os.makedirs(stdout_dir)

if os.path.exists(script_dir) is False:
    os.makedirs(script_dir)


sim_options = ''
if options.world_dir is not None:
    sim_options = sim_options + f'--world_dir {options.world_dir} '
if options.results_dir is not None:
    sim_options = sim_options + f'--results_dir {options.results_dir} '

PYTHON_CMD = f"python3 -u {simulation_script} %d {options.iteration} {options.region} {sim_options}"
#python cmd can now has THREE args:
#    %d: an integer
#    iteration: which "repeat" is this? 
#    region: defaults to london if not provided.

slurm_scripts = []

for i,(low,high) in enumerate(zip(index_list[:-1],index_list[1:])):

    ntasks = high-low

    stdout_name = stdout_dir / f'{options.slurm_name}_{iteration}_{i:03d}.%J'
    if bool(options.is_jasmin):
        loading_python=['module purge',
                        'module load eb/OpenMPI/gcc/4.0.0',
                        'module load jaspy/3.7/r20200606', 
			'source /gws/nopw/j04/covid_june/june_venv/bin/activate']
    else:
        loading_python = [f'module purge',
            f'module load python/3.6.5',
            f'module load gnu_comp/7.3.0',
            f'module load hdf5',
            f'module load openmpi/3.0.1']

    script_lines = [
        '#!/bin/bash -l',
        '',
        f'#SBATCH --ntasks {options.N_cores}',
        f'#SBATCH -J {options.job_name}_{iteration}_{i:03d}',
        f'#SBATCH -o {stdout_name}.out',
        f'#SBATCH -e {stdout_name}.err',
        f'#SBATCH -p {options.partition}',
        f'#SBATCH -A {options.project} #e.g. dp004',
        f'#SBATCH --exclusive',
        f'#SBATCH -t 72:00:00'] +\
        loading_python + \
        [f'# load the modules used to build your program.',
        '',
        f'# Run the program {ntasks} times (on {ntasks} cores).',
        f'mpirun -np {ntasks} {parallel_tasks} {low} {high-1} "{PYTHON_CMD}" ',
    ]

    script_name = f'{options.slurm_name}_{iteration}_{i:03d}.sh'
    script_path = work_dir / script_dir / script_name

    slurm_scripts.append(script_name)

    if i == 0:
        print(f'\nwill submit cmd \n {PYTHON_CMD}')
        print(f'\nsaved scripts at eg. \n    {script_dir / script_name}')

    with open(script_path,'w+') as script:
        for line in script_lines:
            script.write(line+'\n')





###==============Built in "submit several"================###

if options.submit_several_name is None:
    options.submit_several_name = (
        Path(f'submit_{options.region}_several_{options.iteration}.sh')
    )

    

with open(options.submit_several_name,'w+') as f:
    f.write('#!/bin/bash' + '\n\n')
    for slurm_script in slurm_scripts:
        line = f'sbatch {work_dir / script_dir / slurm_script}'
        f.write(line + '\n')

print(f'\nsubmit all scripts with:\n     \033[35mbash {options.submit_several_name}\033[0m')






