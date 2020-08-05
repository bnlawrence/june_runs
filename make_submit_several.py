import sys
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--iteration',action='store',default=1)
parser.add_option('--slurm_name',action='store',default='june_submit')
parser.add_option('--output',action='store',default=None)
(options,args) = parser.parse_args()

print()

iteration = options.iteration

output = options.output
if options.output is None:
    output = f'./{options.slurm_name}_several_{iteration}.sh'

if len(args) == 1:
    n_submit = args[0]

    try:
	    n_submit = int(n_submit)+1
	    entries = [i for i in range(0,n_submit)]
    except:
	    entries = [int(i) for i in n_submit.split(',')]

    print('-- can give 1 or 2 args: [lower,] upper integers eg 10 25 to create files 10,11,..,24,25')



elif len(args) == 2:
    lower = int(args[0])
    upper = int(args[1])+1

    entries = [i for i in range(lower,upper)]

else:
    print('give [lower,] upper job numbers OR comma-sep entries')

lines = []

print('entries are', entries)

warn = True

for i in entries:
    script_path = f'submit_scripts/{options.slurm_name}_{iteration}_{i:03d}.sh'
    if os.path.exists(f'./{script_path}') is False:
        if warn:
            w = (f'\033[31mWARN: no script \033[0meg. {script_path}.\n'
                + f'    Are you sure you set --slurm_name correctly? currently "{options.slurm_name}"')
            print( w )
            warn = False
    else:
        lines.append(f'sbatch {script_path}')



with open(output,'w+') as f:
	f.write('#!/bin/bash' + '\n\n')
	for line in lines:
		f.write(line + '\n')

print(f'written at {output}')




