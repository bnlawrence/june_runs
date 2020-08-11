import psutil
import os
import sys
from optparse import OptionParser

import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import json
#import seaborn as sns

from pathlib import Path

## important, remove
from june import World
from june.demography.geography import Geography
from june.demography import Demography
from june.interaction import Interaction
from june.infection import Infection
from june.infection.health_index import HealthIndexGenerator
from june.infection.transmission import TransmissionConstant
from june.groups import Hospitals, Schools, Companies, Households, CareHomes, Cemeteries, Universities
#from june.groups.leisure import leisure, Cinemas, Pubs, Groceries
from june.groups.leisure import generate_leisure_for_config, Cinemas, Pubs, Groceries
from june.simulator import Simulator
from june.infection_seed import InfectionSeed, Observed2Cases
from june.policy import Policy, Policies, SocialDistancing, Quarantine
from june import paths
from june.logger.read_logger import ReadLogger
from june.infection.infection import InfectionSelector
from june.world import generate_world_from_hdf5, generate_world_from_geography

import generate_parameters as gp
import polict_parameters as pp
import extract_data as extract




def print_memory_status(when='now'):
    mem = psutil.virtual_memory()
    tot = f"total: {mem.total/1024**3:.2f}G"
    used = f"used: {mem.used/1024**3:.2f}G"
    perc = f"perc: {mem.percent:.2f} \%"
    avail = f"avail: {mem.available/1024**3:.2f}G"
    print(f"memory {when}: \n    {tot} {used} {perc} {avail}")



# parameters

work_dir = Path( os.getcwd() )

print('start')

###===============Read the parameters, set some args==============###

parser = OptionParser()
parser.add_option('--test',action='store',default=-99) # ignore this.
parser.add_option('--num_samples',dest='num_samples',action='store',default=250)
parser.add_option('--world_dir',dest='world_dir',action='store',
    default=f'/cosma6/data/dp004/dc-quer1/june_worlds/up_to_date/'
)
parser.add_option('--results_dir',dest='results_dir',action='store',
    default='june_results'
)
(options,args) = parser.parse_args()


#lhs_path = f'{work_dir}/lhs_array-constrained-26-06-2020.npy'
lhs_array = gp.generate_lhs(num_samples=250)
###==========250 PARMAS HARDCODED!!!!!!================###

pol_param_array = pp.generate_lhs(num_samples=options.num_samples)

beta_index = 16
index = int(args[0])
iteration = int(args[1])
if len(args) == 3:
    region_name = args[2]
else:
    region_name = 'london'

CONFIG_PATH = work_dir / 'config.yaml'

world_dir = Path(options.world_dir)
world_file = world_dir / f'{region_name}.hdf5'

checkpoint_date = '2020-03-15'
checkpoint_path = work_dir / Path(
    'june_runs_xnexp/{region}/iteration_1/run_{beta_index:03d}/checkpoint_{checkpoint_date}.pkl'
)

if os.path.exists(world_file) is False:
    raise IOError(f'No world_file {world_file}')

if os.path.exists(CONFIG_PATH) is False:
    raise IOError('No config!')

parameters = gp.generate_parameters_from_lhs(lhs_array=lhs_array, idx=beta_index)
pol_parameters = pp.generate_paramters_from_lhs(lhs_array=pol_param_array,idx=index)

print(index,'parameters',parameters)

print(f'there are {len(parameters.keys())} params!')

SAVE_PATH = work_dir / Path(f'{options.results_dir}/{region_name}/iteration_{iteration}/run_{index:03d}')
summary_dir = work_dir / Path(f'{options.results_dir}/{region_name}/summaries/iteration_{iteration}')

if os.path.exists(SAVE_PATH) is False:
    print(f'make dir {SAVE_PATH}')
    os.makedirs(SAVE_PATH)
else:
    print(f'dir exists {SAVE_PATH}!')
    if os.path.exists(SAVE_PATH / 'logger.hdf5'):
        print('logger exists!! are you sure to overwrite?')

print(f'SAVE TO {SAVE_PATH}')
with open(SAVE_PATH / 'parameters.json', 'w') as f:
    json.dump(parameters,f)


if os.path.exists(summary_dir) is False:
    try:
        os.makedirs(summary_dir)
        print(f'make summary dir {summary_dir}')
    except:
        print(f'Cant make {summary_dir}')
else:
    print(f'summary dir exists {summary_dir}')

print_memory_status(when='before load')


###===============Modify betas==============###

if (parameters is not None) & ('asymptomatic_ratio' in parameters.keys()):
    asymptomatic_ratio = parameters['asymptomatic_ratio']
else:
    print('no "asymptomatic_ratio" in parameters!')
    asymptomatic_ratio = 0.2

health_index_generator = HealthIndexGenerator.from_file(
    asymptomatic_ratio=asymptomatic_ratio
)


transmission_config = Path('defaults/transmission/XNExp.yaml')

selector = InfectionSelector.from_file(
    transmission_config_path=paths.configs_path / transmission_config,    
    health_index_generator=health_index_generator, 
    #trajectories_config_path=f"{work_dir}" +"/shifted_trajectories.yaml"
)

print('selector OK')

interaction = Interaction.from_file()
print('interaction OK')

print('betas before')
print(interaction.beta)

if parameters is not None:
    gp.set_interaction_parameters(parameters, interaction)
else:
    print('NO PARAMETERS - remain default!')

print('beta_after')
print(interaction.beta)





###==============Do policies================###

#policies = Policies.from_file("/cosma6/data/dp004/dc-quer1/june_runs/london_policy.yaml")
policies = Policies.from_file()


###-----mod social distancing

betas_to_reduce = [
    'pub', 'grocery', 'cinema', 'commute_unit', 'commute_city_unit', 'hospital', 
    'care_home', 'company', 'school', 'university'
]


if 'beta_factor_2' in pol_parameters.keys():
    beta_factor_2 = pol_parameters['beta_factor_2'] # so b1 > b2 always
    beta_factor_1 = beta_factor_2 + (1.-beta_factor_2)*pol_parameters['x_factor']
    print(f'beta_factor_1 {beta_factor_1:%.2f}, beta_factor_2 {beta_factor_2:%.2f}')
else:
    beta_factor_2 = 0.8
    print('No beta_factor_2!')

soc_dist1 = {
    'start_date' : checkpoint_date,
    'beta_factors' : {key:beta_factor_1 for key in betas_to_reduce}
} # dict key is **beta_factorS** plural

gp.modify_policy(policies,'social_distancing',number=1,values=soc_dist1)

soc_dist2 = {
    'beta_factors' : {key:beta_factor_2 for key in betas_to_reduce}
}

gp.modify_policy(policies,'social_distancing',number=2,values=soc_dist2)

#--------mod shielding policy

if 'shielding_compliance_2' in pol_parameters.keys():
    sh_comp_2 = pol_parameters['shielding_compliance_2'] # so b1 > b2 always
    sh_comp_1 = sh_comp_2*pol_parameters['x_factor']
    print(f'sh_comp_1 {sh_comp_1:%.2f}, sh_comp_2 {sh_comp_2:%.2f}')

else:
    sh_comp_2 = 0.8
    print('No shielding_factor_2!')

shield1 = {
    'start_date' : checkpoint_date,
    'beta_factors' : {key:shielding_factor_1 for key in betas_to_reduce}
}

gp.modify_policy(policies,'social_distancing',number=1,values=shield1)

shield2 = {
    'beta_factors' : {key:shielding_factor_2 for key in betas_to_reduce}
}

gp.modify_policy(policies,'social_distancing',number=2,values=shield2)

#------mode quarantine compliance

if 'household_compliance_2' in pol_parameters.keys():
    hh_compliance_2 = pol_parameters['household_compliance_2']
    hh_compliance_1 = hh_compliance_2*pol_parameters['x_factor']
else:
    print('no household_compliance_2!')
    household_compliance_2 = 0.6

quar1 = {'start_date' : checkpoint_date, 'household_compliance' : hh_compliance_1}
gp.modify_policy(policies,'quarantine',number=1,values=quar1)

quar2 = {'household_compliance' : hh_compliance_2}
gp.modify_policy(policies,'quarantine',number=2,values=quar2)



#for p in policies:
    #print(p.__dict__)

###==============Load the world==================###

#world_file = "/cosma7/data/dp004/dc-quer1/JUNE/scripts/tests.hdf5"
#world_file = f'{work_dir}/june_worlds/small_test.hdf5'

# these are for small_100k_world tests...
if os.path.exists(world_file) is False:
    raise IOError(f'no world {world_file}!')

else:
    print(f'load world from {world_file}')
    world = generate_world_from_hdf5(world_file, chunk_size=1_000_000)
    print("World loaded OK")
    leisure = generate_leisure_for_config(world, CONFIG_PATH)
    print('leisure ok')


print_memory_status(when='after_load')

###==============Do seeding================###

if (parameters is not None) & ('seed_strength' in parameters.keys()):
    seed_strength = parameters['seed_strength']
else:
    seed_strength = 1.0

oc = Observed2Cases.from_file(
    super_areas=world.super_areas, 
    health_index = selector.health_index_generator,
    )
n_cases_df = oc.cases_from_deaths()
# Seed over 5 days
n_cases_to_seed_df = n_cases_df.loc['2020-03-01':'2020-03-02']
infection_seed = InfectionSeed.from_file(super_areas=world.super_areas, 
        selector=selector,
        n_cases_region = n_cases_to_seed_df,
        seed_strength = seed_strength,
        )

print(f'seed_strength set to {infection_seed.seed_strength}')

print('Seed OK')


###===============Set up sim==============###


t1= time.time()
simulator = Simulator.from_checkpoint(
    world, 
    checkpoint_path=checkpoint_path,
    interaction = interaction, 
    config_filename = CONFIG_PATH,
    leisure = leisure,
    infection_seed = infection_seed,
    infection_selector=selector,
    policies = policies,
    save_path=SAVE_PATH,
)

simulator.timer.reset()
t2= time.time()

print(f'time to load simulator:{iteration},idx:{index} took {(t2-t1)/60.} min')

print_memory_status(when='before sim')

###===========RUUUUUUUUUUUN!==========###

t1 = time.time()
simulator.run()
t2 = time.time()

print(f'simulation iter:{iteration},idx:{index} took {(t2-t1)/60.} min')
print_memory_status(when='after sim')

###===========Gather some summary stuff============###

extract.do_extract(index,SAVE_PATH,summary_dir=summary_dir,iteration=iteration)

