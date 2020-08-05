import numpy as np
import corner
import matplotlib.pyplot as plt


beta_names = [
    'pub',
    'grocery',
    'cinema',
    'commute_unit',
    'commute_city_unit',
    'hospital',
    'care_home',
    'company',
    'school',
    'household',
    'university'
]

beta_bounds = { 
    'pub'       : (0.01, 0.25),
    'grocery'   : (0.01, 0.25),
    'cinema'    : (0.01, 0.25),
    'commute_unit' : (0.01, 0.5),
    'commute_city_unit' : (0.01, 0.5),
    'hospital'  : (0.05,0.5),
    'care_home' : (0.05, 1.),
    'company'   : (0.05, 0.5),
    'school'    : (0.05, 0.5),
    'household' : (0.05, 0.5),
    'university': (0.01, 0.5),
}

assert set(beta_names) == set(beta_bounds.keys())

extra_names = [
    'alpha_physical',
    'asymptomatic_ratio',
    #'beta_factors',
    'seed_strength',
#    'household_complacency'
]

extra_bounds = {                
    'alpha_physical' : (1.8,3.0),
    'asymptomatic_ratio' : (0.05,0.4),
    #'beta_factors' : (0.2,0.6),
    'seed_strength' : (0.5,1.0),
    #'household_complacency' : (0.1,0.9)
}

assert set(extra_names) == set(extra_bounds.keys())
## check names in list are the same as names in bound dict.
## do names in list and dict as i wasn't sure if dict is 'Ordered' by default

all_names = beta_names + extra_names


def generate_lhs(num_samples, seed=1):
    from pyDOE2 import lhs
    from SALib.util import scale_samples
    '''Generates a latin hypercube array.'''

    bounds = (
        [beta_bounds[b] for b in beta_names] +
        [extra_bounds[p] for p in extra_names]
    )

    num_vars = len(bounds)

    # latin hypercube array sampled from 0 to 1 for all variables apart from beta_household
    lhs_array = lhs(n=num_vars, samples=num_samples, criterion='maximin', random_state=seed)
    # scale to the bounds that we want
    scale_samples(lhs_array, bounds)

    return lhs_array

def generate_parameters_from_lhs(lhs_array, idx):
    '''Generates a parameter dictionary from latin hypercube array.
       idx is an integer that should be passed from each run,
       i.e. first run will have idx = 0, second run idx = 1...
       This will index out the row from the latin hypercube.'''

    names = beta_names + extra_names

    parameters = {
        'betas': {}        
    } #'extra' parameters will be added in the loop!

    for i, name in enumerate(names):
        if name in beta_names:
            parameters['betas'][name] = lhs_array[idx][i]
        elif name in extra_names:
            parameters[name] = lhs_array[idx][i]
        else:
            print(f'parameter {name} not in names!!')

    return parameters

def set_interaction_parameters(parameters, interaction):
    '''Sets interaction parameters for the simulation from parameter dictionary.'''
    #===== HAVE TO SET SEED STRENGTH MANUALLY =====#
    for key in parameters['betas']:
        interaction.beta[key] = parameters['betas'][key]
    interaction.alpha_physical = parameters['alpha_physical']
    return interaction

def set_beta_factors(parameters, policies):
    for key in parameters['betas']:
        if key != 'household':
            policies.beta_factor[key] = parameters['beta_factors']
    return policies

if __name__ == "__main__":
    num_samples = 500 
    lhs_array = generate_lhs(num_samples, 1)
    # to generate parameters for first run use idx=0, etc
    parameters = generate_parameters_from_lhs(lhs_array=lhs_array, idx=0)

    print(lhs_array.shape)
    print(parameters)
    print(lhs_array[:5,:])

    #corner.corner(
    #    lhs_array,
    #    labels=all_names,
    #    figsize=(10,10)
    #    )

    #plt.show()