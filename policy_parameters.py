import numpy as np
import corner
import matplotlib.pyplot as plt


beta_names = [
#    'pub',
#    'grocery',
#    'cinema',
#    'commute_unit',
#    'commute_city_unit',
#    'hospital',
#    'care_home',
#    'company',
#    'school',
#    'household',
#    'university'
]

f = 1.

beta_bounds = { 
#    'pub'       : (f*0.01, f*0.25),
#    'grocery'   : (f*0.01, f*0.25),
#    'cinema'    : (f*0.01, f*0.25),
#    'commute_unit' : (f*0.01, f*0.5),
#    'commute_city_unit' : (f*0.01, f*0.5),
#    'hospital'  : (f*0.05, f*0.5),
#    'care_home' : (f*0.05, f*1.),
#    'company'   : (f*0.05, f*0.5),
#    'school'    : (f*0.05, f*0.5),
#    'household' : (f*0.05, f*0.5),
#    'university': (f*0.01, f*0.5),
}

assert set(beta_names) == set(beta_bounds.keys())

extra_names = [
#    'alpha_physical',
#    'asymptomatic_ratio',
#    'seed_strength',
#    'beta_factor_1',
    'x_factor',
    'beta_factor_2',
    'household_compliance_2'
    'shielding_compliance_2',
]

extra_bounds = {                
#    'alpha_physical' : (1.8,3.0),
#    'asymptomatic_ratio' : (0.05,0.4),
#    'seed_strength' : (0.5,1.3),
#    'beta_factor_1' : (0.3,0.9),
    'x_factor' : (0.3,0.8),
    'beta_factor_2' : (0.3,0.9),
    'household_compliance_2' : (0.2,0.8)
    'shielding_compliance_2' : (0.2,0.8)
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


def set_values(pol_dict,vals_dict):
    for k,v in vals_dict.items():
        if type(v) is dict:
            set_values(pol_dict[k],v)
        else:
            pol_dict[k] = v

def modify_policy(policies,policy,number=None,values=None):
    relevant_policies = []
    for p in policies:
        spec = p.get_spec()
        if spec == policy:
            relevant_policies.append(p)

    if len(relevant_policies) == 0:
        print(f'No policy {policy} in policies!')

    if number is None:
        if len(relevant_policies) > 1:
            raise ValueError(
            f'there are {len(relevant_policies)} instances of {policy} - provide number=1 (eg)')
        pol = relevant_policies[0]
    else:
        pol = relevant_policies[ number-1 ]  
    
    set_values(pol.__dict__,values)

    print(f'modified {policy} {number}: {pol.__dict__}')    


    return None

    
            


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
