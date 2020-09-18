import sys

# sys.path.insert(0, '../../runner/parameter_generator.py')
sys.path.insert(0, "../../runner")

import pandas as pd
import parameter_generator as pg


def test__from_list():

    parameter_list = [
        {"beta_pub": 0, "beta_household": 0, "beta_care_home": 0.2},
        {"beta_pub": 1, "beta_household": 2, "beta_care_home": 0.5},
        {"beta_pub": 2, "beta_household": 3, "beta_care_home": 0.9},
        {"beta_pub": 3, "beta_household": 1, "beta_care_home": 0.2},
    ]

    parameter_generator = pg.ParameterGenerator(parameter_list)
    for i, parameters in enumerate(parameter_list):
        parameters['run_number'] = i
    assert parameter_generator[0] == parameter_list[0]
    assert parameter_generator[1] == parameter_list[1]
    assert parameter_generator[2] == parameter_list[2]

    assert parameter_generator[0]['run_number'] == 0

    parameter_generator = pg.ParameterGenerator(
        parameter_list, parameters_to_run=[2, 3]
    )

    assert parameter_generator[0] == parameter_list[2]
    assert parameter_generator[1] == parameter_list[3]


def test__from_file():
    parameter_list = [
        {"beta_pub": 0, "beta_household": 0, "beta_care_home": 0.2},
        {"beta_pub": 1, "beta_household": 2, "beta_care_home": 0.5},
        {"beta_pub": 2, "beta_household": 3, "beta_care_home": 0.9},
        {"beta_pub": 3, "beta_household": 1, "beta_care_home": 0.2},
    ]
    df = pd.DataFrame(parameter_list)
    df.to_csv('parameters.csv', sep=' ', index_label=False)

    parameter_generator = pg.ParameterGenerator.from_file(
        path_to_parameters="parameters.csv", parameters_to_run="all"
    )
    for i, parameters in enumerate(parameter_list):
        parameters['run_number'] = i

    assert parameter_generator[0] == parameter_list[0]
    assert parameter_generator[1] == parameter_list[1]
    assert parameter_generator[2] == parameter_list[2]

    parameter_generator = pg.ParameterGenerator.from_file(
        path_to_parameters="parameters.csv", parameters_to_run=[2, 3]
    )

    assert parameter_generator[0] == parameter_list[2]
    assert parameter_generator[1] == parameter_list[3]


def test__from_grid():
    parameter_dict = {"beta_pub": [0, 1, 2], "beta_household": [3, 4, 5]}
    parameter_generator = pg.ParameterGenerator.from_grid(parameter_dict=parameter_dict)
    assert len(parameter_generator.parameter_list) == 9
    assert parameter_generator[0] == {"beta_pub": 0, "beta_household": 3, "run_number":0}
    assert parameter_generator[1] == {"beta_pub": 0, "beta_household": 4, "run_number":1}
    assert parameter_generator[2] == {"beta_pub": 0, "beta_household": 5, "run_number":2}
    assert parameter_generator[3] == {"beta_pub": 1, "beta_household": 3, "run_number":3}
    assert parameter_generator[4] == {"beta_pub": 1, "beta_household": 4, "run_number":4}
    assert parameter_generator[5] == {"beta_pub": 1, "beta_household": 5, "run_number":5}
    assert parameter_generator[6] == {"beta_pub": 2, "beta_household": 3, "run_number":6}
    assert parameter_generator[7] == {"beta_pub": 2, "beta_household": 4, "run_number":7}
    assert parameter_generator[8] == {"beta_pub": 2, "beta_household": 5, "run_number":8}


def test__from_regular_grid():
    parameter_dict = {"beta_pub": [0, 1, 3], "beta_household": [0, 1, 3]}
    parameter_generator = pg.ParameterGenerator.from_regular_grid(
        parameter_dict=parameter_dict
    )
    assert len(parameter_generator.parameter_list) == 9
    assert parameter_generator[0] == {"beta_pub": 0.0, "beta_household": 0.0, "run_number":0}
    assert parameter_generator[1] == {"beta_pub": 0.0, "beta_household": 0.5, "run_number":1}
    assert parameter_generator[2] == {"beta_pub": 0.0, "beta_household": 1.0, "run_number":2}
    assert parameter_generator[3] == {"beta_pub": 0.5, "beta_household": 0.0, "run_number":3}
    assert parameter_generator[4] == {"beta_pub": 0.5, "beta_household": 0.5, "run_number":4}
    assert parameter_generator[5] == {"beta_pub": 0.5, "beta_household": 1.0,"run_number":5}

def test__from_lhs():
    parameter_bounds_dict = {"beta_pub": [0., 1.], "beta_household": [100., 200.]}
    parameter_generator = pg.ParameterGenerator.from_latin_hypercube(
       parameter_bounds=parameter_bounds_dict, n_samples= 10
    )
    assert len(parameter_generator.parameter_list) == 10 

    for params in parameter_generator.parameter_list:
        assert 0. < params['beta_pub'] < 1.
        assert 100. < params['beta_household'] < 200.

def test__fix_parameters():
    parameter_list = [
        {"beta_pub": 0, "beta_household": 0, "beta_care_home": 0.2},
        {"beta_pub": 1, "beta_household": 2, "beta_care_home": 0.5},
        {"beta_pub": 2, "beta_household": 3, "beta_care_home": 0.9},
        {"beta_pub": 3, "beta_household": 1, "beta_care_home": 0.2},
    ]
    parameters_to_fix = {'beta_grocery': 10, 'quarantine_household_compliance': 0.}

    parameter_generator = pg.ParameterGenerator(parameter_list,
            parameters_to_fix=parameters_to_fix
    )
    for i, parameters in enumerate(parameter_list):
        parameters['run_number'] = i
        parameters['beta_grocery'] = 10
        parameters['quarantine_household_compliance'] = 0.
    assert parameter_generator[0] == parameter_list[0]
    assert parameter_generator[1] == parameter_list[1]
    assert parameter_generator[2] == parameter_list[2]

    assert parameter_generator[0]['run_number'] == 0


