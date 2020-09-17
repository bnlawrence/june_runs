import sys

# sys.path.insert(0, '../../runner/parameter_generator.py')
sys.path.insert(0, "../../runner")

import csv
import parameter_generator as pg


def test__from_list():

    parameter_list = [
        {"beta_pub": 0, "beta_household": 0, "beta_care_home": 0.2},
        {"beta_pub": 1, "beta_household": 2, "beta_care_home": 0.5},
        {"beta_pub": 2, "beta_household": 3, "beta_care_home": 0.9},
        {"beta_pub": 3, "beta_household": 1, "beta_care_home": 0.2},
    ]

    parameter_generator = pg.ParameterGenerator(parameter_list)

    assert parameter_generator[0] == parameter_list[0]
    assert parameter_generator[1] == parameter_list[1]
    assert parameter_generator[2] == parameter_list[2]

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
    with open("parameters.csv", "w", newline="") as f:
        dict_writer = csv.DictWriter(f, parameter_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(parameter_list)

    parameter_generator = pg.ParameterGenerator.from_file(
        path_to_parameters="parameters.csv", parameters_to_run="all"
    )

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
    assert parameter_generator[0] == {"beta_pub": 0, "beta_household": 3}
    assert parameter_generator[1] == {"beta_pub": 0, "beta_household": 4}
    assert parameter_generator[2] == {"beta_pub": 0, "beta_household": 5}
    assert parameter_generator[3] == {"beta_pub": 1, "beta_household": 3}
    assert parameter_generator[4] == {"beta_pub": 1, "beta_household": 4}
    assert parameter_generator[5] == {"beta_pub": 1, "beta_household": 5}
    assert parameter_generator[6] == {"beta_pub": 2, "beta_household": 3}
    assert parameter_generator[7] == {"beta_pub": 2, "beta_household": 4}
    assert parameter_generator[8] == {"beta_pub": 2, "beta_household": 5}


def test__from_regular_grid():
    parameter_dict = {"beta_pub": [0, 1, 3], "beta_household": [0, 1, 3]}
    parameter_generator = pg.ParameterGenerator.from_regular_grid(
        parameter_dict=parameter_dict
    )
    assert len(parameter_generator.parameter_list) == 9
    assert parameter_generator[0] == {"beta_pub": 0.0, "beta_household": 0.0}
    assert parameter_generator[1] == {"beta_pub": 0.0, "beta_household": 0.5}
    assert parameter_generator[2] == {"beta_pub": 0.0, "beta_household": 1.0}
    assert parameter_generator[3] == {"beta_pub": 0.5, "beta_household": 0.0}
    assert parameter_generator[4] == {"beta_pub": 0.5, "beta_household": 0.5}
    assert parameter_generator[5] == {"beta_pub": 0.5, "beta_household": 1.0}

def test__from_lhs():
    parameter_bounds_dict = {"beta_pub": [0., 1.], "beta_household": [100., 200.]}
    parameter_generator = pg.ParameterGenerator.from_latin_hypercube(
       parameter_bounds=parameter_bounds_dict, n_samples= 10
    )
    assert len(parameter_generator.parameter_list) == 10 

    for params in parameter_generator.parameter_list:
        assert 0. < params['beta_pub'] < 1.
        assert 100. < params['beta_household'] < 200.


