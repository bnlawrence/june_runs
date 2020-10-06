import sys
import pytest
import json
import numpy as np

from pathlib import Path

import pandas as pd
from june_runs import ParameterGenerator

test_directory = Path(__file__).parent


def make_parameter_dict(
    beta_pub,
    beta_household,
    soft_lockdown_beta,
    hard_lockdown_beta,
    soft_compliance,
    hard_compliance,
):

    return {
        "interaction": {"betas": {"pub": beta_pub, "household": beta_household}},
        "policies": {
            "social_distancing": {
                "1": {
                    "start_time": "2020-03-16",
                    "beta_factors": {
                        "pub": soft_lockdown_beta,
                        "grocery": soft_lockdown_beta,
                        "cinema": soft_lockdown_beta,
                        "city_transport": soft_lockdown_beta,
                        "inter_city_transport": soft_lockdown_beta,
                        "hospital": soft_lockdown_beta,
                        "care_home": soft_lockdown_beta,
                        "company": soft_lockdown_beta,
                        "school": soft_lockdown_beta,
                        "household": 1.00,
                        "university": soft_lockdown_beta,
                    },
                },
                "2": {
                    "start_time": "2020-03-24",
                    "beta_factors": {
                        "pub": hard_lockdown_beta,
                        "grocery": hard_lockdown_beta,
                        "cinema": hard_lockdown_beta,
                        "city_transport": hard_lockdown_beta,
                        "inter_city_transport": hard_lockdown_beta,
                        "hospital": hard_lockdown_beta,
                        "care_home": hard_lockdown_beta,
                        "company": hard_lockdown_beta,
                        "school": hard_lockdown_beta,
                        "household": 1.00,
                        "university": hard_lockdown_beta,
                    },
                },
            },
            "quarantine": {
                "1": {
                    "start_time": "2020-03-16",
                    "compliance": soft_compliance,
                    "household_compliance": soft_compliance,
                },
                "2": {
                    "start_time": "2020-03-24",
                    "compliance": hard_compliance,
                    "household_compliance": hard_compliance,
                },
            },
        },
    }


@pytest.fixture(name="parameter_list")
def make_list():
    parameter_list = [
        {
            "interaction": {"betas": {"pub": 0, "household": 5}},
            "policies": {
                "lockdown": {
                    "soft_lockdown_date": "2020-03-16",
                    "hard_lockdown_date": "2020-03-24",
                    "lockdown_ratio": 0.5,
                    "hard_lockdown_policy_parameters": {
                        "social_distancing": {"overall_beta_factor": 0.6},
                        "quarantine": {"overall_compliance": 0.3},
                    },
                }
            },
        },
        {
            "interaction": {"betas": {"pub": 1, "household": 2}},
            "policies": {
                "lockdown": {
                    "soft_lockdown_date": "2020-03-16",
                    "hard_lockdown_date": "2020-03-24",
                    "lockdown_ratio": 1,
                    "hard_lockdown_policy_parameters": {
                        "social_distancing": {"overall_beta_factor": 0.5},
                        "quarantine": {"overall_compliance": 0.4},
                    },
                }
            },
        },
        {
            "interaction": {"betas": {"pub": 2, "household": 3}},
            "policies": {
                "lockdown": {
                    "soft_lockdown_date": "2020-03-16",
                    "hard_lockdown_date": "2020-03-24",
                    "lockdown_ratio": 0,
                    "hard_lockdown_policy_parameters": {
                        "social_distancing": {"overall_beta_factor": 0.5},
                        "quarantine": {"overall_compliance": 0.4},
                    },
                }
            },
        },
    ]
    return parameter_list


def test__from_list(parameter_list):
    parameter_generator = ParameterGenerator(parameter_list)
    expected_0 = make_parameter_dict(
        beta_pub=0,
        beta_household=5,
        soft_lockdown_beta=0.8,
        hard_lockdown_beta=0.6,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[0] == expected_0
    expected_1 = make_parameter_dict(
        beta_pub=1,
        beta_household=2,
        soft_lockdown_beta=0.5,
        hard_lockdown_beta=0.5,
        soft_compliance=0.4,
        hard_compliance=0.4,
    )
    assert parameter_generator[1] == expected_1
    expected_2 = make_parameter_dict(
        beta_pub=2,
        beta_household=3,
        soft_lockdown_beta=1,
        hard_lockdown_beta=0.5,
        soft_compliance=0.0,
        hard_compliance=0.4,
    )
    assert parameter_generator[2] == expected_2
    parameter_generator = ParameterGenerator(parameter_list, parameters_to_run=[1, 2])
    # this works cause we override parameter_list inside the generator.
    assert parameter_generator[0] == expected_1
    assert parameter_generator[1] == expected_2

    # check from file
    with open(test_directory / "parameters.json", "w") as f:
        json.dump(parameter_list, f)

    parameter_generator = ParameterGenerator.from_file(
        path_to_parameters=test_directory / "parameters.json", parameters_to_run="all"
    )
    assert parameter_generator[0] == expected_0
    assert parameter_generator[1] == expected_1
    assert parameter_generator[2] == expected_2

    parameter_generator = ParameterGenerator.from_file(
        path_to_parameters=test_directory / "parameters.json", parameters_to_run=[1, 2]
    )

    assert parameter_generator[0] == expected_1
    assert parameter_generator[1] == expected_2


def test__from_grid():
    parameter_dict = {
        "interaction": {"betas": {"pub": [0, 1], "household": 5}},
        "policies": {
            "lockdown": {
                "soft_lockdown_date": "2020-03-16",
                "hard_lockdown_date": "2020-03-24",
                "lockdown_ratio": 0.5,
                "hard_lockdown_policy_parameters": {
                    "social_distancing": {"overall_beta_factor": [0.3, 0.9]},
                    "quarantine": {"overall_compliance": [0.3, 0.9]},
                },
            }
        },
    }
    parameter_generator = ParameterGenerator.from_grid(parameter_dict=parameter_dict)
    assert len(parameter_generator.parameter_list) == 8
    assert parameter_generator[0] == make_parameter_dict(
        beta_pub=0,
        beta_household=5,
        soft_lockdown_beta=0.65,
        hard_lockdown_beta=0.3,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[1] == make_parameter_dict(
        beta_pub=0,
        beta_household=5,
        soft_lockdown_beta=0.65,
        hard_lockdown_beta=0.3,
        soft_compliance=0.45,
        hard_compliance=0.9,
    )
    assert parameter_generator[2] == make_parameter_dict(
        beta_pub=0,
        beta_household=5,
        soft_lockdown_beta=0.95,
        hard_lockdown_beta=0.9,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[3] == make_parameter_dict(
        beta_pub=0,
        beta_household=5,
        soft_lockdown_beta=0.95,
        hard_lockdown_beta=0.9,
        soft_compliance=0.45,
        hard_compliance=0.9,
    )
    assert parameter_generator[4] == make_parameter_dict(
        beta_pub=1,
        beta_household=5,
        soft_lockdown_beta=0.65,
        hard_lockdown_beta=0.3,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[5] == make_parameter_dict(
        beta_pub=1,
        beta_household=5,
        soft_lockdown_beta=0.65,
        hard_lockdown_beta=0.3,
        soft_compliance=0.45,
        hard_compliance=0.9,
    )
    assert parameter_generator[6] == make_parameter_dict(
        beta_pub=1,
        beta_household=5,
        soft_lockdown_beta=0.95,
        hard_lockdown_beta=0.9,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[7] == make_parameter_dict(
        beta_pub=1,
        beta_household=5,
        soft_lockdown_beta=0.95,
        hard_lockdown_beta=0.9,
        soft_compliance=0.45,
        hard_compliance=0.9,
    )


def test__from_regular_grid():
    parameter_dict = {
        "interaction": {"betas": {"pub": [0, 1, 3], "household": 5}},
        "policies": {
            "lockdown": {
                "soft_lockdown_date": "2020-03-16",
                "hard_lockdown_date": "2020-03-24",
                "lockdown_ratio": 0.5,
                "hard_lockdown_policy_parameters": {
                    "social_distancing": {"overall_beta_factor": [0.3, 0.9, 2]},
                    "quarantine": {"overall_compliance": 0.3},
                },
            }
        },
    }
    parameter_generator = ParameterGenerator.from_regular_grid(
        parameter_dict=parameter_dict
    )
    assert len(parameter_generator.parameter_list) == 6
    assert parameter_generator[0] == make_parameter_dict(
        beta_pub=0,
        beta_household=5,
        soft_lockdown_beta=0.65,
        hard_lockdown_beta=0.3,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[1] == make_parameter_dict(
        beta_pub=0,
        beta_household=5,
        soft_lockdown_beta=0.95,
        hard_lockdown_beta=0.9,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[2] == make_parameter_dict(
        beta_pub=0.5,
        beta_household=5,
        soft_lockdown_beta=0.65,
        hard_lockdown_beta=0.3,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[3] == make_parameter_dict(
        beta_pub=0.5,
        beta_household=5,
        soft_lockdown_beta=0.95,
        hard_lockdown_beta=0.9,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[4] == make_parameter_dict(
        beta_pub=1.0,
        beta_household=5,
        soft_lockdown_beta=0.65,
        hard_lockdown_beta=0.3,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )
    assert parameter_generator[5] == make_parameter_dict(
        beta_pub=1.0,
        beta_household=5,
        soft_lockdown_beta=0.95,
        hard_lockdown_beta=0.9,
        soft_compliance=0.15,
        hard_compliance=0.3,
    )


def test__from_lhs():
    parameter_bounds_dict = {
        "interaction": {"betas": {"pub": [0, 1], "household": 5}},
        "policies": {
            "lockdown": {
                "soft_lockdown_date": "2020-03-16",
                "hard_lockdown_date": "2020-03-24",
                "lockdown_ratio": 0.5,
                "hard_lockdown_policy_parameters": {
                    "social_distancing": {"overall_beta_factor": [0.3, 0.9]},
                    "quarantine": {"overall_compliance": [0.4, 0.6]},
                },
            }
        },
    }
    parameter_generator = ParameterGenerator.from_latin_hypercube(
        parameter_dict=parameter_bounds_dict, n_samples=100
    )
    assert len(parameter_generator.parameter_list) == 100
    for params in parameter_generator.parameter_list:
        assert 0.0 < params["interaction"]["betas"]["pub"] < 1.0
        assert params["interaction"]["betas"]["household"] == 5
        for beta, value in params["policies"]["social_distancing"]["1"][
            "beta_factors"
        ].items():
            if beta == "household":
                assert value == 1
                assert (
                    params["policies"]["social_distancing"]["2"]["beta_factors"][beta]
                    == 1
                )
            else:
                assert 0.65 < value < 0.95
                assert np.isclose(
                    params["policies"]["social_distancing"]["2"]["beta_factors"][beta],
                    1 + (value - 1) / 0.5,
                    rtol=0.01,
                )

        comp1 = params["policies"]["quarantine"]["1"]["compliance"]
        comp2 = params["policies"]["quarantine"]["1"]["household_compliance"]
        assert 0.2 < comp1 < 0.3
        assert 0.2 < comp2 < 0.3
        assert params["policies"]["quarantine"]["2"]["compliance"] == comp1 * 2
        assert (
            params["policies"]["quarantine"]["2"]["household_compliance"] == comp2 * 2
        )


# TODO not available yet.
# def test__fix_parameters():
#    parameter_list = [
#        {"beta_pub": 0, "beta_household": 0, "beta_care_home": 0.2},
#        {"beta_pub": 1, "beta_household": 2, "beta_care_home": 0.5},
#        {"beta_pub": 2, "beta_household": 3, "beta_care_home": 0.9},
#        {"beta_pub": 3, "beta_household": 1, "beta_care_home": 0.2},
#    ]
#    parameters_to_fix = {"beta_grocery": 10, "quarantine_household_compliance": 0.0}
#
#    parameter_generator = ParameterGenerator(
#        parameter_list, parameters_to_fix=parameters_to_fix
#    )
#    for i, parameters in enumerate(parameter_list):
#        parameters["run_number"] = i
#        parameters["beta_grocery"] = 10
#        parameters["quarantine_household_compliance"] = 0.0
#    assert parameter_generator[0] == parameter_list[0]
#    assert parameter_generator[1] == parameter_list[1]
#    assert parameter_generator[2] == parameter_list[2]
#
#    assert parameter_generator[0]["run_number"] == 0
