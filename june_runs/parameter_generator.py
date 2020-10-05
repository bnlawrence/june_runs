import numpy as np
import csv
import os
import sys
import yaml
import json
from copy import deepcopy
from datetime import datetime

import pandas as pd
from pyDOE2 import lhs
from SALib.util import scale_samples
from pathlib import Path
from collections import OrderedDict, defaultdict, Counter
from sklearn.model_selection import ParameterGrid
import itertools

from typing import List, Optional

default_config_file = Path(__file__).parent.parent / "run_configs/config_example.yaml"

all_groups = [
    "pub",
    "grocery",
    "cinema",
    "city_transport",
    "inter_city_transport",
    "hospital",
    "care_home",
    "company",
    "school",
    "household",
    "university",
]


def iter_paths(d):
    def iter1(d, path):
        paths = []
        for k, v in d.items():
            if isinstance(v, dict):
                paths += iter1(v, path + [k])
            paths.append((path + [k], v))
        return paths

    return iter1(d, [])


def set_value_in_path(d, path, value):
    i = 0
    while i < len(path) - 1:
        if path[i] not in d:
            d[path[i]] = {}
        d = d[path[i]]
        i += 1
    d[path[-1]] = value


class ParameterGenerator:
    def __init__(
        self, parameter_list: List[dict], parameters_to_run: List[int] = "all",
    ):
        self.parameter_list = self._read_parameter_list(parameter_list)
        self.parameters_to_run = self._read_parameters_to_run(parameters_to_run)

    def _read_parameter_list(self, parameter_list):
        """
        Reads the parameter list. If there is any file marked as lockdown
        then we build a config for soft and hard lockdown using the parameters specified.
        See the runner tests for examples.
        """
        for parameters in parameter_list:
            if "policies" in parameters:
                for policy in parameters["policies"]:
                    if "lockdown" in policy:
                        ret = deepcopy(parameters["policies"])
                        lockdown_config = ret.pop("lockdown")
                        lockdown_policies = build_policy_config_for_lockdown(
                            **lockdown_config
                        )
                        parameters["policies"] = {**ret, **lockdown_policies}
        return parameter_list

    @classmethod
    def from_file(
        cls, path_to_parameters: str, parameters_to_run="all",
    ):
        with open(path_to_parameters, "r") as f:
            parameter_list = json.load(f)
        return cls(parameter_list=parameter_list, parameters_to_run=parameters_to_run,)

    @classmethod
    def from_grid(
        cls, parameter_dict: dict, parameters_to_run="all",
    ):
        paths = []
        value_ranges = []
        fixed_parameters = {}
        for path, value in iter_paths(parameter_dict):
            if type(value) == dict:
                continue
            elif type(value) == list:
                paths.append(path)
                value_ranges.append(value)
            else:
                set_value_in_path(fixed_parameters, path, value)
        parameter_list = []
        permutations = itertools.product(*value_ranges)
        for permutation in permutations:
            ret = deepcopy(fixed_parameters)
            for path, value in zip(paths, permutation):
                set_value_in_path(ret, path=path, value=value)
            parameter_list.append(ret)
        return cls(parameter_list=parameter_list, parameters_to_run=parameters_to_run,)

    @classmethod
    def from_regular_grid(
        cls, parameter_dict: dict, parameters_to_run="all",
    ):
        ret = {}
        for path, value in iter_paths(parameter_dict):
            if type(value) == dict:
                continue
            elif type(value) == list:
                if len(value) != 3:
                    raise ValueError(f"For regular grid lists must have size 3")
                set_value_in_path(ret, path, list(np.linspace(*value)))
            else:
                set_value_in_path(ret, path, value)
        return cls.from_grid(parameter_dict=ret, parameters_to_run=parameters_to_run,)

    @classmethod
    def from_latin_hypercube(
        cls, parameter_dict: dict, n_samples, parameters_to_run="all",
    ):
        paths = []
        value_ranges = []
        fixed_parameters = {}
        for path, value in iter_paths(parameter_dict):
            if type(value) == dict:
                continue
            elif type(value) == list:
                paths.append(path)
                value_ranges.append(value)
            else:
                set_value_in_path(fixed_parameters, path, value)
        parameter_list = []
        sampled_parameters = cls._generate_lhs(
            parameter_bounds=value_ranges, n_samples=n_samples
        )
        for parameter_values in sampled_parameters:
            ret = deepcopy(fixed_parameters)
            for path, value in zip(paths, parameter_values):
                set_value_in_path(ret, path=path, value=value)
            parameter_list.append(ret)
        return cls(parameter_list=parameter_list, parameters_to_run=parameters_to_run,)

    @classmethod
    def _generate_lhs(cls, parameter_bounds, n_samples, seed=1):
        """
        Generates a latin hypercube array.
        """
        num_vars = len(parameter_bounds)
        lhs_array = lhs(
            n=num_vars, samples=n_samples, criterion="maximin", random_state=seed
        )
        # scale to the bounds
        scale_samples(lhs_array, parameter_bounds)
        return lhs_array

    def _read_parameters_to_run(self, parameters_to_run):
        if parameters_to_run is None:
            parameters_to_run = "all"
        if type(parameters_to_run) == str:
            if parameters_to_run == "all":
                parameters_to_run = np.arange(0, len(self.parameter_list))
            else:
                low, high = list(map(int, parameters_to_run.split("-")))
                parameters_to_run = np.arange(low, high + 1)
        return parameters_to_run

    def get_parameters_from_index(self, idx):
        index_to_run = self.parameters_to_run[idx]
        return self.parameter_list[index_to_run]

    def __getitem__(self, idx):
        return self.get_parameters_from_index(idx)

    def __iter__(self):
        return iter([self[idx] for idx in range(len(self.parameters_to_run))])


def build_policy_config_for_lockdown(
    soft_lockdown_date: str,
    hard_lockdown_date: str,
    lockdown_ratio: int,
    hard_lockdown_policy_parameters: dict,
):
    policy_dict = {}
    for policy_type in hard_lockdown_policy_parameters:
        policy_dict[policy_type] = {
            "1": {"start_time": soft_lockdown_date},
            "2": {"start_time": hard_lockdown_date},
        }
        # soft first
        if policy_type == "social_distancing":
            overall_beta_factor = hard_lockdown_policy_parameters["social_distancing"][
                "overall_beta_factor"
            ]
            policy_dict[policy_type]["1"]["beta_factors"] = {}
            policy_dict[policy_type]["2"]["beta_factors"] = {}
            for group in all_groups:
                if group == "household":
                    policy_dict[policy_type]["1"]["beta_factors"][group] = 1.0
                    policy_dict[policy_type]["2"]["beta_factors"][group] = 1.0
                    continue
                policy_dict[policy_type]["1"]["beta_factors"][
                    group
                ] = 1 + lockdown_ratio * (overall_beta_factor - 1)
                policy_dict[policy_type]["2"]["beta_factors"][
                    group
                ] = overall_beta_factor
        elif policy_type == "quarantine":
            overall_compliance = hard_lockdown_policy_parameters["quarantine"][
                "overall_compliance"
            ]
            policy_dict[policy_type]["1"]["compliance"] = (
                lockdown_ratio * overall_compliance
            )
            policy_dict[policy_type]["1"]["household_compliance"] = (
                lockdown_ratio * overall_compliance
            )
            policy_dict[policy_type]["2"]["compliance"] = overall_compliance
            policy_dict[policy_type]["2"]["household_compliance"] = overall_compliance
    return policy_dict
