import numpy as np
import csv
import os
import sys
import yaml
import json

from pyDOE2 import lhs
from SALib.util import scale_samples
from pathlib import Path
from collections import OrderedDict, defaultdict
from sklearn.model_selection import ParameterGrid
import itertools

from typing import List, Optional

default_config_file = Path(__file__).parent.parent / "run_configs/config_example.yaml"


class ParameterGenerator:
    def __init__(
        self, parameter_list: List[dict], parameters_to_run: List[int] = "all"
    ):
        self.parameter_list = parameter_list
        self.parameters_to_run = self._read_parameters_to_run(parameters_to_run)

    @classmethod
    def from_file(cls, path_to_parameters: str, parameters_to_run="all"):
        with open(path_to_parameters) as f:
            parameter_list = [
                {key: float(value) for key, value in row.items()}
                for row in csv.DictReader(f)
            ]
        return cls(parameter_list=parameter_list, parameters_to_run=parameters_to_run)

    @classmethod
    def from_grid(cls, parameter_dict: List[dict], parameters_to_run="all"):
        keys, values = zip(*parameter_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return cls(
            parameter_list=permutations_dicts, parameters_to_run=parameters_to_run
        )

    @classmethod
    def from_regular_grid(cls, parameter_dict: List[dict], parameters_to_run="all"):
        for key, value in parameter_dict.items():
            parameter_dict[key] = np.linspace(
                start=value[0], stop=value[1], num=value[2]
            )
        return cls.from_grid(
            parameter_dict=parameter_dict, parameters_to_run=parameters_to_run
        )

    @classmethod
    def from_latin_hypercube(cls, parameter_bounds, n_samples, parameters_to_run="all"):
        return cls(
            parameter_list=cls._generate_lhs(
                cls, parameter_bounds=parameter_bounds, n_samples=n_samples
            ),
            parameters_to_run=parameters_to_run,
        )

    def _generate_lhs(self, parameter_bounds, n_samples, seed=1):
        """
        Generates a latin hypercube array.
        """
        bounds = list(parameter_bounds.values())
        num_vars = len(bounds)
        lhs_array = lhs(
            n=num_vars, samples=n_samples, criterion="maximin", random_state=seed
        )
        # scale to the bounds
        scale_samples(lhs_array, bounds)
        parameter_dicts = []
        for i in range(len(lhs_array)):
            parameter_dicts.append(
                {
                    key: value
                    for key, value in zip(parameter_bounds.keys(), lhs_array[i])
                }
            )
        return parameter_dicts

    @staticmethod  # need to fix "_get_len_parameter_grid()" if remove @staticmethod
    def _read_parameter_configuration(parameter_configuration):
        parameter_dict = OrderedDict()
        for parameter_type, parameter_values in parameter_configuration.items():
            if parameter_type == "betas":
                for beta_name, beta_range in parameter_values.items():
                    parameter_dict["beta_" + beta_name] = beta_range
            elif parameter_type == "policies":
                for policy_name, policy_parameters in parameter_values.items():
                    for parameter_name, parameter_value in policy_parameters.items():
                        parameter_dict[
                            policy_name + "_" + parameter_name
                        ] = parameter_value
            else:
                parameter_dict[parameter_type] = parameter_values
        return parameter_dict

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
        return iter([self[idx] for idx in self.parameter_list])
