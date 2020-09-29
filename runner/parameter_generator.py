import numpy as np
import csv
import os
import sys
import yaml
import json

import pandas as pd
from pyDOE2 import lhs
from SALib.util import scale_samples
from pathlib import Path
from collections import OrderedDict, defaultdict, Counter
from sklearn.model_selection import ParameterGrid
import itertools

from typing import List, Optional

default_config_file = Path(__file__).parent.parent / "run_configs/config_example.yaml"


class ParameterGenerator:
    def __init__(
        self,
        parameter_list: List[dict],
        parameters_to_fix: Optional[dict] = None,
        parameters_to_run: List[int] = "all",
    ):
        self._check_sanity(parameter_list, parameters_to_fix)
        self.parameter_list = parameter_list
        for i, parameters in enumerate(self.parameter_list):
            parameters["run_number"] = i
            if parameters_to_fix is not None:
                for key, value in parameters_to_fix.items():
                    parameters[key] = value
        self.parameters_to_run = self._read_parameters_to_run(parameters_to_run)

    @classmethod
    def from_file(cls, path_to_parameters: str, parameters_to_fix: Optional[dict] = None, parameters_to_run="all"):
        parameter_list = pd.read_csv(path_to_parameters).to_dict("records")
        return cls(parameter_list=parameter_list, parameters_to_fix=parameters_to_fix, parameters_to_run=parameters_to_run)

    @classmethod
    def from_grid(cls, parameter_dict: List[dict], parameters_to_fix: Optional[dict] = None, parameters_to_run="all"):
        keys, values = zip(*parameter_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return cls(
            parameter_list=permutations_dicts, parameters_to_fix = parameters_to_fix, parameters_to_run=parameters_to_run
        )

    @classmethod
    def from_regular_grid(cls, parameter_dict: List[dict], parameters_to_fix: Optional[dict] = None, parameters_to_run="all"):
        for key, value in parameter_dict.items():
            parameter_dict[key] = np.linspace(
                start=value[0], stop=value[1], num=value[2]
            )
        return cls.from_grid(
            parameter_dict=parameter_dict, parameters_to_fix=parameters_to_fix, parameters_to_run=parameters_to_run
        )

    @classmethod
    def from_latin_hypercube(cls, parameter_bounds, n_samples, parameters_to_fix: Optional[dict] = None, parameters_to_run="all"):
        return cls(
            parameter_list=cls._generate_lhs(
                cls, parameter_bounds=parameter_bounds, n_samples=n_samples
            ),
            parameters_to_fix = parameters_to_fix,
            parameters_to_run=parameters_to_run,
        )

    def _check_sanity(self, parameters_to_vary, parameters_to_fix):
        # Check that in list of parameters all have the same keys
        counter_list = []
        for parameter in parameters_to_vary:
            # check a key is not repeated within same row
            counter = Counter(list(parameter.keys()))
            try:
                for key, value in counter.items():
                    assert value == 1
            except:
                raise ValueError("There are rows with repeated parameters!")
            counter_list.append(counter)
        try:
            unique_counter = [
                dict(s) for s in set(frozenset(c.items()) for c in counter_list)
            ]
            assert len(unique_counter) == 1
        except:
            raise ValueError(
                "some of the parameter configurations dont have the same parameters"
            )

        # Check that parameters_to_fix are not included in parameters_to_vary otherwise error
        try:
            if parameters_to_fix is not None:
                print(f"Parameters being fixed = ", parameters_to_fix.keys())
                assert (
                    set(parameters_to_fix.keys()) & set(parameters_to_vary[0].keys())
                    == set()
                )
            else:
                print(f"No fixed parameters")
        except:
            raise ValueError(
                "Check what you are doing! You are varying some of the parameters that you are fixing and I can tell you that it is a bad idea"
            )
        # Print what parameters are being varied and what parameters are fixed
        print(f"Paramters being varied = ", parameters_to_vary[0].keys())

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
