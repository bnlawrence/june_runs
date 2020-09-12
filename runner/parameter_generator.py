import numpy as np
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

default_config_file = Path(__file__).parent.parent / "run_configs/config_example.yaml"

def verbose_print(*args,verbose=False):
    if verbose:
        print(*args)

def _read_parameters_to_run(parameters_to_run, num_runs):
    if parameters_to_run is None:
        parameters_to_run = "all"
    if type(parameters_to_run) == str:
        if parameters_to_run == "all":
            parameters_to_run = np.arange(0, num_runs)
        else:
            low, high = list(map(int, parameters_to_run.split("-")))
            parameters_to_run = np.arange(low, high+1)
    return parameters_to_run

def _get_len_parameter_grid(parameter_configuration):
    len_grid = len(
        ParameterGenerator._do_grid(
            parameter_configuration["parameters_to_vary"]
        )
    )
    return len_grid


class ParameterGenerator:
    """
    Given a parameter configuration with parameter bounds, generates a 
    latin hypercube sampler that returns random parameter variations.
    """

    config_types = ["latin_hypercube", "grid"]

    def __init__(self, parameter_configuration: dict = None, verbose=False):
        self.parameter_dict = self._read_parameter_configuration(
            parameter_configuration["parameters_to_vary"]
        )
        if parameter_configuration.get("fixed_parameters") is not None:
            if type(parameter_configuration["fixed_parameters"]) is dict:
                self.fixed_paramters = _read_parameter_configuration(
                    parameter_configuration["fixed_parameters"]
                )
            elif type(parameter_configuration["fixed_parameters"]) is str:
                with open(parameter_configuration["fixed_parameters"]) as json_file:
                    self.fixed_parameters = json.load(json_file)
        else:
            self.fixed_parameters = dict()

        if parameter_configuration.get("config_type") in [None, "default"]:
            self.config_type = "latin_hypercube"
        elif parameter_configuration.get("config_type") in self.config_types: 
            self.config_type = parameter_configuration["config_type"]
        else:
            print("Available config_types:", config_types)
        verbose_print(f"set config type to {self.config_type}", verbose=True)
        
        if self.config_type == "latin_hypercube":
            # Get num samples first -- as LatHyp depends on num_samples.
            self.num_samples = parameter_configuration["number_of_samples"]
            self.parameter_array = self._generate_lhs_array()
        elif self.config_type == "grid":
            # Get num samples last -- n_samples determined by the grid permutations.
            self.parameter_array = self._generate_grid_array()
            self.num_samples = len(self.parameter_array)      

        self.parameters_to_run = _read_parameters_to_run(
            parameter_configuration.get("parameters_to_run"), 
            self.num_samples
        )       
        # self.lhs_array = self._generate_lhs_array_from_config()

    @classmethod
    def from_file(cls, config_path: str = default_config_file):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(config["parameter_configuration"])

    @staticmethod # need to fix "_get_len_parameter_grid()" if remove @staticmethod
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

    def _generate_lhs_array(self, seed=1):
        """
        Generates a latin hypercube array.
        """
        bounds = list(self.parameter_dict.values())
        num_vars = len(bounds)
        lhs_array = lhs(
            n=num_vars, samples=self.num_samples, criterion="maximin", random_state=seed
        )
        # scale to the bounds
        scale_samples(lhs_array, bounds)
        return lhs_array

    @staticmethod
    def _do_grid(parameters_dict):
        keys, values = zip(*parameters_dict.items())
        permutations_dicts = [
            dict(zip(keys, v)) for v in itertools.product(*values)
        ]
        return permutations_dicts

    def _generate_grid_array(self, ):  
        return self._do_grid(self.parameter_dict)

    def get_parameters_from_index(self, idx):
        """Generates a parameter dictionary from latin hypercube array.
        idx is an integer that should be passed from each run,
        i.e. first run will have idx = 0, second run idx = 1...
        This will index out the row from the latin hypercube."""
        index_to_run = self.parameters_to_run[idx]
        parameter_values = self.parameter_array[index_to_run]
        ret = {}
        for i, (parameter_name,fixed_val) in enumerate(self.fixed_parameters.items()):
            ret[parameter_name] = fixed_val
        for i, parameter_name in enumerate(self.parameter_dict.keys()):
            if self.config_type == "latin_hypercube":
                ret[parameter_name] = parameter_values[i]
            elif self.config_type == "grid":
                ret[parameter_name] = parameter_values[parameter_name]
        ret["run_number"] = int(index_to_run)
        return ret

    def __getitem__(self, idx):
        return self.get_parameters_from_index(idx)

    def __iter__(self):
        return iter([self[idx] for idx in self.parameters_to_run])

