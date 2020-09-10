import numpy as np
import os
import sys
import yaml
from pyDOE2 import lhs
from SALib.util import scale_samples
from pathlib import Path
from collections import OrderedDict, defaultdict

default_config_file = Path(__file__).parent.parent / "run_configs/config_example.yaml"

def _read_parameters_to_run(parameters_to_run, num_runs):
    if type(parameters_to_run) == str:
        if parameters_to_run == "all":
            parameters_to_run = np.arange(0, num_runs)
        else:
            low, high = list(map(int, parameters_to_run.split("-")))
            parameters_to_run = np.arange(low, high+1)
    return parameters_to_run


class FixedParameterGenerator:
    def __init__(self, parameter_configuration: dict = None):
        self.num_samples = parameter_configuration["number_of_samples"]
        self.parameters_to_run = parameter_configuration['fixed_parameters']

    def get_parameters_from_index(self, idx):
        ret = {}
        for parameter_name, parameter_values in self.parameters_to_run.items():
            ret[parameter_name] = parameter_values[idx]
        ret["run_number"] = int(idx)
        return ret

    def __getitem__(self, idx):
        return self.get_parameters_from_index(idx)


class ParameterGenerator:
    """
    Given a parameter configuration with parameter bounds, generates a 
    latin hypercube sampler that returns random parameter variations.
    """

    def __init__(self, parameter_configuration: dict = None):
        self.parameter_dict = self._read_parameter_configuration(
            parameter_configuration
        )
        self.num_samples = parameter_configuration["number_of_samples"]
        self.parameters_to_run = _read_parameters_to_run(parameter_configuration["parameters_to_run"], self.num_samples)
        self.lhs_array = self._generate_lhs_array()
        # self.lhs_array = self._generate_lhs_array_from_config()

    @classmethod
    def from_file(cls, config_path: str = default_config_file):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(config["parameter_configuration"])

    def _read_parameter_configuration(self, parameter_configuration):
        parameter_dict = OrderedDict()
        for parameter_type, parameter_values in parameter_configuration[
            "parameters_to_vary"
        ].items():
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

    def get_parameters_from_index(self, idx):
        """Generates a parameter dictionary from latin hypercube array.
        idx is an integer that should be passed from each run,
        i.e. first run will have idx = 0, second run idx = 1...
        This will index out the row from the latin hypercube."""
        index_to_run = self.parameters_to_run[idx]
        parameter_values = self.lhs_array[index_to_run]
        ret = {}
        for i, parameter_name in enumerate(self.parameter_dict.keys()):
            ret[parameter_name] = parameter_values[i]
        ret["run_number"] = int(index_to_run)
        return ret

    def __getitem__(self, idx):
        return self.get_parameters_from_index(idx)

    def __iter__(self):
        return iter([self[idx] for idx in self.parameters_to_run])
