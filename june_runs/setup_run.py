import argparse
import yaml
import json
from copy import deepcopy
from pathlib import Path

from june_runs.utils import parse_paths
from june_runs import ParameterGenerator


class RunSetup:
    """
    General class to handle setting runs.
    """

    def __init__(self, paths_configuration, parameter_configuration):
        self.paths = parse_paths(paths_configuration)
        self.parameters = parameter_configuration
        self.parameter_generator = self._init_parameter_generator()

    def _init_parameter_generator(self):
        sampling_type = self.parameters.get("sampling_type", None)
        parameters_to_run = self.parameters.get("parameters_to_run", "all")
        parameters = deepcopy(self.parameters["parameters"])
        if sampling_type == "file":
            parameter_generator = ParameterGenerator.from_file(
                parameters["parameter_file"], parameters_to_run=parameters_to_run
            )
        elif sampling_type == "latin_hypercube":
            n_samples = parameters.pop("n_samples")
            parameter_generator = ParameterGenerator.from_latin_hypercube(
                parameter_dict=parameters,
                n_samples=n_samples,
                parameters_to_run=parameters_to_run,
            )
        elif sampling_type == "grid":
            parameter_generator = ParameterGenerator.from_grid(
                parameter_dict=parameters, parameters_to_run=parameters_to_run
            )
        elif sampling_type == "regular_grid":
            parameter_generator = ParameterGenerator.from_regular_grid(
                parameter_dict=parameters, parameters_to_run=parameters_to_run
            )
        else:
            raise NotImplementedError
        return parameter_generator

    def generate_parameters(self):
        for i, parameter in enumerate(self.parameter_generator):
            save_path = self.paths["runs_path"] / f"run_{i:03d}"
            save_path.mkdir(exist_ok=True, parents=True)
            with open(save_path / "parameters.json", "w") as f:
                json.dump(parameter, f, indent=4, sort_keys=True, default=str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tool to setup JUNE runs.")

    parser.add_argument(
        "-c", "--config", help="Path to run config.", required=True,
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run_setup = RunSetup(
        parameter_configuration=config["parameter_configuration"],
        paths_configuration=config["paths_configuration"],
    )
    run_setup.generate_parameters()
