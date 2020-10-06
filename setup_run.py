import argparse
import yaml
import json
import random
import os
from copy import deepcopy
from pathlib import Path

from june_runs.utils import parse_paths
from june_runs import ParameterGenerator, ScriptMaker


class RunSetup:
    """
    General class to handle setting runs.
    """

    def __init__(self, run_configuration):
        self.run_configuration = run_configuration
        self.paths = parse_paths(run_configuration["paths_configuration"])
        self.parameters = run_configuration["parameter_configuration"]
        self.parameter_generator = self._init_parameter_generator()
        system_configuration = run_configuration["system_configuration"]
        self.script_maker = ScriptMaker(
            system=system_configuration["system_to_use"],
            run_directory=self.paths["runs_path"],
            job_name = self.paths["run_name"],
            memory_per_job = system_configuration["memory_per_job"],
            cpus_per_job = system_configuration["cpus_per_job"],
            number_of_jobs = len(self.parameter_generator),
            virtual_env_path = self.paths["virtual_env_path"]
        )

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

    def save_run_parameters(self):
        for i, parameter in enumerate(self.parameter_generator):
            ret = {}
            save_path = self.paths["runs_path"] / f"run_{i:03d}"
            random_seed = self.run_configuration.get("random_seed", "random")
            if random_seed == "random":
                random_seed = random.randint(0, 1_000_000_000)
            ret["run_number"] = i
            ret["purpose_of_the_run"] = self.run_configuration.get("purpose_of_the_run", "no comment")
            ret["random_seed"] = random_seed
            ret["parameters"] = parameter
            ret["paths"] = {
                "june_runs_path": self.paths["june_runs_path"],
                "save_path": str(save_path),
                "world_path": self.paths["world_path"],
                "summary_path": self.paths["summary_path"],
                "baseline_policy_path": self.paths["baseline_policy_path"],
                "baseline_interaction_path": self.paths["baseline_interaction_path"],
                "simulation_config_path": self.paths["simulation_config_path"],
            }
            save_path.mkdir(exist_ok=True, parents=True)
            with open(save_path / "parameters.json", "w") as f:
                json.dump(ret, f, indent=4, default=str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tool to setup JUNE runs.")

    parser.add_argument(
        "-c", "--config", help="Path to run config.", required=True,
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run_setup = RunSetup(
        run_configuration = config,
    )
    run_setup.save_run_parameters()
    run_setup.script_maker.write_scripts()
