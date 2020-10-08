import argparse
import yaml
import json
import random
import os
import sys
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
        self.parameter_generator = self.init_parameter_generator(
            self.parameters, paths=self.paths
        )
        system_configuration = run_configuration["system_configuration"]
        self.script_maker = self.init_script_maker(
            system_configuration,
            self.paths,
            number_of_jobs=len(self.parameter_generator),
        )

    @staticmethod
    def init_parameter_generator(parameter_configuration, paths=None):
        sampling_type = parameter_configuration.get("sampling_type", None)
        parameters_to_run = parameter_configuration.get("parameters_to_run", "all")
        parameters = deepcopy(parameter_configuration["parameters"])
        if sampling_type == "file":
            path_to_parameters = parameters.pop("path")
            if "@" in path_to_parameters:
                placeholder = path_to_parameters.split("/")[0].split("@")[-1]
                tail = "/".join(path_to_parameters.split("/")[1:])
                path_to_parameters = paths[placeholder] / tail
            parameter_generator = ParameterGenerator.from_file(
                path_to_parameters=path_to_parameters,
                additional_parameters=parameters,
                parameters_to_run=parameters_to_run,
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

    @classmethod
    def init_script_maker(cls, system_configuration, paths, number_of_jobs):
        extra_header_lines = cls._process_placeholders_in_lines(
            lines=system_configuration.get("extra_header_lines", []), paths=paths
        )
        extra_module_lines = cls._process_placeholders_in_lines(
            lines=system_configuration.get("extra_module_lines", []), paths=paths
        )
        extra_command_lines = cls._process_placeholders_in_lines(
            lines=system_configuration.get("extra_command_lines", []), paths=paths
        )
        return ScriptMaker(
            system=system_configuration["system_to_use"],
            run_directory=paths["runs_path"],
            job_name=paths["run_name"],
            memory_per_job=system_configuration["memory_per_job"],
            cpus_per_job=system_configuration["cpus_per_job"],
            number_of_jobs=number_of_jobs,
            extra_header_lines=extra_header_lines,
            extra_module_lines=extra_module_lines,
            extra_command_lines=extra_command_lines,
        )

    @classmethod
    def _process_placeholders_in_lines(cls, lines, paths):
        ret = []
        for line in lines:
            line2 = []
            for word in line.split(" "):
                if "@" in word:
                    placeholder = word.split("/")[0].split("@")[-1]
                    tail = "/".join(word.split("/")[1:])
                    word = paths[placeholder] / tail
                line2.append(str(word))
            line2 = " ".join(line2)
            ret.append(line2)
        return ret

    def save_run_parameters(self):
        for i, parameter in enumerate(self.parameter_generator):
            ret = {}
            save_path = self.paths["runs_path"] / f"run_{i:03d}"
            random_seed = self.run_configuration.get("random_seed", "random")
            if random_seed == "random":
                random_seed = random.randint(0, 1_000_000_000)
            ret["run_number"] = i
            ret["purpose_of_the_run"] = self.run_configuration.get(
                "purpose_of_the_run", "no comment"
            )
            ret["random_seed"] = random_seed
            ret["parameters"] = parameter
            ret["n_days"] = self.parameters["n_days"]
            ret["paths"] = {
                "june_runs_path": self.paths["june_runs_path"],
                "save_path": str(save_path),
                "world_path": self.paths["world_path"],
                "results_path": self.paths["results_path"] / f"run_{i:03d}",
                "baseline_policy_path": self.paths["baseline_policy_path"],
                "baseline_interaction_path": self.paths["baseline_interaction_path"],
                "simulation_config_path": self.paths["simulation_config_path"],
            }
            ret["paths"]["results_path"].mkdir(exist_ok=True, parents=True)
            save_path.mkdir(exist_ok=True, parents=True)
            with open(save_path / "parameters.json", "w") as f:
                json.dump(ret, f, indent=4, default=str)
            with open(ret["paths"]["results_path"] / "parameters.json", "w") as f:
                json.dump(ret, f, indent=4, default=str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tool to setup JUNE runs.")

    parser.add_argument(
        "-c", "--config", help="Path to run config.", required=True,
    )
    parser.add_argument("-s", "--save", help="Store parameters file.", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.save:
        parameter_generator = RunSetup.init_parameter_generator(
            config["parameter_configuration"]
        )
        parameter_generator.save_parameters_to_file(args.save)
        sys.exit()

    run_setup = RunSetup(run_configuration=config,)
    run_setup.save_run_parameters()
    run_setup.script_maker.write_scripts()
