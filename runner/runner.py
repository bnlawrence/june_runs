import numpy as np
import os
import sys
import yaml
from pyDOE2 import lhs
from SALib.util import scale_samples
from pathlib import Path
from collections import OrderedDict, defaultdict

from june.interaction import Interaction
from june.infection.health_index import HealthIndexGenerator
from june.infection.transmission import TransmissionConstant
from june.groups.leisure import generate_leisure_for_config, Cinemas, Pubs, Groceries
from june.simulator import Simulator
from june.infection_seed import InfectionSeed, Observed2Cases
from june.policy import Policy, Policies, SocialDistancing, Quarantine
from june.infection.infection import InfectionSelector
from june.world import generate_world_from_hdf5
from june import paths
from .parameter_generator import ParameterGenerator

default_config_file = Path(__file__).parent.parent / "run_configs/config_example.yaml"

default_values = {
    "asymptomatic_ratio": 0.2,
    "infectivity_profile": "XNExp",
    "alpha_physical": 2,
    "lockdown_ratio": 0.5,
    "seed_strength": 1.0,
}


class Runner:
    """
    A class to handle parameter runs of the JUNE code.
    """

    def __init__(
        self,
        region: str = None,
        iteration: int = 1,
        system_configuration: dict = None,
        paths_configuration: dict = None,
        infection_configuration: dict = None,
        region_configuration: dict = None,
        parameter_configuration: dict = None,
        policy_configuration: dict = None
    ):
        self.system_configuration = system_configuration
        self.region = region
        self.iteration = iteration
        self.paths_configuration = self._read_paths(paths_configuration)
        self.infection_configuration = infection_configuration
        self.region_configuration = region_configuration
        self.parameter_configuration = parameter_configuration
        self.policy_configuration = policy_configuration
        self.parameter_generator = ParameterGenerator(parameter_configuration)

    @classmethod
    def from_file(cls, config_path: str = default_config_file):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config)

    def _read_paths(self, paths_configuration):
        """
        Substitutes placeholders in config.
        """
        ret = {}
        names_with_placeholder = [
            key for key, value in paths_configuration.items() if "@" in value
        ]
        names_without_placeholder = [
            key for key, value in paths_configuration.items() if "@" not in value
        ]
        for name in names_without_placeholder:
            ret[name] = paths_configuration[name]
        if names_with_placeholder:
            for name in names_with_placeholder:
                value = paths_configuration[name]
                value_split = value.split("/")
                reconstructed = []
                for split in value_split:
                    if "@" in split:
                        placeholder_name = split[1:]
                        reconstructed.append(ret[placeholder_name])
                    else:
                        reconstructed.append(split)
                reconstructed = "/".join(reconstructed)
                ret[name] = Path(reconstructed)
        ret["results_path"] = (
            ret["results_path"] / self.region / f"iteration_{self.iteration:02}"
        )
        ret["results_path"].mkdir(exist_ok=True, parents=True)
        return ret

    @staticmethod
    def _set_beta_factors(self, social_distancing_policy, beta_factor):
        for key in social_distancing_policy.beta_factors:
            if key != "household":
                social_distancing_policy.beta_factors[key] = beta_factor

    def generate_world(self):
        world = generate_world_from_hdf5(self.paths_configuration["world_path"])
        return world
    
    def generate_health_index_generator(self, parameters_dict):
        if "asymptomatic_ratio" in parameters_dict:
            asymptomatic_ratio = parameters_dict["asymptomatic_ratio"]
        else:
            asymptomatic_ratio = default_values["asymptomatic_ratio"]
        return HealthIndexGenerator.from_file(asymptomatic_ratio=asymptomatic_ratio)


    def generate_infection_selector(self, health_index_generator):
        if "infectivity_profile" in self.infection_configuration:
            infectivity_profile = self.infection_configuration["infectivity_profile"]
        else:
            infectivity_profile = default_values["infectivity_profile"]
        transmission_config = Path(f"defaults/transmission/{infectivity_profile}.yaml")
        infection_selector = InfectionSelector.from_file(
            transmission_config_path=paths.configs_path / transmission_config,
            health_index_generator=health_index_generator,
        )
        return infection_selector

    def generate_interaction(self, parameters_dict):
        interaction = Interaction.from_file()
        if "alpha_physical" in parameters_dict:
            alpha_physical = parameters_dict["alpha_physical"]
        else:
            alpha_physical = default_values["alpha_physical"]
        interaction.alpha_physical = alpha_physical
        for beta in interaction.beta:
            beta_parameter_name = f"beta_{beta}"
            if beta_parameter_name in parameters_dict:
                interaction.beta[beta] = parameters_dict[beta_parameter_name]
        return interaction

    def generate_policies(self, parameters_dict):
        policies = Policies.from_file()
        policies_to_modify = defaultdict(list)
        policy_types = set()
        if "lockdown_ratio" in self.policy_configuration:
            lockdown_ratio = self.policy_configuration["lockdown_ratio"]
        else:
            lockdown_ratio = default_values["lockdown_ratio"]
        for policy in policies:
            policy_name = policy.spec
            for parameter_name in parameters_dict:
                if policy_name in parameter_name:
                    policies_to_modify[policy_name].append(policy)
        for policy_name, policies_mod in policies_to_modify.items():
            parameters = [
                (parameter.split(policy_name)[-1], value)
                for parameter, value in parameters_dict.items()
                if policy_name in parameter
            ]
            for parameter_name, parameter_value in parameters:
                # for now we assume we have two lockdown policies.
                # TODO: make it general
                if (
                    policy_name == "social_distancing"
                    and parameter_name == "beta_factor"
                ):
                    first_policy = policies_mod[0]
                    second_policy = policies_mod[1]
                    self._set_beta_factors(
                        (parameter_value - 1) * lockdown_ratio + 1, first_policy
                    )
                    self._set_beta_factors(parameter_value, second_policy)
                elif "compliance" in parameter_name:
                    first_policy = policies_mod[0]
                    second_policy = policies_mod[1]
                    setattr(
                        first_policy, parameter_name, lockdown_ratio * parameter_value
                    )
                    setattr(second_policy, parameter_name, parameter_value)
        return policies

    def generate_infection_seed(self, parameters_dict, infection_selector, world):
        if "seed_strength" in parameters_dict:
            seed_strength = parameters_dict["seed_strength"]
        else:
            seed_strength = default_values["seed_strength"]
        oc = Observed2Cases.from_file(
            super_areas=world.super_areas,
            health_index=infection_selector.health_index_generator,
        )
        n_cases_df = oc.cases_from_deaths()
        # Seed over 5 days
        n_cases_to_seed_df = n_cases_df.loc["2020-03-01":"2020-03-02"]
        infection_seed = InfectionSeed.from_file(
            super_areas=world.super_areas,
            selector=infection_selector,
            n_cases_region=n_cases_to_seed_df,
            seed_strength=seed_strength,
        )
        return infection_seed

    def generate_simulator(self, parameter_index):
        world = self.generate_world()
        parameters_dict = self.parameter_generator[parameter_index]
        health_index_generator = self.generate_health_index_generator(parameters_dict)
        infection_selector = self.generate_infection_selector(health_index_generator)
        interaction = self.generate_interaction(parameters_dict)
        policies = self.generate_policies(parameters_dict)
        infection_seed = self.generate_infection_seed(
            parameters_dict, infection_selector, world
        )
        leisure = generate_leisure_for_config(
            world, self.paths_configuration["config_path"]
        )
        simulator = Simulator.from_file(
            world=world,
            interaction=interaction,
            config_filename=self.paths_configuration["config_path"],
            leisure=leisure,
            infection_seed=infection_seed,
            infection_selector=infection_selector,
            policies=policies,
            save_path=self.paths_configuration["results_path"],
        )
        return simulator

