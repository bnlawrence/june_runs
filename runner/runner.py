import numpy as np
import os
import sys
import json
import yaml
import time
import psutil

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
from june.logger.read_logger import ReadLogger
from .parameter_generator import ParameterGenerator
from .extract_data import *

default_config_file = Path(__file__).parent.parent / "run_configs/config_example.yaml"

default_values = {
    "asymptomatic_ratio": 0.2,
    "infectivity_profile": "XNExp",
    "alpha_physical": 2,
    "lockdown_ratio": 0.5,
    "seed_strength": 1.0,
}


def parse_paths(paths_configuration, region, iteration):
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
        ret[name] = Path(paths_configuration[name])
    if names_with_placeholder:
        for name in names_with_placeholder:
            value = paths_configuration[name]
            value_split = value.split("/")
            reconstructed = []
            for split in value_split:
                if "@" in split:
                    placeholder_name = split[1:]
                    reconstructed.append(ret[placeholder_name].as_posix())
                else:
                    reconstructed.append(split)
            reconstructed = "/".join(reconstructed)
            ret[name] = Path(reconstructed)

    ret["results_path"] = ret["results_path"] / region / f"iteration_{iteration:02}"
    ret["results_path"].mkdir(exist_ok=True, parents=True)

    ret["summary_path"] = ret["summary_path"] / region / f"iteration_{iteration:02}"
    ret["summary_path"].mkdir(exist_ok=True, parents=True)    

    return ret

def verbose_print(*args,verbose=False):
    if verbose:
        print(*args)

def memory_status(when='now'):
    mem = psutil.virtual_memory()
    tot = f"total: {mem.total/1024**3:.2f}G"
    used = f"used: {mem.used/1024**3:.2f}G"
    perc = f"percent used: {mem.percent:.2f}%"
    avail = f"avail: {mem.available/1024**3:.2f}G"
    return f"memory {when}: \n    {tot}, {used}, {perc}, {avail}"

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
        policy_configuration: dict = None,
        summary_configuration: dict = None,
        verbose: bool = False
    ):
        self.system_configuration = system_configuration
        self.region = region
        self.iteration = iteration
        self.paths_configuration = parse_paths(
            paths_configuration, region=region, iteration=iteration
        )
        self.infection_configuration = infection_configuration
        self.region_configuration = region_configuration
        self.parameter_configuration = parameter_configuration
        self.policy_configuration = policy_configuration
        self.parameter_generator = ParameterGenerator(parameter_configuration)
        self.summary_configuration = summary_configuration
        self.verbose = system_configuration["verbose"]

    @classmethod
    def from_file(cls, config_path: str = default_config_file):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config)

    @staticmethod
    def _set_beta_factors(self, social_distancing_policy, beta_factor):
        for key in social_distancing_policy.beta_factors:
            if key != "household":
                social_distancing_policy.beta_factors[key] = beta_factor

    def generate_world(self):
        world = generate_world_from_hdf5(self.paths_configuration["world_path"])
        return world

    def generate_health_index_generator(self, parameters_dict, verbose=False):
        if "asymptomatic_ratio" in parameters_dict:
            asymptomatic_ratio = parameters_dict["asymptomatic_ratio"]
        else:
            asymptomatic_ratio = default_values["asymptomatic_ratio"]
        return HealthIndexGenerator.from_file(asymptomatic_ratio=asymptomatic_ratio)

    def generate_infection_selector(self, health_index_generator, verbose=False):
        if "infectivity_profile" in self.infection_configuration:
            infectivity_profile = self.infection_configuration["infectivity_profile"]
            verbose_print(f"set infectivity profile {infectivity_profile}", verbose=verbose)
        else:
            infectivity_profile = default_values["infectivity_profile"]
            verbose_print(f"no infectivity profile; default {infectivity_profile}", verbose=verbose)
        transmission_config = Path(f"defaults/transmission/{infectivity_profile}.yaml")
        infection_selector = InfectionSelector.from_file(
            transmission_config_path=paths.configs_path / transmission_config,
            health_index_generator=health_index_generator,
        )
        return infection_selector

    def generate_interaction(self, parameters_dict, verbose=False):
        interaction = Interaction.from_file()
        if "alpha_physical" in parameters_dict:
            alpha_physical = parameters_dict["alpha_physical"]
            verbose_print(f"set alpha_physical {alpha_physical:.3f}", verbose=verbose)
        else:
            alpha_physical = default_values["alpha_physical"]
            verbose_print(f"no alpha_physical; default {alpha_physical:.3f}",  verbose=verbose)
        interaction.alpha_physical = alpha_physical
        for beta in interaction.beta:
            beta_parameter_name = f"beta_{beta}"
            if beta_parameter_name in parameters_dict:
                interaction.beta[beta] = parameters_dict[beta_parameter_name]
        return interaction

    def generate_policies(self, parameters_dict,verbose=False):
        policies = Policies.from_file()
        policies_to_modify = defaultdict(list)
        policy_types = set()
        if "lockdown_ratio" in self.policy_configuration:
            lockdown_ratio = self.policy_configuration["lockdown_ratio"]
            verbose_print(f"set lockdown_ratio {lockdown_ratio:.3f}", verbose=verbose)
        else:
            lockdown_ratio = default_values["lockdown_ratio"]
            verbose_print(f"no lockdown_ratio; default {lockdown_ratio:.3f}", verbose=verbose)
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

    def generate_infection_seed(self, parameters_dict, infection_selector, world, verbose=False):
        if "seed_strength" in parameters_dict:
            seed_strength = parameters_dict["seed_strength"]
            verbose_print(f"set seed strength {seed_strength:.3f}",verbose=verbose)
        else:
            seed_strength = default_values["seed_strength"]
            verbose_print(f"no seed strength; default {seed_strength:.3f}",verbose=verbose)
        oc = Observed2Cases.from_file(
            super_areas=world.super_areas,
            health_index=infection_selector.health_index_generator,
        )
        n_cases_df = oc.cases_from_deaths()
        # Seed over 2 days
        n_cases_to_seed_df = n_cases_df.loc["2020-02-28":"2020-02-29"]
        infection_seed = InfectionSeed.from_file(
            super_areas=world.super_areas,
            selector=infection_selector,
            n_cases_region=n_cases_to_seed_df,
            seed_strength=seed_strength,
        )
        return infection_seed

    def generate_simulator(self, parameter_index, verbose=None):
        if verbose is None:
            verbose=self.verbose
        parameters_dict = self.parameter_generator[parameter_index]
        verbose_print(f"{parameter_index} params:",parameters_dict, verbose=verbose) #
        run_name = f"run_{parameter_index:03}"
        save_path = self.paths_configuration["results_path"] / run_name
        save_path.mkdir(exist_ok=True, parents=True)
        with open(save_path / "parameters.json", "w") as f:
            json.dump(parameters_dict, f)
        health_index_generator = self.generate_health_index_generator(parameters_dict)
        infection_selector = self.generate_infection_selector(health_index_generator)
        interaction = self.generate_interaction(parameters_dict)
        policies = self.generate_policies(parameters_dict)
        verbose_print(memory_status(when='before world'), verbose=verbose) #
        world = self.generate_world()
        verbose_print(memory_status(when='after world'), verbose=verbose) #
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
            save_path=save_path,
        )
        return simulator

    #@staticmethod # Can't decide, static or not - would be helpful to call as static for failed loggers...
    def extract_summaries(self, parameter_index=None, logger_dir=None, summary_dir=None, verbose=False):
        
        if parameter_index is not None:
            run_name = f"run_{parameter_index:03}"

        if logger_dir is None:
            logger_dir = self.paths_configuration["results_path"] / run_name

        if summary_dir is None:
            summary_dir = self.paths_configuration["summary_path"]

        t1 = time.time()
        try:
            logger = ReadLogger(logger_dir)
        except Exception as e:
            print(str(e))
            l1 = '***'+19*' '+'***'
            print(f'{l1}\n{4*" "}CAN\'T READ LOGGER{4*" "}\n{l1}')
            return None
        t2 = time.time()
        verbose_print(f"{(t2-t1)/60.}",verbose=verbose)

        world_path = summary_dir / f"world_summary_{parameter_index:03}.csv"
        daily_world_path = summary_dir / f"daily_world_summary_{parameter_index:03}.csv"
        save_world_summaries(logger,world_path, daily_world_path)

        if self.summary_configuration:
            age_bins = self.summary_configuration["age_bins"]
        else:
            age_bins = None

        age_path = summary_dir / f"age_summary_{parameter_index:03}.csv"
        daily_age_path = summary_dir / f"daily_age_summary_{parameter_index:03}.csv"   
        save_age_summaries(logger,age_path,daily_age_path,age_bins=age_bins)

        hospital_path = summary_dir / f"hospital_summary_{parameter_index:03}.csv"
        save_hospital_summary(logger,hospital_path)
        
        return None
        
















