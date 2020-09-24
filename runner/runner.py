import numpy as np
import os
import sys
import json
import yaml
import time
import h5py
import psutil
import numba as nb
import random
from mpi4py import MPI

from pathlib import Path
from collections import OrderedDict, defaultdict

from june.interaction import Interaction
from june.infection.health_index import HealthIndexGenerator
from june.infection.transmission import TransmissionConstant
from june.groups.leisure import generate_leisure_for_config, Cinemas, Pubs, Groceries
from june.groups.travel import Travel
from june.simulator import Simulator
from june.infection_seed import InfectionSeed, Observed2Cases
from june.policy import Policy, Policies, SocialDistancing, Quarantine
from june.infection import InfectionSelector
from june.hdf5_savers import generate_world_from_hdf5, load_population_from_hdf5
from june.hdf5_savers.utils import read_dataset
from june import paths
from june.logger.read_logger import ReadLogger
from june.domain import Domain, generate_super_areas_to_domain_dict
from june.logger import Logger

from .parameter_generator import ParameterGenerator
from .extract_data_new import (
    save_regional_summaries,
    save_age_summaries,
    save_world_summaries,
    save_hospital_summary,
    save_infection_locations,
)
from .plotter import Plotter
from .utils import parse_paths, config_checks, git_checks, verbose_print, memory_status

default_config_file = Path(__file__).parent.parent / "run_configs/config_example.yaml"

default_values = {
    "asymptomatic_ratio": 0.2,
    "infectivity_profile": "XNExp",
    "alpha_physical": 2,
    "lockdown_ratio": 0.5,
    "seed_strength": 1.0,
}


def set_random_seed(seed=999):
    """
    Sets global seeds for testing in numpy, random, and numbaized numpy.
    """

    @nb.njit(cache=True)
    def set_seed_numba(seed):
        random.seed(seed)
        np.random.seed(seed)

    np.random.seed(seed)
    set_seed_numba(seed)
    random.seed(seed)
    return


class Runner:
    """
    A class to handle parameter runs of the JUNE code.
    """

    def __init__(
        self,
        region: str = None,
        iteration: int = 1,
        comment: str = None,
        system_configuration: dict = None,
        paths_configuration: dict = None,
        infection_configuration: dict = None,
        region_configuration: dict = None,
        parameter_configuration: dict = None,
        policy_configuration: dict = None,
        summary_configuration: dict = None,
        verbose: bool = False,
    ):
        # fix seed before everything for reproducibility
        # TODO: save this to the logger or wherever
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()
        if (
            "random_seed" not in parameter_configuration
            or parameter_configuration["random_seed"] == "random"
        ):
            random_seed = random.randint(0, 1_000_000_000)
            print(f"Random seed set to a random value ({random_seed})")
        else:
            random_seed = set_random_seed(int(parameter_configuration["random_seed"]))
            print("Random seed set to {random_seed}")
        set_random_seed(random_seed)
        self.random_seed = random_seed
        self.system_configuration = system_configuration
        self.region = region
        self.comment = f"{comment} - random_state_seed set to {random_seed}"
        self.iteration = iteration
        self.paths_configuration = parse_paths(
            paths_configuration, region=region, iteration=iteration
        )
        self.infection_configuration = infection_configuration
        self.region_configuration = region_configuration
        self.parameter_configuration = parameter_configuration
        self.policy_configuration = policy_configuration
        processed_parameters = self._read_parameter_configuration(
            parameter_configuration["parameters_to_vary"]
        )
        if parameter_configuration.get("config_type", None) == "latin_hypercube":
            self.parameter_generator = ParameterGenerator.from_latin_hypercube(
                processed_parameters,
                n_samples=parameter_configuration["number_of_samples"],
                parameters_to_run=parameter_configuration["parameters_to_run"],
                parameters_to_fix=parameter_configuration.get(
                    "parameters_to_fix", None
                ),
            )
        elif parameter_configuration.get("config_type", None) == "grid":
            self.parameter_generator = ParameterGenerator.from_grid(
                processed_parameters,
                parameters_to_run=parameter_configuration["parameters_to_run"],
                parameters_to_fix=parameter_configuration.get(
                    "parameters_to_fix", None
                ),
            )
        elif parameter_configuration.get("config_type", None) == "regular_grid":
            self.parameter_generator = ParameterGenerator.from_regular_grid(
                processed_parameters,
                parameters_to_run=parameter_configuration["parameters_to_run"],
                parameters_to_fix=parameter_configuration.get(
                    "parameters_to_fix", None
                ),
            )
        elif parameter_configuration.get("config_type", None) == "file":
            self.parameter_generator = ParameterGenerator.from_file(
                parameter_configuration["parameters_to_vary"]["path"],
                parameters_to_run=parameter_configuration["parameters_to_run"],
                parameters_to_fix=parameter_configuration.get(
                    "parameters_to_fix", None
                ),
            )
        else:
            raise NotImplementedError
        self.summary_configuration = summary_configuration
        self.verbose = system_configuration["verbose"]

    @classmethod
    def from_file(cls, config_path: str = default_config_file):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config_checks(config)
        git_checks()
        return cls(**config)

    @staticmethod
    def _set_beta_factors(social_distancing_policy, beta_factor):
        for key in social_distancing_policy.beta_factors:
            if key != "household":
                social_distancing_policy.beta_factors[key] = beta_factor

    def _read_parameter_configuration(self, parameter_configuration):
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

    def generate_domain(self):
        """
        Given the current mpi rank, generates a split of the world (domain) from an hdf5 world.
        If mpi_size is 1 this will return the entire world.
        """
        with h5py.File(self.paths_configuration["world_path"], "r") as f:
            n_super_areas = f["geography"].attrs["n_super_areas"]
        super_areas_to_domain_dict = generate_super_areas_to_domain_dict(
            number_of_super_areas=n_super_areas, number_of_domains=self.mpi_size
        )
        domain = Domain.from_hdf5(
            domain_id=self.mpi_rank,
            super_areas_to_domain_dict=super_areas_to_domain_dict,
            hdf5_file_path=self.paths_configuration["world_path"],
        )
        return domain

    def generate_health_index_generator(self, parameters_dict, verbose=False):
        if "asymptomatic_ratio" in parameters_dict:
            asymptomatic_ratio = parameters_dict["asymptomatic_ratio"]
        else:
            asymptomatic_ratio = default_values["asymptomatic_ratio"]
        return HealthIndexGenerator.from_file(asymptomatic_ratio=asymptomatic_ratio)

    def generate_infection_selector(self, health_index_generator, verbose=False):
        if "infectivity_profile" in self.infection_configuration:
            infectivity_profile = self.infection_configuration["infectivity_profile"]
            verbose_print(
                f"set infectivity profile {infectivity_profile}", verbose=verbose
            )
        else:
            infectivity_profile = default_values["infectivity_profile"]
            verbose_print(
                f"no infectivity profile; default {infectivity_profile}",
                verbose=verbose,
            )
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
            verbose_print(
                f"no alpha_physical; default {alpha_physical:.3f}", verbose=verbose
            )
        interaction.alpha_physical = alpha_physical
        for beta in interaction.beta:
            beta_parameter_name = f"beta_{beta}"
            if beta_parameter_name in parameters_dict:
                interaction.beta[beta] = parameters_dict[beta_parameter_name]
        return interaction

    def generate_leisure(self, domain: Domain):
        leisure = generate_leisure_for_config(
            domain, self.paths_configuration["config_path"]
        )
        return leisure

    def generate_travel(self):
        travel = Travel()
        return travel

    def generate_policies(self, parameters_dict, verbose=False):
        policies = Policies.from_file()
        policies_to_modify = defaultdict(list)
        policy_types = set()
        if "lockdown_ratio" in self.policy_configuration:
            lockdown_ratio = self.policy_configuration["lockdown_ratio"]
            verbose_print(f"set lockdown_ratio {lockdown_ratio:.3f}", verbose=verbose)
        else:
            lockdown_ratio = default_values["lockdown_ratio"]
            verbose_print(
                f"no lockdown_ratio; default {lockdown_ratio:.3f}", verbose=verbose
            )
        for policy in policies:
            policy_name = policy.spec
            for parameter_name in parameters_dict:
                if policy_name in parameter_name:
                    policies_to_modify[policy_name].append(policy)
            if policy_name == "quarantine":
                print(policy.__dict__)

        print(policies_to_modify)

        for policy_name, policies_mod in policies_to_modify.items():

            #!!!!===============================================================!!!!#
            # fix added [1:]
            #!!!!===============================================================!!!!#
            parameters = [
                (parameter.split(policy_name)[-1][1:], value)
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
                        first_policy, (parameter_value - 1) * lockdown_ratio + 1
                    )
                    self._set_beta_factors(second_policy, parameter_value)

                    print("\n\n", first_policy.__dict__)
                    print("\n\n", second_policy.__dict__)
                    print("\n\n UNMODIFIED", policies_mod[2].__dict__)
                ### THIS NEEDS FIXING, ASAP!!!
                elif "compliance" in parameter_name:
                    print("this is a complinance")
                    first_policy = policies_mod[0]
                    second_policy = policies_mod[1]
                    setattr(
                        first_policy, parameter_name, lockdown_ratio * parameter_value
                    )
                    setattr(second_policy, parameter_name, parameter_value)
                    print("\n\n", first_policy.__dict__)
                    print("\n\n", second_policy.__dict__)

        return policies

    def generate_infection_seed(
        self, parameters_dict, domain, infection_selector, verbose=True
    ):
        if "seed_strength" in parameters_dict:
            seed_strength = parameters_dict["seed_strength"]
            verbose_print(f"set seed strength {seed_strength:.3f}", verbose=verbose)
        else:
            seed_strength = default_values["seed_strength"]
            verbose_print(
                f"no seed strength; default {seed_strength:.3f}", verbose=verbose
            )
        if "age_profile" in parameters_dict:
            age_profile = parameters_dict["age_profile"]
            print(f"doing something with age_profile", age_profile)
        else:
            verbose_print(f"no age_profile", verbose=verbose)
        oc = Observed2Cases.from_file(
            health_index_generator=infection_selector.health_index_generator,
            smoothing=True,
        )
        daily_cases_per_region = oc.get_regional_latent_cases()
        daily_cases_per_super_area = oc.convert_regional_cases_to_super_area(
                daily_cases_per_region, 
                dates=["2020-02-28","2020-03-02"]
                )

        infection_seed = InfectionSeed(
            world=domain,
            infection_selector=infection_selector,
            daily_super_area_cases=daily_cases_per_super_area,
            seed_strength=seed_strength,
        )
        print("\n\n infection seed\n")
        print(infection_seed.__dict__)
        return infection_seed

    def generate_logger(self, save_path: str):
        logger = Logger(save_path=save_path, file_name=f"logger.{self.mpi_rank}.hdf5")
        return logger

    def get_index_to_run(self, parameter_index):
        index_to_run = self.parameter_generator.parameters_to_run[parameter_index]
        return index_to_run

    def generate_simulator(self, parameter_index, verbose=None):
        print("Running with config: \n")
        print("sys config", self.system_configuration)
        print("path config", self.paths_configuration)
        print("inf config", self.infection_configuration)
        print("region config", self.region_configuration)
        print("param config", self.parameter_configuration)
        print("pol config", self.policy_configuration)
        print("summ config", self.summary_configuration)
        domain = self.generate_domain()
        if verbose is None:
            verbose = self.verbose
        parameters_dict = self.parameter_generator[parameter_index]
        run_number = parameters_dict["run_number"]
        verbose_print(
            f"Run number {run_number} params:", parameters_dict, verbose=verbose
        )  #
        run_name = f"run_{run_number:03}"
        save_path = self.paths_configuration["results_path"] / run_name
        save_path.mkdir(exist_ok=True, parents=True)
        with open(save_path / "parameters.json", "w") as f:
            json.dump(parameters_dict, f)
        health_index_generator = self.generate_health_index_generator(parameters_dict)
        infection_selector = self.generate_infection_selector(health_index_generator)
        interaction = self.generate_interaction(parameters_dict)
        print("\n\ninteraction:\n", interaction.__dict__, "\n\n")
        leisure = self.generate_leisure(domain=domain)
        travel = self.generate_travel()
        policies = self.generate_policies(parameters_dict)
        verbose_print(memory_status(when="before world"), verbose=verbose)  #
        verbose_print(memory_status(when="after world"), verbose=verbose)  #
        # TODO: put comment into logger here (and save path) to not clog simulator
        logger = self.generate_logger(save_path=save_path)
        logger.log_population(domain.people)
        infection_seed = self.generate_infection_seed(
            parameters_dict=parameters_dict, domain=domain, infection_selector=infection_selector
        )
        print("Comment is...", self.comment)
        simulator = Simulator.from_file(
            world=domain,
            interaction=interaction,
            config_filename=self.paths_configuration["config_path"],
            leisure=leisure,
            travel=travel,
            infection_seed=infection_seed,  
            infection_selector=infection_selector,
            policies=policies,
            logger=logger,
            # comment=self.comment,#TODO: move this to logger
        )
        return simulator

    # @staticmethod # Can't decide, static or not - would be helpful to call as static for failed loggers...
    def extract_summaries(
        self, parameter_index=None, logger_dir=None, summary_dir=None, verbose=False
    ):

        if parameter_index is not None:
            index_to_run = self.get_index_to_run(parameter_index)
            run_name = f"run_{parameter_index:03}"

        if logger_dir is None:
            logger_dir = self.paths_configuration["results_path"] / run_name

        if summary_dir is None:
            summary_dir = self.paths_configuration["summary_path"]

        t1 = time.time()
        try:
            logger = ReadLogger(logger_dir)
            logger.load_infection_location()
        except Exception as e:
            print(str(e))
            l1 = "***" + 19 * " " + "***"
            print(f'{l1}\n{4*" "}CAN\'T READ LOGGER{4*" "}\n{l1}')
            return None
        t2 = time.time()
        verbose_print(f"{(t2-t1)/60.}", verbose=verbose)

        # contains info on super areas, probably the most complete run summary
        run_summary_path = summary_dir / f"run_summary_{index_to_run:03}.csv"
        daily_regional_path = (
            summary_dir / f"daily_regional_summary_{index_to_run:03}.csv"
        )
        save_regional_summaries(logger, run_summary_path, daily_regional_path)

        world_path = summary_dir / f"world_summary_{index_to_run:03}.csv"
        daily_world_path = summary_dir / f"daily_world_summary_{index_to_run:03}.csv"
        save_world_summaries(logger, world_path, daily_world_path)

        if self.summary_configuration:
            age_bins = self.summary_configuration["age_bins"]
        else:
            age_bins = None

        age_path = summary_dir / f"age_summary_{parameter_index:03}.csv"
        daily_age_path = summary_dir / f"daily_age_summary_{index_to_run:03}.csv"
        save_age_summaries(logger, age_path, daily_age_path, age_bins=age_bins)

        hospital_path = summary_dir / f"hospital_summary_{index_to_run:03}.csv"
        save_hospital_summary(logger, hospital_path)

        infection_locations_path = (
            summary_dir / f"total_infection_locations_{index_to_run:03}.csv"
        )
        daily_loc_ts_path = (
            summary_dir / f"daily_infection_loc_timeseries_{index_to_run:03}.csv"
        )
        save_infection_locations(logger, infection_locations_path, daily_loc_ts_path)

        # logger does these differently now
        # inf_loc_ts_path = summary_dir / f"infection_loc_timeseries_{parameter_index:03}.csv"
        # save_location_infections_timeseries(logger, inf_loc_ts_path,daily_loc_ts_path)

        real_data_path = Path("/cosma5/data/durham/dc-truo1/june_analysis")
        # real_data_path = Path('/home/htruong/Documents/JUNE/Notebooks')
        plotter = Plotter(logger, real_data_path, summary_dir, parameter_index)
        plotter.plot_region_data()
        plotter.plot_age_stratified(sitrep_bins=False)
        plotter.plot_age_stratified(sitrep_bins=True)

        return None
