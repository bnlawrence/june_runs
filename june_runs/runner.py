import json
import h5py
import numba as nb
import random
import numpy as np
import datetime
from pathlib import Path
from time import time

from june.domain import Domain, DomainSplitter
from june.mpi_setup import mpi_rank, mpi_size, mpi_comm
from june.groups.leisure import generate_leisure_for_config
from june.groups.travel import Travel
from june.records import Record
from june.records.records_writer import combine_records
from june.simulator import Simulator

from june_runs.setters import (
    InteractionSetter,
    PolicySetter,
    InfectionSeedSetter,
    InfectionSelectorSetter,
    HealthIndexSetter,
)


def keys_to_int(x):
    return {int(k): v for k, v in x.items()}


def set_random_seed(seed=999):
    """
    Sets global seeds for testing in numpy, random, and numbaized numpy.
    """

    @nb.njit()
    def set_seed_numba(seed):
        random.seed(seed)
        np.random.seed(seed)

    np.random.seed(seed)
    set_seed_numba(seed)
    random.seed(seed)
    return


class Runner:
    def __init__(self, run_config):
        with open(run_config, "r") as f:
            run_config = json.load(f)
        self.random_seed = run_config["random_seed"]
        set_random_seed(self.random_seed)
        self.paths = run_config["paths"]
        self.parameters = run_config["parameters"]
        self.purpose_of_the_run = run_config["purpose_of_the_run"]
        self.run_number = run_config["run_number"]
        self.n_days = run_config["n_days"]

    def generate_domain(self):
        """
        Given the current mpi rank, generates a split of the world (domain) from an hdf5 world.
        If mpi_size is 1 this will return the entire world.
        """
        save_path = Path(self.paths["save_path"])
        if mpi_rank == 0:
            with h5py.File(self.paths["world_path"], "r") as f:
                super_area_names = [
                    name.decode() for name in f["geography"]["super_area_name"]
                ]
                super_area_ids = [
                    int(sa_id) for sa_id in f["geography"]["super_area_id"]
                ]
            super_area_name_to_id = {
                key: value for key, value in zip(super_area_names, super_area_ids)
            }
            # make dictionary super_area_id -> domain
            domain_splitter = DomainSplitter(
                #number_of_domains=mpi_size, world_path=self.paths["world_path"]
                number_of_domains=mpi_size, super_areas=super_area_names
            )
            super_areas_per_domain = domain_splitter.generate_domain_split(niter=12)
            super_area_names_to_domain_dict = {}
            super_area_ids_to_domain_dict = {}
            for domain, super_areas in super_areas_per_domain.items():
                for super_area in super_areas:
                    super_area_names_to_domain_dict[super_area] = domain
                    super_area_ids_to_domain_dict[
                        int(super_area_name_to_id[super_area])
                    ] = domain

            with open(save_path / "super_area_ids_to_domain.json", "w") as f:
                json.dump(super_area_ids_to_domain_dict, f)
            with open(save_path / "super_area_names_to_domain.json", "w") as f:
                json.dump(super_area_names_to_domain_dict, f)
        mpi_comm.Barrier() # wait until rank 0 writes domain partition
        if mpi_rank > 0:
            with open(save_path / "super_area_ids_to_domain.json", "r") as f:
                super_area_ids_to_domain_dict= json.load(f, object_hook=keys_to_int)
        domain = Domain.from_hdf5(
            domain_id=mpi_rank,
            super_areas_to_domain_dict=super_area_ids_to_domain_dict,
            hdf5_file_path=self.paths["world_path"],
        )
        return domain

    def generate_health_index_generator(self):
        health_index_setter = HealthIndexSetter.from_parameters(self.parameters)
        return health_index_setter.make_health_index()

    def generate_infection_selector(self, health_index_generator):
        infection_selector_setter = InfectionSelectorSetter.from_parameters(
            self.parameters
        )
        return infection_selector_setter.make_infection_selector(
            health_index_generator=health_index_generator
        )

    def generate_interaction(self, baseline_interaction_path, population):
        interaction_setter = InteractionSetter.from_parameters(
            self.parameters, baseline_interaction_path, population=population
        )
        return interaction_setter.make_interaction()

    def generate_leisure(self, domain: Domain):
        leisure = generate_leisure_for_config(
            domain, self.paths["simulation_config_path"]
        )
        return leisure

    def generate_travel(self):
        travel = Travel()
        return travel

    def generate_policies(self):
        policy_setter = PolicySetter.from_parameters(
            baseline_policy_path=self.paths["baseline_policy_path"],
            policies_to_modify=self.parameters.get("policies", None),
        )
        return policy_setter.make_policies()

    def generate_record(self):
        record = Record(
            record_path=self.paths["save_path"],
            record_static_data=True,
            mpi_rank=mpi_rank,
        )
        return record

    def generate_infection_seed(self, infection_selector, world):
        infection_seed_setter = InfectionSeedSetter.from_parameters(self.parameters)
        return infection_seed_setter.make_infection_seed(
            world=world, infection_selector=infection_selector
        )

    def generate_simulator(self):
        domain = self.generate_domain()
        health_index_generator = self.generate_health_index_generator()
        infection_selector = self.generate_infection_selector(
            health_index_generator=health_index_generator
        )
        interaction = self.generate_interaction(
            baseline_interaction_path=self.paths["baseline_interaction_path"],
            population=domain.people,
        )
        leisure = self.generate_leisure(domain=domain)
        travel = self.generate_travel()
        policies = self.generate_policies()
        record = self.generate_record()
        record.static_data(world=domain)
        infection_seed = self.generate_infection_seed(
            world=domain, infection_selector=infection_selector,
        )
        record.meta_information(
            comment=self.purpose_of_the_run,
            random_state=self.random_seed,
            number_of_cores=mpi_size,
        )
        simulator = Simulator.from_file(
            world=domain,
            interaction=interaction,
            config_filename=self.paths["simulation_config_path"],
            leisure=leisure,
            travel=travel,
            infection_seed=infection_seed,
            infection_selector=infection_selector,
            policies=policies,
            record=record,
        )
        # change number of days, this can only be done like this for now
        simulator.timer.total_days = self.n_days
        simulator.timer.final_date = simulator.timer.initial_date + datetime.timedelta(
            days=self.n_days
        )
        return simulator

    def run(self):
        simulator = self.generate_simulator()
        time1 = time()
        simulator.run()
        time2 = time()
        if mpi_rank == 0:
            print(f"Finished! Simulation took {time2-time1} seconds!")
            self.save_results()
            print(f"Results saved!")

    def save_results(self):
        results_path = self.paths["results_path"]
        combine_records(Path(self.paths["save_path"]), remove_left_overs=False, save_dir=Path(results_path))
