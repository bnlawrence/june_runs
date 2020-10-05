import json
import h5py

from june.domain import Domain, generate_super_areas_to_domain_dict
from june.mpi_setup import mpi_rank, mpi_size
from june.groups.leisure import generate_leisure_for_config
from june.groups.travel import Travel
from june.records import Record
from june.simulator import Simulator

from june_runs.setters import (
    InteractionSetter,
    PolicySetter,
    InfectionSeedSetter,
    InfectionSelectorSetter,
    HealthIndexSetter,
)


class Runner:
    def __init__(self, parameter_file):
        with open(parameter_file, "r") as f:
            self.parameters = json.load(f)

    def generate_domain(self):
        """
        Given the current mpi rank, generates a split of the world (domain) from an hdf5 world.
        If mpi_size is 1 this will return the entire world.
        """
        with h5py.File(self.parameters["world_path"], "r") as f:
            n_super_areas = f["geography"].attrs["n_super_areas"]
        super_areas_to_domain_dict = generate_super_areas_to_domain_dict(
            number_of_super_areas=n_super_areas, number_of_domains=mpi_size
        )
        domain = Domain.from_hdf5(
            domain_id=mpi_rank,
            super_areas_to_domain_dict=super_areas_to_domain_dict,
            hdf5_file_path=self.parameters["world_path"],
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

    def generate_interaction(self):
        interaction_setter = InteractionSetter.from_parameters(self.parameters)
        return interaction_setter.make_interaction()

    def generate_leisure(self, domain: Domain):
        leisure = generate_leisure_for_config(
            domain, self.paths_configuration["config_path"]
        )
        return leisure

    def generate_travel(self):
        travel = Travel()
        return travel

    def generate_policies(self):
        policy_setter = PolicySetter.from_parameters(
            baseline_policy_path=self.parameters["baseline_policy_path"],
            policies_to_modify=self.parameters["policies"],
        )
        return policy_setter.make_policies()

    def generate_record(self):
        record = Record(
            record_path=self.parameters["save_path"],
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
        interaction = self.generate_interaction()
        leisure = self.generate_leisure(domain=domain)
        travel = self.generate_travel()
        policies = self.generate_policies()
        record = self.generate_record()
        record.static_data(world=domain)
        infection_seed = self.generate_infection_seed(
            world=domain,
            infection_selector=infection_selector,
        )
        record.meta_information(
            comment=self.comment,
            random_state=self.random_seed,
            number_of_cores=self.system_configuration["cores_per_job"],
        )
        simulator = Simulator.from_file(
            world=domain,
            interaction=interaction,
            config_filename=self.paths_configuration["config_path"],
            leisure=leisure,
            travel=travel,
            infection_seed=infection_seed,
            infection_selector=infection_selector,
            policies=policies,
            record=record,
        )
        return simulator

    def run(self):
        simulator = self.generate_simulator()
        simulator.run()
