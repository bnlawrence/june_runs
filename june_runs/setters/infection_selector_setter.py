from pathlib import Path

from june import paths
from june.infection import InfectionSelector, HealthIndexGenerator

transmission_configs = paths.configs_path / "defaults/transmission"


class InfectionSelectorSetter:
    def __init__(self, infectivity_profile: str = "xnexp"):
        self.infectivity_profile = infectivity_profile

    @classmethod
    def from_parameters(cls, parameters: dict):
        infectivity_profile = parameters.get("infection", {}).get(
            "infectivity_profile", "xnexp"
        )
        return cls(infectivity_profile=infectivity_profile)

    def make_infection_selector(self, health_index_generator: HealthIndexGenerator):
        if self.infectivity_profile == "xnexp":
            transmission_path = transmission_configs / "XNExp.yaml"
        elif self.infectivity_profile == "nature":
            transmission_path = transmission_configs / "nature.yaml"
        elif self.infectivity_profile == "correction_nature":
            transmission_path = transmission_configs / "correction_nature.yaml"
        elif self.infectivity_profile == "constant":
            transmission_path = transmission_configs / "TransmissionConstant.yaml"
        else:
            raise NotImplementedError()
        infection_selector = InfectionSelector.from_file(
            transmission_config_path=transmission_path,
            health_index_generator=health_index_generator,
        )
        return infection_selector
