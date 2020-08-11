import numpy as np
import os
import sys
import yaml
from pathlib import Path

default_config_file = Path(__file__).parent.parent / "configs/config_example.yaml"


class Runner:
    """
    A class to handle parameter runs of the JUNE code.
    """

    def __init__(
        self,
        system_configuration: dict = None,
        paths_configuration: dict = None,
        infection_configuration: dict = None,
        region_configuration: dict = None,
        parameter_configuration: dict = None,
    ):
        self.system_configuration = system_configuration
        self.paths_configuration = self._read_paths(paths_configuration)
        self.infection_configuration = infection_configuration
        self.region_configuration = region_configuration
        self.parameter_configuration = parameter_configuration

    @classmethod
    def from_file(cls, config_path: str = default_config_file):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config)

    def _read_paths(self, paths_configuration):
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
                ret[name] = reconstructed
        return ret
