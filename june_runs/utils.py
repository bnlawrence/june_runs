import sys
import psutil
import shutil
import os
import subprocess

from pathlib import Path

import june
from june import paths


def parse_paths(paths_configuration):
    """
    Substitutes placeholders in config.
    """
    june_runs_path = Path(__file__).parent.parent
    default_configs_path = paths.configs_path
    # default configs
    # june default configs path
    names_with_placeholder = []
    names_without_placeholder = []
    ret = {}
    if paths_configuration.get("june_runs_path", "auto") == "auto":
        paths_configuration["june_runs_path"] = june_runs_path.as_posix()
    else:
        june_runs_path = Path(paths_configuration["june_runs_path"])
    if paths_configuration.get("save_path", "auto") == "auto":
        paths_configuration["save_path"] = (june_runs_path / paths_configuration["run_name"]).as_posix()
    if paths_configuration.get("baseline_policy_path", "default") == "default":
        paths_configuration["baseline_policy_path"] = (
            june_runs_path / "configuration/default_baseline_configs/policy.yaml"
        ).as_posix()
    if paths_configuration.get("baseline_interaction_path", "default") == "default":
        paths_configuration["baseline_interaction_path"] = (
            june_runs_path / "configuration/default_baseline_configs/interaction.yaml"
        ).as_posix()
    if paths_configuration.get("simulation_config_path", "default"):
        paths_configuration["simulation_config_path"] = (
            june_runs_path
            / "configuration/default_baseline_configs/simulation_config.yaml"
        ).as_posix()
    for key, value in paths_configuration.items():
        if "path" in key:
            if "@" in str(value):
                names_with_placeholder.append(key)
            else:
                names_without_placeholder.append(key)

    for name in names_without_placeholder:
        if name in ["default", "auto"]:
            raise ValueError("path value incorrect parsed, please submit an issue")
        ret[name] = Path(paths_configuration[name])
    if names_with_placeholder:
        for name in names_with_placeholder:
            value = paths_configuration[name]
            value_split = value.split("/")
            reconstructed = []
            for split in value_split:
                if "@" in split:
                    placeholder_name = split[1:]
                    if placeholder_name == "june_runs_path":
                        reconstructed.append(june_runs_path.as_posix())
                    else:
                        reconstructed.append(ret[placeholder_name].as_posix())
                else:
                    reconstructed.append(split)
            reconstructed = "/".join(reconstructed)
            ret[name] = Path(reconstructed)
    ret["run_name"] = paths_configuration["run_name"]
    run_name = ret["run_name"]
    if ret["save_path"].is_dir():
        i = 1
        newpath = Path(str(ret["save_path"]) + f"_{i}")
        while newpath.is_dir():
            i += 1
            newpath = Path(str(ret["save_path"]) + f"_{i}")
        print(f"{run_name} has already a results folder, saving to {newpath}")
        ret["save_path"] = newpath
    ret["save_path"].mkdir(exist_ok=True, parents=True)

    ret["data_path"] = ret["save_path"] / "data"
    ret["data_path"].mkdir(exist_ok=True, parents=True)

    ret["summary_path"] = ret["save_path"] / "summaries"
    ret["summary_path"].mkdir(exist_ok=True, parents=True)

    ret["runs_path"] = ret["save_path"] / "runs"
    ret["runs_path"].mkdir(exist_ok=True, parents=True)
    return ret


def verbose_print(*args, verbose=False):
    if verbose:
        print(*args)


def memory_status(when="now"):
    mem = psutil.virtual_memory()
    tot = f"total: {mem.total/1024**3:.2f}G"
    used = f"used: {mem.used/1024**3:.2f}G"
    perc = f"percent used: {mem.percent:.2f}%"
    avail = f"avail: {mem.available/1024**3:.2f}G"
    return f"memory {when}: \n    {tot}, {used}, {perc}, {avail}"


def config_checks(config):
    check = "\033[33mCHECK:\033[0m\n   "

    paths = parse_paths(
        config["paths_configuration"],
        region=config["region"],
        iteration=config["iteration"],
    )
    wp = paths["world_path"].stem
    if config["region"] not in paths["world_path"].stem:
        print(check, "Have you set the world_path or region in config correctly?")
    if paths["world_path"].exists() is False:
        print(check, "world_path does not exist.")
    if config["parameter_configuration"].get("parameters_to_run") not in [None, "all"]:
        print(check, 'are you sure you don\'t want "all" parameters_to_run?')
    return None


def git_checks():
    """
    Print the JUNE git version.
    Print the JUNE git SHA
    """
    check = "\033[33mCHECK:\033[0m\n   "
    june_git = Path(june.__path__[0]).parent / ".git"
    branch_cmd = f"git --git-dir {june_git} rev-parse --abbrev-ref HEAD".split()
    try:
        branch = (
            subprocess.run(branch_cmd, stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )
    except:
        return None

    local_SHA_cmd = f'git --git-dir {june_git} log -n 1 --format="%h"'.split()
    try:
        local_SHA = (
            subprocess.run(local_SHA_cmd, stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )
        print(f"You're running with commitID {local_SHA}")
    except:
        print("Can't read local git SHA")
        local_SHA = "unavailable"
    print(f"You're running on branch {branch.upper()}")


def copy_data(new_data_path, june_data_path=None):
    input_data_path = new_data_path / "input"
    covid_real_data_path = new_data_path / "covid_real_data"

    if june_data_path is None:
        june_data_path = june.paths.data_path

    # if input_data_path.exists() is False:
    #    shutil.copytree(june_data_path, new_data_path, dirs_exist_ok=True)
    # else:
    #    print("Skip data copy")

    return None
