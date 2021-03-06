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
        paths_configuration["save_path"] = (
            june_runs_path / paths_configuration["run_name"]
        ).as_posix()
    policy_path = paths_configuration.get("baseline_policy_path", "default")
    if policy_path == "default":
        paths_configuration["baseline_policy_path"] = (
            june_runs_path / "configuration/default_baseline_configs/policy.yaml"
        ).as_posix()
    if paths_configuration.get("baseline_interaction_path", "default") == "default":
        paths_configuration["baseline_interaction_path"] = (
            june_runs_path / "configuration/default_baseline_configs/interaction.yaml"
        ).as_posix()
    if paths_configuration.get("simulation_config_path", "default") == "default":
        paths_configuration["simulation_config_path"] = (
            june_runs_path
            / "configuration/default_baseline_configs/simulation_config.yaml"
        ).as_posix()
    for key, value in paths_configuration.items():
        if type(value) == list and key == "baseline_policy_path":
            ret["baseline_policy_path"] = value
            continue
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
    if "*" in str(ret["baseline_policy_path"]):
        # multiple policy files
        path_split = str(ret["baseline_policy_path"]).split("/")
        policy_folder = Path("/".join(path_split[:-1]))
        glob_pattern = path_split[-1]
        ret["baseline_policy_path"] = list(
            Path(policy_folder).glob(glob_pattern)
        )
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

    ret["data_path"] = ret["save_path"] / "data_used"
    ret["data_path"].mkdir(exist_ok=True, parents=True)

    ret["results_path"] = ret["save_path"] / "results"
    ret["results_path"].mkdir(exist_ok=True, parents=True)

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


def config_checks(
    paths_configuration=None, parameter_configuration=None, system_configuration=None
):
    check = "\033[33mCHECK:\033[0m\n   "
    if paths_configuration is not None:
        if paths_configuration["world_path"].exists() is False:
            print(check, "world_path does not exist.")
    if parameter_configuration is not None:
        if parameter_configuration.get("parameters_to_run") not in [None, "all"]:
            print(check + 'are you sure you don\'t want "all" parameters_to_run?')
    if system_configuration is not None:
        pass
    print("\n")


def git_checks():
    """
    Print the JUNE git version.
    Print the JUNE git SHA
    """
    # TODO: suppress irritating OpenMPI call to fork warning on subprocess call...?
    june_git = Path(june.__path__[0]).parent / ".git"
    branch_cmd = f"git --git-dir {june_git} rev-parse --abbrev-ref HEAD".split()
    try:
        branch = (
            subprocess.run(branch_cmd, stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )
        _branch = (
            f"\033[33m{branch.upper()}\033[0m"
            if branch != "master"
            else f"{branch.upper()}"
        )
        branch_info = f"You're running JUNE on branch {_branch}"
    except:
        branch_info = f"Can't read git branch"
    local_SHA_cmd = f'git --git-dir {june_git} log -n 1 --format="%h"'.split()
    try:
        local_SHA = (
            subprocess.run(local_SHA_cmd, stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )
        SHA_info = f"You're at commitID {local_SHA}"
    except:
        SHA_info = "Can't read local git SHA"
        local_SHA = "unavailable"

    print(branch_info)
    print(SHA_info)


def copy_input_data(new_data_path, june_data_path=None):
    if june_data_path is None:
        june_data_path = june.paths.data_path

    shutil.copytree(june_data_path / "input", new_data_path / "input")
    shutil.copytree(
        june_data_path / "covid_real_data", new_data_path / "covid_real_data"
    )

    return None
