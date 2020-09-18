import sys
import psutil
import shutil
import os
import subprocess

from pathlib import Path

import june

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

    ret["data_path"] = ret["results_path"] / "data"
    ret["data_path"].mkdir(exist_ok=True, parents=True)    

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

def config_checks(config):
    check = "\033[33mCHECK:\033[0m\n   "

    paths = parse_paths(
        config["paths_configuration"], 
        region=config["region"], 
        iteration=config["iteration"]
    )
    wp = paths["world_path"].stem
    if config["region"] not in paths["world_path"].stem:
        print(check,"Have you set the world_path or region in config correctly?")
    if paths["world_path"].exists() is False:
        print(check,"world_path does not exist.")
    if config["parameter_configuration"].get(
            "parameters_to_run"
        ) not in [None, "all"]:
            print(check,"are you sure you don't want \"all\" parameters_to_run?")
    return None

def git_checks():
    """
    Print the JUNE git version.
    Print the JUNE git SHA
    """
    check = "\033[33mCHECK:\033[0m\n   "
    june_git = Path(june.__path__[0]).parent / '.git'
    branch_cmd = f'git --git-dir {june_git} rev-parse --abbrev-ref HEAD'.split()
    try:
        branch = subprocess.run(
            branch_cmd,stdout=subprocess.PIPE
        ).stdout.decode('utf-8').strip()
    except:
        return None

    local_SHA_cmd = f'git --git-dir {june_git} log -n 1 --format="%h"'.split()
    try:
        local_SHA = subprocess.run(
            local_SHA_cmd,stdout=subprocess.PIPE
        ).stdout.decode('utf-8').strip()
        print(f"You\'re running with commitID {local_SHA}")
    except:
        print("Can\'t read local git SHA")
        local_SHA = "unavailable"
    print(f"You\'re running on branch {branch.upper()}")

def copy_data(new_data_path: "PosixPath", june_data_path=None):
    input_data_path = new_data_path / "input"
    covid_real_data_path = new_data_path / "covid_real_data"
    
    if june_data_path is None:
        june_data_path = june.paths.data_path

    #if input_data_path.exists() is False:
    #    shutil.copytree(june_data_path, new_data_path, dirs_exist_ok=True)
    #else:
    #    print("Skip data copy")
    
    return None    










