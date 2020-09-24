import sys
import psutil
import shutil
import os
import yaml
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

def parse_extra_paths(extra_config,parsed_paths,runner_variables={},verbose=False):
    if extra_config is None:
        return None
    for key,value in extra_config.items():
        if type(value) is dict:
            parse_extra_paths(value,parsed_paths,runner_variables)
        if key.endswith('path'):
            value_split = value.split('/')
            reconstructed = []
            for split in value_split:
                if "@" in split:
                    placeholder_name = split[1:]
                    reconstructed.append(parsed_paths[placeholder_name].as_posix())
                elif "{" in split and "}" in split:
                    print("modding",split)
                    split = split.format(**runner_variables)
                    print("now",split)
                    reconstructed.append(split)
                else:
                    reconstructed.append(split)
            reconstructed = "/".join(reconstructed)
            extra_config[key] = Path(reconstructed)
        
    verbose_print("modified config to", extra_config,verbose=verbose)

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
    sim_config_path = paths["config_path"]
    with open(sim_config_path, "r") as f:
        sim_config = yaml.load(f, Loader=yaml.FullLoader)

    wp = paths["world_path"].stem
    if config["region"] not in paths["world_path"].stem:
        print(check,"Have you set the world_path or region in config correctly?")
    if paths["world_path"].exists() is False:
        print(check,"world_path does not exist.")
    if config["parameter_configuration"].get(
            "parameters_to_run"
        ) not in [None, "all"]:
            print(check,"are you sure you don't want \"all\" parameters_to_run?")
    if config["checkpoint_configuration"] is not None:
        run_config_checkpoint_date = config["checkpoint_configuration"].get("checkpoint_date")
        sim_initial_date = sim_config["time"].get("initial_day")
        if run_config_checkpoint_date != sim_initial_date:
            print(check, "sim config initial_date {sim_initial_date} NOT equal to run config checkpoint_date {run_config_checkpoint_date}")

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

def copy_data(new_data_path, june_data_path=None):
    input_data_path = new_data_path / "input"
    covid_real_data_path = new_data_path / "covid_real_data"
    
    if june_data_path is None:
        june_data_path = june.paths.data_path

    #if input_data_path.exists() is False:
    #    shutil.copytree(june_data_path, new_data_path, dirs_exist_ok=True)
    #else:
    #    print("Skip data copy")
    
    return None    










