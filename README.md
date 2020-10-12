# JUNE RUNS



## Software to set up simulations of the JUNE code.



## Requirements

These instructions assume an up-to-date working JUNE installation. Please refer to the JUNE repository for details on how to set it up.

The ``june_runs`` specific requirements can be installed with

```bash
pip install -r requirements.txt
```

## Installation

``june_runs`` can be installed with

```bash
pip install -e .
```



## Usage

The required steps to run a batch of simulations in a cluster is:

### 1. Parameter file

An example of a parameter file can be found in `configuration/run_sets/quick_examples/jasmin.yaml` . Let us analyses it part by part.

The ``system_configuration`` part specifies the cluster to use, the memory per job (a typical England job would take 80-100 GB), and the number of cpus per job that we want:

```yaml
system_configuration:
  system_to_use: jasmin
  memory_per_job: 10 # GB
  cpus_per_job: 10
  extra_header_lines: []
  extra_module_lines: ["source activate @june_runs_path/june_venv"]
  extra_command_lines: ["echo \"job done\""]
```
we also have options to append commands to the script header, module loading, or at the bottom where the actual commands are.

Next is a small line explaining why are we running this set of simulations.

```yaml
purpose_of_the_run: "this is an example run."
```
and we can also fix the seed for reproducibility.
```yaml
random_seed: "random"
```

This is the part where we specify the paths. First we give a ``run_name``, if the ``save_path`` is set to ``"auto"`` then the working directory and the results will be saved in the ``june_runs/run_name``path. Don't worry! If the folder already exists then they will be saved to ``run_name_1``, etc. so that no work is lost.
The ``world_path`` should point to the world file we want to run on. Finally, the baseline paths are the paths to the baseline interaction and policy configs. Parameters of these files might be altered using the ``parameter_configuration`` below. A virtual environment to load can also be specified, keep in mind that this is the path to the environment so in this case it will resolve to ``source june_runs_path/june_venv/bin/activate``.
Path configuration also supports place holders with the use of ``@``.

```yaml
paths_configuration: # use @ as placeholder
  run_name: "example_run" # will be used to store results
  june_runs_path: "auto"
  save_path: "auto" # if auto, it will be june_runs_path/run_name (it won't override anything)
  world_path: "@june_runs_path/june_worlds/tests.hdf5" 
  # if these are defults, they will be picked from the june_runs_path/configuration/default_baseline_configs
  baseline_interaction_path: "default"
  basline_policy_path: "@june_runs_path/configuration/default_baseline_configs/policy.yaml"
  simulation_config_path: "default"
  virtual_env_path: "@june_runs_path/june_venv"
```


Finally, we need to tell the runner which parameter should it vary across all runs. There is a variety of sampling techniques available: ``grid``, ``regular_grid``, and ``latin_hypercube``. In this case, we run a lth, and all the parameters that are given as a list of two numbers are interpreted as the bounds of the hypercube dimension. If a parameter is given as a scalar, then that parameter is fixed across all runs.

There is a special feature in ``policies`` which allow the user to define a soft-hard lockdown transition, specifying the date and the relative strength of the lockdowns.

The number of days to run the simulation for is specified in ``n_days``.

```yaml
parameter_configuration:
  parameters_to_run: "all"
  sampling_type: latin_hypercube # available: [latin_hypercube, grid, regular_grid]
  parameters:
    n_samples: 10
    n_days: 10
    interaction:
      betas:
        pub       : [0.01, 0.25]
        grocery   : [0.01, 0.25]
        cinema    : [0.01, 0.25]
        city_transport : [0.01, 0.5]
        inter_city_transport : [0.01, 0.5]
        hospital  : [0.05, 0.5]
        care_home : [0.05, 1.]
        company   : [0.05, 0.5]
        school    : [0.05, 0.5]
        household : [0.05, 0.5]
        university: [0.01, 0.5]
      alpha_physical : [1.8,3.0]
      susceptibilities:
        "0-13" : 0.5
        "13-150" : 1.0
  
    infection:
      asymptomatic_ratio: [0.05, 0.4]
      seed_strength : [0.5,1.3]
      infectivity_profile: xnexp
  
    policies:
      lockdown:
        soft_lockdown_date: 2020-03-16
        hard_lockdown_date: 2020-03-24
        lockdown_ratio: 0.5 # relative strength of lockdowns
        hard_lockdown_policy_parameters:
          social_distancing:
            overall_beta_factor: [0.3, 0.9]
          quarantine: 
            overall_compliance: [0.2, 0.8]

```

### 2. Creating the results and submission directory.

Once we are happy with our parameter file, we can create the working directory using

```
python setup_run.py -c configuration/run_sets/quick_examples/jasmin.yaml
```

In this case, a folder named ``example_run`` will be created. This directory contains 3 sub-directories:
- Data: The data that was used to run the code. It's just there for reproducibility in the case we want to reproduce a run.
- Runs: Directory containing sub-directories for every run.
- Results: Output directory where the result summaries and records of the simulation will be stored.

The most important directory is the runs folder, which contains the following:
- ``run_000``
- ``run_001``
- ``...``
- ``stdout``
- ``submit_all.sh``


The ``stdout`` folder is where all the standard output / error will be stored.
Each run is represented as a folder ``run_xxx``, which contains a ``parameters.json`` file with the parameters varied for that specific run. It also contains it's own slurm/pbs/lsf script.

### 3. Submitting the jobs

To submit the jobs, the most convenient way is to use the ``submit_all.sh`` script:

```
bash example_run/runs/submit_all.sh
```

A nice trick to monitor the jobs is to use the ``tail`` command

```
tail -f example_run/runs/stdout*.out
```

and in case we fear errors
```
tail -f example_run/runs/stdout*.err
```

### 4. Getting the results.

At the end of the simulation, all summaries should have been stored in ``example_run/summaries``.
