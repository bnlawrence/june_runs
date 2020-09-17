from pathlib import Path
import numpy as np
import yaml
import subprocess

from sklearn.model_selection import ParameterGrid

from .parameter_generator import _read_parameters_to_run, _get_len_parameter_grid
from .utils import parse_paths, config_checks, git_checks 
import june

queue_to_max_cpus = {"cosma": 16, "cosma6": 16, "cosma7": 28, "jasmin": 20, "cosma-prince" : 16}
default_parallel_tasks_path = (
    Path(__file__).parent.parent / "parallel_tasks/parallel_tasks"
)
default_run_simulation_script = Path(__file__).parent.parent / "run_simulation.py"
default_config_path = Path(__file__).parent.parent / "run_configs/config_example.yaml"


class SlurmScriptMaker:
    def __init__(
        self,
        config_path=default_config_path,
        jobs_per_node=4,
        system="cosma",
        queue="cosma",
        account="durham",
        email_notifications=False,
        email_address=None,
        max_time="72:00:00",
        region="london",
        iteration=1,
        config_type="latin_hypercube",
        num_runs=250,
        parameters_to_run="all",
        output_path="june_results",
        stdout_path=None,
        jobname=None,
        parallel_tasks_path=default_parallel_tasks_path,
        runner_path=default_run_simulation_script,
    ):
        self.region = region
        self.jobs_per_node = jobs_per_node
        self.system = system
        self.queue = queue
        self.account = account
        self.email_notifications = email_notifications
        self.email_address = email_address
        self.iteration = iteration
        self.max_time = max_time
        self.num_runs = num_runs
        if num_runs is not None:
            self.parameters_to_run = _read_parameters_to_run(parameters_to_run, num_runs)
        self.max_cpus_per_node = queue_to_max_cpus[queue]
        self.parallel_tasks_path = Path(parallel_tasks_path)
        self.runner_path = Path(runner_path)
        self.output_path = output_path
        self.config_path = Path(config_path)
        if stdout_path is None or stdout_path == Path("default"):
            self.stdout_dir = self.output_path / "stdout"
        else:
            self.stdout_dir = stdout_path
        if jobname is None or jobname == "default":
            self.jobname = self.region
        else:
            self.jobname = jobname
        self.stdout_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file(cls, config_path: str = default_config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config_checks(config)
        git_checks()
        system_configuration = config["system_configuration"]
        system = system_configuration["name"]
        queue = system_configuration["queue"]
        max_time = system_configuration["max_time"]
        account = system_configuration["account"]
        email_notifications = system_configuration["email_notifications"]
        email_address = system_configuration["email_address"]
        if system_configuration["parallel_tasks_path"] == "default":
            parallel_tasks_path = default_parallel_tasks_path
        else:
            parallel_tasks_path = system_configuration["parallel_tasks_path"]
        if system_configuration["runner_path"] == "default":
            run_script = default_run_simulation_script
        else:
            run_script = system_configuration["runner_path"]
        if "jobname" in system_configuration:
            jobname = system_configuration["jobname"]            
        else:
            jobname = None
        jobs_per_node = system_configuration["jobs_per_node"]
        region = config["region"]
        iteration = config["iteration"]
        if config["parameter_configuration"].get("config_type") == "grid":
            num_runs = _get_len_parameter_grid(
                config["parameter_configuration"]
            )
        else:
            num_runs = config["parameter_configuration"].get("number_of_samples")
        if config["parameter_configuration"].get("parameters_to_run") is None: 
            parameters_to_run = "all"
        else:
            parameters_to_run = config["parameter_configuration"]["parameters_to_run"]
        paths = parse_paths(
            config["paths_configuration"], region=region, iteration=iteration
        )
        return cls(
            config_path=config_path,
            jobs_per_node=jobs_per_node,
            system=system,
            queue=queue,
            max_time=max_time,
            account=account,
            email_notifications=email_notifications,
            email_address=email_address,
            parallel_tasks_path=parallel_tasks_path,
            runner_path=run_script,
            region=region,
            iteration=iteration,
            num_runs=num_runs,
            parameters_to_run=parameters_to_run,
            output_path=paths["results_path"],
            stdout_path=paths["stdout_path"],
            jobname = jobname
        )
    
    def make_script_lines(self, script_number, index_low, index_high):
        stdout_name = (
            self.stdout_dir / f"{self.region}_{self.iteration}_{script_number:03d}"
        )
        if self.system == "jasmin":
            loading_python = [
                "module purge",
                "module load eb/OpenMPI/gcc/4.0.0",
                "module load jaspy/3.7/r20200606",
                "source /gws/nopw/j04/covid_june/june_venv/bin/activate",
            ]
        elif self.system == "cosma":
            loading_python = [
                f"module purge",
                f"module load python/3.6.5",
                f"module load gnu_comp/7.3.0",
                f"module load hdf5",
                f"module load openmpi/3.0.1",
            ]
        else:
            raise ValueError(f"System {self.system} is not supported")
        if (self.email_notifications) and (self.email_address is not None):
            email_lines = [
                f"#SBATCH --mail-type=BEGIN,END",
                f"#SBATCH --mail-user={self.email_address}"
            ]
        else:
            email_lines = []
                
        python_cmd = f"python3 -u {self.runner_path.absolute()} {self.config_path.absolute()} -i %d "

        script_lines = (
            [
                "#!/bin/bash -l",
                "",
                f"#SBATCH --ntasks {self.max_cpus_per_node}",
                f"#SBATCH -J {self.jobname}_{self.iteration}_{script_number:03d}",
                f"#SBATCH -o {stdout_name}.out",
                f"#SBATCH -e {stdout_name}.err",
                f"#SBATCH -p {self.queue}",
                f"#SBATCH -A {self.account}",
                f"#SBATCH --exclusive",
                f"#SBATCH -t {self.max_time}",
            ]
            + email_lines
            + loading_python
            + [
                f'mpirun -np {index_high-index_low+1} {self.parallel_tasks_path.absolute()} {index_low} {index_high} "{python_cmd}"',
            ]
        )
        return script_lines

    def make_scripts(self):
        script_dir = self.output_path / "slurm_scripts"
        script_dir.mkdir(exist_ok=True, parents=True)
        number_of_scripts = int(np.ceil(len(self.parameters_to_run)/ self.jobs_per_node))
        script_names = []
        for i in range(number_of_scripts):
            idx1 = i * self.jobs_per_node
            idx2 = min((i + 1) * self.jobs_per_node - 1, len(self.parameters_to_run) - 1) 
            script_lines = self.make_script_lines(
                script_number=i, index_low=idx1, index_high=idx2
            )
            script_name = script_dir / f"{self.region}_{i:03}.sh"
            script_names.append(script_name)
            with open(script_name, "w") as f:
                for line in script_lines:
                    f.write(line + "\n")

        # make submission script
        submit_scripts_path = self.output_path / "submit_scripts.sh"
        with open(submit_scripts_path, "w") as f:
            f.write("#!/bin/bash" + "\n\n")
            for i,script_name in enumerate(script_names):
                line = f"sbatch {script_name.absolute()}"
                f.write(line + "\n")
                if i == 0:
                    try:
                        print_path = script_name.relative_to(Path.cwd())
                    except:
                        print_path = script_name
                    print(f'scripts written to eg.:\n    {print_path}\n')

        try:
            print_path = submit_scripts_path.relative_to(Path.cwd())
        except:
            print_path = submit_scripts_path
        print(f'submit all_scripts with:\n    \033[35mbash {print_path}\033[0m')


if __name__ == "__main__":
    ssm = SlurmScriptMaker()
    ssm.make_scripts()
