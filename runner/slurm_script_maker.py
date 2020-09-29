from pathlib import Path
import numpy as np
import yaml
import subprocess

from sklearn.model_selection import ParameterGrid

from .utils import parse_paths, config_checks, git_checks, copy_data

queue_to_max_cpus = {
    "cosma": 16,
    "cosma6": 16,
    "cosma7": 28,
    "jasmin": 20,
    "cosma-prince": 16,
}
default_parallel_tasks_path = (
    Path(__file__).parent.parent / "parallel_tasks/build/parallel_tasks"
)
default_run_simulation_script = Path(__file__).parent.parent / "run_simulation.py"
default_config_path = Path(__file__).parent.parent / "run_configs/config_example.yaml"


class SlurmScriptMaker:
    def __init__(
        self,
        config_path=default_config_path,
        cores_per_job=4,
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
        parameters_to_run="all",
        output_path="june_results",
        stdout_path=None,
        jobname=None,
        parallel_tasks_path=default_parallel_tasks_path,
        runner_path=default_run_simulation_script,
        virtual_env=None,
    ):
        self.region = region
        self.cores_per_job = cores_per_job
        self.jobs_per_node = jobs_per_node
        self.system = system
        self.queue = queue
        self.account = account
        self.email_notifications = email_notifications
        self.email_address = email_address
        self.iteration = iteration
        self.max_time = max_time
        self.parameters_to_run = parameters_to_run
        self.num_runs = len(self.parameters_to_run)
        self.max_cpus_per_node = queue_to_max_cpus[queue]
        self.parallel_tasks_path = Path(parallel_tasks_path)
        self.runner_path = Path(runner_path)
        self.output_path = output_path
        self.config_path = Path(config_path)
        self.virtual_env = virtual_env
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
    def from_file(cls, parameters_to_run, config_path: str = default_config_path):
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
        if "virtual_env" in system_configuration:
            virtual_env = system_configuration['virtual_env']
        else:
            virtual_env = None
        jobs_per_node = system_configuration["jobs_per_node"]
        cores_per_job = system_configuration["cores_per_job"]
        region = config["region"]
        iteration = config["iteration"]
        paths = parse_paths(
            config["paths_configuration"], region=region, iteration=iteration
        )
        copy_data(paths["data_path"])
        return cls(
            config_path=config_path,
            jobs_per_node=jobs_per_node,
            cores_per_job=cores_per_job,
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
            parameters_to_run=parameters_to_run,
            output_path=paths["results_path"],
            stdout_path=paths["stdout_path"],
            jobname=jobname,
            virtual_env=virtual_env
        )

    def make_script_lines(self, script_number, index_low, index_high, virtual_env=None):
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
                f"module load gnu-parallel",
            ]
        else:
            raise ValueError(f"System {self.system} is not supported")
        if virtual_env is not None:
            loading_python.append(virtual_env)
        if (self.email_notifications) and (self.email_address is not None):
            email_lines = [
                f"#SBATCH --mail-type=BEGIN,END",
                f"#SBATCH --mail-user={self.email_address}",
            ]
        else:
            email_lines = []
        ntasks = max(self.max_cpus_per_node, self.cores_per_job)
        slurm_header = [
            "#!/bin/bash -l",
            "",
            f"#SBATCH --ntasks {ntasks}",
            f"#SBATCH -J {self.jobname}_{self.iteration}_{script_number:03d}",
            f"#SBATCH -o {stdout_name}.out",
            f"#SBATCH -e {stdout_name}.err",
            f"#SBATCH -p {self.queue}",
            f"#SBATCH -A {self.account}",
            f"#SBATCH --exclusive",
            f"#SBATCH -t {self.max_time}",
        ]

        parallel_cmd = f"parallel -u --delay .2 -j {index_high-index_low+1}"
        python_cmd = f'"mpirun -np {self.cores_per_job} python3 -u {self.runner_path.absolute()} {self.config_path.absolute()} -i {{1}}"'
        full_cmd = [
            parallel_cmd + " " + python_cmd + f" ::: {{{index_low}..{index_high}}}"
        ]
        script_lines = slurm_header + email_lines + loading_python + full_cmd
        return script_lines

    def make_scripts(self):
        script_dir = self.output_path / "slurm_scripts"
        script_dir.mkdir(exist_ok=True, parents=True)
        number_of_scripts = int(
            np.ceil(len(self.parameters_to_run) / self.jobs_per_node)
        )
        script_names = []
        for i in range(number_of_scripts):
            idx1 = i * self.jobs_per_node
            idx2 = min(
                (i + 1) * self.jobs_per_node - 1, len(self.parameters_to_run) - 1
            )
            script_lines = self.make_script_lines(
                script_number=i, index_low=idx1, index_high=idx2, virtual_env=self.virtual_env
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
            for i, script_name in enumerate(script_names):
                line = f"sbatch {script_name.absolute()}"
                f.write(line + "\n")
                if i == 0:
                    try:
                        print_path = script_name.relative_to(Path.cwd())
                    except:
                        print_path = script_name
                    print(f"scripts written to eg.:\n    {print_path}\n")

        try:
            print_path = submit_scripts_path.relative_to(Path.cwd())
        except:
            print_path = submit_scripts_path
        print(f"submit all_scripts with:\n    \033[35mbash {print_path}\033[0m")


if __name__ == "__main__":
    ssm = SlurmScriptMaker()
    ssm.make_scripts()
