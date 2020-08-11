from pathlib import Path
import numpy as np
import yaml

queue_to_max_cpus = {"cosma": 16, "cosma6": 16, "cosma7": 28, "jasmin": 20}
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
        stdout_dir="stdout",
        max_time="72:00:00",
        region="london",
        iteration=1,
        num_runs=250,
        output_path="june_results",
        parallel_tasks_path=default_parallel_tasks_path,
        runner_path=default_run_simulation_script,
    ):
        self.region = region
        self.jobs_per_node = jobs_per_node
        self.system = system
        self.queue = queue
        self.account = account
        self.iteration = iteration
        self.max_time = max_time
        self.max_cpus_per_node = queue_to_max_cpus[queue]
        self.parallel_tasks_path = Path(parallel_tasks_path)
        self.stdout_dir = Path(stdout_dir)
        self.runner_path = Path(runner_path)
        self.num_runs = num_runs
        self.output_path = Path(output_path) / self.region / f"iteration_{self.iteration:02}"
        self.config_path = config_path

    @classmethod
    def from_file(cls, config_path: str = default_config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        system_configuration = config["system_configuration"]
        system = system_configuration["name"]
        queue = system_configuration["queue"]
        max_time = system_configuration["max_time"]
        account = system_configuration["account"]
        if system_configuration["parallel_tasks_path"] == "default":
            parallel_tasks_path = default_parallel_tasks_path
        else:
            parallel_tasks_path = system_configuration["parallel_tasks_path"]
        if system_configuration["runner_path"] == "default":
            run_script = default_run_simulation_script
        else:
            run_script = system_configuration["runner_path"]
        jobs_per_node = system_configuration["jobs_per_node"]
        region = config["region"]
        iteration = config["iteration"]
        num_samples = config["parameter_configuration"]["number_of_samples"]
        return cls(
            config_path=config_path,
            jobs_per_node=jobs_per_node,
            system=system,
            queue=queue,
            max_time=max_time,
            account=account,
            parallel_tasks_path=parallel_tasks_path,
            runner_path=run_script,
            region=region,
            iteration=iteration,
            num_runs=num_samples
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
        script_lines = (
            [
                "#!/bin/bash -l",
                "",
                f"#SBATCH --ntasks {self.max_cpus_per_node}",
                f"#SBATCH -J {self.region}_{self.iteration}_{script_number:03d}",
                f"#SBATCH -o {stdout_name}.out",
                f"#SBATCH -e {stdout_name}.err",
                f"#SBATCH -p {self.queue}",
                f"#SBATCH -A {self.account}",
                f"#SBATCH --exclusive",
                f"#SBATCH -t {self.max_time}",
            ]
            + loading_python
            + [
                f"mpirun -np {index_high-index_low+1} {self.parallel_tasks_path.absolute()} {index_low} {index_high} \"python -u {self.runner_path.absolute()} {self.config_path} -i %d \"",
            ]
        )
        return script_lines

    def make_scripts(self):
        script_dir = self.output_path / "slurm_scripts"
        script_dir.mkdir(exist_ok=True, parents=True)
        number_of_scripts = int(np.ceil(self.num_runs / self.jobs_per_node))
        script_names = []
        for i in range(number_of_scripts):
            idx1 = i * self.jobs_per_node
            idx2 = (i + 1) * self.jobs_per_node - 1
            script_lines = self.make_script_lines(
                script_number=i, index_low=idx1, index_high=idx2
            )
            script_name = script_dir / f"{self.region}_{i:03}.sh"
            script_names.append(script_name)
            with open(script_name, "w") as f:
                for line in script_lines:
                    f.write(line + "\n")

        # make submission script
        with open(self.output_path / "submit_scripts.sh", "w") as f:
            f.write("#!/bin/bash" + "\n\n")
            for script_name in script_names:
                line = f"sbatch {script_name.absolute()}"
                f.write(line + "\n")


if __name__ == "__main__":
    ssm = SlurmScriptMaker()
    ssm.make_scripts()
