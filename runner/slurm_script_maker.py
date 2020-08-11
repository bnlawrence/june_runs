from pathlib import Path
import numpy as np

queue_to_max_cpus = {"cosma": 16, "cosma6": 16, "cosma7": 28, "jasmin": 20}
default_parallel_tasks_path = (
    Path(__file__).parent.parent / "parallel_tasks/parallel_tasks"
)
default_runner_path = Path(__file__).parent / "runner.py"


class SlurmScriptMaker:
    def __init__(
        self,
        jobs_per_node=4,
        system="cosma",
        queue="cosma",
        account="durham",
        stdout_dir="stdout",
        job_base_name="london",
        max_time="72:00:00",
        iteration=1,
        parallel_tasks_path=default_parallel_tasks_path,
        runner_path=default_runner_path,
    ):
        self.jobs_per_node = jobs_per_node
        self.system = system
        self.queue = queue
        self.account = account
        self.job_base_name = job_base_name
        self.iteration = iteration
        self.max_time = max_time
        self.max_cpus_per_node = queue_to_max_cpus[queue]
        self.parallel_tasks_path = Path(parallel_tasks_path)
        self.stdout_dir = Path(stdout_dir)
        self.runner_path = Path(runner_path)

    def make_script_lines(self, script_number, index_low, index_high, n_cpus=-1):
        if n_cpus == -1:
            n_cpus = self.max_cpus_per_node
        stdout_name = (
            self.stdout_dir
            / f"{self.job_base_name}_{self.iteration}_{script_number:03d}"
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
                f"#SBATCH --ntasks {n_cpus}",
                f"#SBATCH -J {self.job_base_name}_{self.iteration}_{script_number:03d}",
                f"#SBATCH -o {stdout_name}.out",
                f"#SBATCH -e {stdout_name}.err",
                f"#SBATCH -p {self.queue}",
                f"#SBATCH -A {self.account}",
                f"#SBATCH --exclusive",
                f"#SBATCH -t {self.max_time}",
            ]
            + loading_python
            + [
                f"mpirun -np {index_high-index_low+1} {self.parallel_tasks_path.absolute()} {index_low} {index_high} {self.runner_path.absolute()} %d ",
            ]
        )
        return script_lines

    def make_scripts(self, output_dir: str, number_of_runs=250, n_cpus=-1):
        output_path = Path(output_dir)
        script_dir = output_path / "slurm_scripts"
        script_dir.mkdir(exist_ok=True, parents=True)
        output_path.mkdir(exist_ok=True, parents=True)
        number_of_scripts = int(np.ceil(number_of_runs / self.jobs_per_node))
        script_names = []
        for i in range(number_of_scripts):
            idx1 = i * self.jobs_per_node
            idx2 = (i + 1) * self.jobs_per_node - 1
            script_lines = self.make_script_lines(
                script_number=i, index_low=idx1, index_high=idx2, n_cpus=n_cpus
            )
            script_name = script_dir / f"script_{i:03}.sh"
            script_names.append(script_name)
            with open(script_name, "w") as f:
                for line in script_lines:
                    f.write(line + "\n")

        # make submission script
        with open(output_path / "submit_scripts.sh", "w") as f:
            f.write("#!/bin/bash" + "\n\n")
            for script_name in script_names:
                line = f"sbatch {script_name}"
                f.write(line + "\n")


if __name__ == "__main__":
    ssm = SlurmScriptMaker()
    ssm.make_scripts("testing_scripts", 15)
