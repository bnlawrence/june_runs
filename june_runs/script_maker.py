import yaml
from pathlib import Path

supported_systems = ["cosma5", "cosma6", "cosma7", "jasmin", "archer"]

from june_runs.paths import configuration_path


class ScriptMaker:
    """
    Class to make scripts for submission systems in clusters.
    Aims to support Slurm and PBS.
    """

    def __init__(
        self,
        system: str,
        run_directory: str,
        job_name: str = "june",
        memory_per_job: int = 100,
        cpus_per_job: int = 32,
        number_of_jobs=250,
        extra_header_lines = None,
        extra_module_lines = None,
        extra_command_lines =None,
    ):
        if system not in supported_systems:
            raise ValueError(f"System {system} not supported yet.")
        self.run_directory = Path(run_directory)
        self.stdout_directory = self.run_directory / "stdout"
        self.stdout_directory.mkdir(exist_ok=True, parents=True)
        self.job_name = job_name
        self.system_configuration = self._load_system_configuration(system)
        self.nodes_required = self.calculate_number_of_nodes(
            memory_per_job=memory_per_job,
            cpus_per_job=cpus_per_job,
            number_of_jobs=number_of_jobs,
        )
        self.cpus_per_job = cpus_per_job
        self.number_of_jobs = number_of_jobs
        self.extra_header_lines = extra_header_lines
        self.extra_module_lines = extra_module_lines
        self.extra_command_lines = extra_command_lines

    def _load_system_configuration(self, system):
        system_configuration_path = configuration_path / f"system/{system}.yaml"
        with open(system_configuration_path, "r") as f:
            system_configuration = yaml.load(f, Loader=yaml.FullLoader)
        return system_configuration

    def _get_script_dir(self, script_number):
        return self.run_directory / f"run_{script_number:03d}"

    def calculate_number_of_nodes(self, memory_per_job, cpus_per_job, number_of_jobs):
        cores_per_node = self.system_configuration["cores_per_node"]
        memory_per_node = self.system_configuration["memory_per_node"]
        total_number_of_cpus = cpus_per_job * number_of_jobs
        total_memory = memory_per_job * number_of_jobs
        cpu_nodes = total_number_of_cpus / cores_per_node
        memory_nodes = total_memory / memory_per_node
        return max(cpu_nodes, memory_nodes)

    def make_submission_script(self, script_number):
        header = self.make_script_header(script_number)
        modules_to_load = self.make_script_modules()
        command = self.make_python_command(script_number)
        return header + ["\n"] +  modules_to_load + ["\n"] + command

    def make_running_script(self, script_number):
        script_dir = self._get_script_dir(script_number)
        parameters_path = script_dir / "parameters.json"
        python_script = [
            "from june_runs import Runner\n",
            f"runner = Runner(\"{parameters_path}\")",
            "runner.run()",
        ]
        return python_script

    def make_script_header(self, script_number):
        queue = self.system_configuration["queue"]
        account = self.system_configuration["account"]
        max_time = self.system_configuration["max_time"]
        scheduler = self.system_configuration["scheduler"]
        stdout_path = self.stdout_directory / f"run_{script_number:03d}"
        if scheduler == "slurm":
            header = [
                "#!/bin/bash -l",
                "",
                f"#SBATCH --ntasks {self.cpus_per_job}",
                f"#SBATCH -J {self.job_name}_{script_number:03d}",
                f"#SBATCH -p {queue}",
                f"#SBATCH -A {account}",
                f"#SBATCH -o {stdout_path}.out",
                f"#SBATCH -e {stdout_path}.err",
                f"#SBATCH -t {max_time}",
            ]
        elif scheduler == "pbs":
            header = [
                "#!/bin/bash -l",
                "",
                f"#PBS -N {self.job_name}_{script_number:03d}",
                f"#PBS -l procs={self.cpus_per_job}",
                f"#PBS -l walltime={max_time}",
                f"#PBS -q {queue}",
                f"#PBS -A {account}",
                f"#PBS -o {stdout_path}.out",
                f"#PBS -e {stdout_path}.err",
            ]
        elif scheduler == "lsf":
            header = [
                "#!/bin/bash -l",
                "",
                f"#BSUB -n {self.cpus_per_job}",
                f"#BSUB -J {self.job_name}_{script_number:03d}",
                f"#BSUB -q {queue}",
                f"#BSUB -P {account}",
                f"#BSUB -o {stdout_path}.out",
                f"#BSUB -e {stdout_path}.err",
                f"#BSUB -x",
                f"#BSUB -W {max_time}",
            ]
        else:
            raise ValueError(f"Scheduler {scheduler} not yet supported.")
        if self.extra_header_lines:
            header += self.extra_header_lines
        return header

    def make_script_modules(self):
        modules = ["module purge"] + [
            f"module load {module}"
            for module in self.system_configuration["modules_to_load"]
        ]
        if self.extra_module_lines:
            modules += self.extra_module_lines
        return modules

    def make_python_command(self, script_number):
        script_path = self._get_script_dir(script_number)
        python_script_path = script_path / "run.py"
        python_command = [
            f"mpirun -np {self.cpus_per_job} python3 -u {python_script_path}"
        ]
        if self.extra_command_lines:
            python_command += self.extra_command_lines
        return python_command

    def write_scripts(self):
        script_paths = []
        for i in range(self.number_of_jobs):
            submission_script = self.make_submission_script(i)
            running_script = self.make_running_script(i)
            save_dir = self._get_script_dir(i)
            assert save_dir.is_dir()
            script_path = save_dir / "submit.sh"
            script_paths.append(script_path)
            with open(script_path, "w") as f:
                for line in submission_script:
                    f.write(line + "\n")
            with open(save_dir / "run.py", "w") as f:
                for line in running_script:
                    f.write(line + "\n")
        # make scrit to submit all jobs
        submit_all_script = self.make_submit_all_script(script_paths)
        with open(self.run_directory / "submit_all.sh", "w") as f:
            for line in submit_all_script:
                f.write(line + "\n")

    def make_submit_all_script(self, script_paths):
        script = ["#!/bin/bash -l \n"]
        scheduler = self.system_configuration["scheduler"]
        if scheduler == "slurm":
            submission_command = "sbatch"
        elif scheduler == "pbs":
            submission_command = "qsub"
        elif scheduler == "lsf":
            submission_command = "bsub"
        else:
            raise ValueError(f"Scheduler {scheduler} not yet supported.")
        for path in script_paths:
            script += [f"{submission_command} {path}"]
        return script

