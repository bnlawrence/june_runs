from runner import Runner
from runner import SlurmScriptMaker
import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser(description="main running script for June")
parser.add_argument("run_configuration", help="file with running configuration")
parser.add_argument("-f", "--file")
parser.add_argument("-i", "--index", help="parameter latin hypercube index", type=int)
parser.add_argument("--setup", help="Sets up running directory", action="store_true")
args = parser.parse_args()

parameter_index = args.index
setup = bool(args.setup)
config_path = args.run_configuration

runner = Runner.from_file(config_path)
if setup:
    slurm_script_maker = SlurmScriptMaker.from_file(
        runner.parameter_generator.parameters_to_run, config_path
    )
    slurm_script_maker.make_scripts()
else:
    if parameter_index is None:
        raise ValueError("provide parameter index")
#    simulator = runner.generate_simulator(parameter_index)
#    simulator.run()
    runner.extract_summaries(parameter_index=parameter_index)
