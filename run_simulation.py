from runner import Runner
import sys


runner = Runner.from_file()
simulator = runner.generate_simulator(int(sys.argv[1]))
simulator.run()

