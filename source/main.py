# ---------------------------------------------------------
# main.py
# ---------------------------------------------------------

from agent_torch.core import Registry, Runner
from agent_torch.core.helpers import read_from_file

from utilities import print_header, print_footer, parse_config
from substeps import *

CONFIG_FILE = 'config/solar.yaml'

# say hi :)
print_header()

# parse the configuration we have written, turn it into a
# dict that agent_torch can parse, and return that.
print('parsing configuration...')

parsed, config = parse_config(CONFIG_FILE)

# setup the components of the simulation
print('setting up simulation...')

registry = Registry()
registry.register(read_from_file, 'read_from_file', 'initialization')

runner = Runner(parsed, registry)
runner.init()

# finally, run the simulation
print('running simulation...')

steps = config['simulation']['steps']
for _ in range(steps):
  runner.step(1)

# say bye :)
print_footer()
