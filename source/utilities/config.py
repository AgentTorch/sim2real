# ---------------------------------------------------------
# config.py
# ---------------------------------------------------------

import regex as re

from omegaconf import OmegaConf as conf
from agent_torch.core.helpers import get_by_path

def read_var(state, var):
    return get_by_path(state, re.split("/", var))

def parse_type(type_str):
  type_info = [x.strip() for x in type_str.split(',')]

  if type_info[0] == 'array':
    shape, data_type = parse_type(','.join(type_info[2:]))
    return [int(type_info[1]), *shape], data_type
  else:
    return [], type_info[0]

def parse_metadata(pretty_config):
  return {
    'device': 'cpu',
    'num_episodes': 1,
    'num_steps_per_episode': pretty_config['simulation']['steps'],
    'num_substeps_per_step': len(pretty_config['substeps']),

    'calibration': False,
    'visualize': False
  }

def parse_environment(pretty_config):
  parsed_vars = {}

  for var, metadata in pretty_config['environment'].items():
    parsed_vars[var] = {}

    shape, data_type = parse_type(metadata['type'])
    shape = [1] if len(shape) == 0 else shape

    parsed_vars[var]['name'] = var
    parsed_vars[var]['learnable'] = False
    parsed_vars[var]['dtype'] = data_type
    parsed_vars[var]['shape'] = shape

    parsed_vars[var]['value'] = metadata['value']
    parsed_vars[var]['initialization_function'] = None

  return parsed_vars

def parse_agents(pretty_config):
  parsed_agents = {}

  for agent, metadata in pretty_config['agents'].items():
    parsed_props = {}

    for name, type_str in metadata['properties'].items():
      parsed_props[name] = {}
      shape, data_type = parse_type(type_str)
      shape = [1] if len(shape) == 0 else shape

      parsed_props[name]['name'] = name
      parsed_props[name]['learnable'] = False
      parsed_props[name]['dtype'] = data_type
      parsed_props[name]['shape'] = (metadata["count"], *shape)

      parsed_props[name]['initialization_function'] = {
        'generator': 'read_from_file',
        'arguments': {
          'file_path': {
            'name': 'file_path', 'learnable': False,
            'shape': [1],
            'value': f"config/data/{agent}/{name}.csv",
            'initialization_function': None
          }
        }
      }

    parsed_agents[agent] = {
      'number': metadata['count'],
      'properties': parsed_props
    }

  return parsed_agents

def parse_substeps(pretty_config):
  parsed_substeps = {}

  for index, substep in enumerate(pretty_config['substeps']):
    agent = substep['agent']
    observation = substep['observation']
    action = substep['action']
    transition = substep['transition']

    parsed_substeps[str(index)] = {
      'name': substep['name'],
      'active_agents': [agent],
      'observation': { agent: None },
      'policy': { agent: None },
      'transition': { agent: None },
    }

    if observation is not None:
      parsed_substeps[str(index)]['observation'][agent] = {
        observation['name']: {
          'generator': observation['func'],
          'arguments': None,
          'input_variables': dict(zip(
            [x.partition('/')[2] for x in observation['observes']],
            [x if x.startswith('environment/') else f'agents/{x}' for x in observation['observes']],
          )),
          'output_variables': observation['produces']
        }
      }
    if action is not None:
      parsed_substeps[str(index)]['policy'][agent] = {
        action['name']: {
          'generator': action['func'],
          'arguments': None,
          'input_variables': dict(zip(
            [x.partition('/')[2] for x in action['requires']],
            [x if x.startswith('environment/') else f'agents/{x}' for x in action['requires']],
          )),
          'output_variables': action['decides']
        }
      }
    if transition is not None:
      parsed_substeps[str(index)]['transition'] = {
        transition['name']: {
          'generator': transition['func'],
          'arguments': None,
          'input_variables': dict(zip(
            [x.partition('/')[2] for x in transition['updates']],
            [x if x.startswith('environment/') else f'agents/{x}' for x in transition['updates']],
          )),
          'output_variables': [x.partition('/')[2] for x in transition['updates']]
        }
      }

  return parsed_substeps

def parse_config(file_name):
  pretty_config = conf.load(file_name)
  parsed_config = { 'state': {} }

  parsed_config['simulation_metadata'] = parse_metadata(pretty_config)
  parsed_config['state']['environment'] = parse_environment(pretty_config)
  parsed_config['state']['agents'] = parse_agents(pretty_config)
  parsed_config['state']['objects'] = {}
  parsed_config['state']['network'] = {}
  parsed_config['substeps'] = parse_substeps(pretty_config)

  return parsed_config, pretty_config
