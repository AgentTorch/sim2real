# ---------------------------------------------------------
# format.py
# ---------------------------------------------------------

import torch
from rich.console import Console
from rich.tree import Tree

def print_header():
  print()
  print('solar_model')
  print('-----------')

def print_footer():
  print()
  print('-----------')
  print()

def format_tensor(tensor):
  # Get the first 2 and last 2 values of the tensor and print '...' in between
  flattened = tensor.flatten()  # Flatten the tensor to a 1D array for easier manipulation
  total_values = flattened.size(0)

  if total_values <= 4:
    # If the tensor has 4 or fewer values, print all values
    return str(', '.join([str(x) for x in flattened.tolist()]))
  else:
    # Otherwise, print the first 2, '...', and the last 2 values
    return f"{', '.join([str(x) for x in flattened[:2].tolist()])}, ..., {', '.join([str(x) for x in flattened[-2:].tolist()])}"

def inspect_tensor(name, tensor_dict, parent=None):
  console = Console()
  orig_parent = parent
  tree = parent if parent is not None else Tree(f"[magenta]{name}[/magenta]")

  for key, value in tensor_dict.items():
    # If it's a nested dictionary, we call inspect_tensor_dict recursively
    if isinstance(value, dict):
      subtree = tree.add(f"[green]{key}[/green]")
      inspect_tensor(None, value, parent=subtree)
    elif isinstance(value, torch.Tensor):
      # Print tensor details
      formatted_tensor = format_tensor(value)
      tree.add(
        f"[cyan]{key}[/cyan] [magenta]{tuple(value.shape)}[/magenta] [green]{{{value.count_nonzero()}}}[/green]: {formatted_tensor}"
      )
    else:
      # Print non-tensor elements normally
      tree.add(f"[yellow]{key}[/yellow]: {value}")

  if orig_parent is None:
    console.print(tree)
  else:
    return tree
