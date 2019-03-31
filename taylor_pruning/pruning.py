""" Pruning related functions.

Reference - https://jacobgil.github.io/deeplearning/pruning-deep-learning
"""
import re
import functools
import logging
module_logger = logging.getLogger('pruning')
module_logger.setLevel(logging.DEBUG)

import torch
import torch.nn as nn
from torch.autograd import Variable


def get_act_hook(mod, inp, out, mod_name=None, act_map=None):
  """ The forward hook to collect activation. """
  assert isinstance(mod_name, str)
  assert isinstance(act_map, dict)

  act_map[mod_name] = out


def get_grad_hook(grad, var_name=None, grad_map=None):
  """ The hook to collect gradient. """
  assert isinstance(var_name, str)
  assert isinstance(grad_map, dict)

  grad_map[var_name] = grad


def register_hooks(model,
                   act_map,
                   grad_map,
                   param_name_pattern='weight',
                   logger=None):
  """ Register hooks on model to collect activations and
    gradients to act_map and grad_map respectively.
    
    Args:
      model(nn.Module):
      act_map(dict): activation map
      grad_map(dict): gradient map
      param_name_pattern(str): regex pattern for parameters to be registered
      logger(Logger): log information
  """

  assert isinstance(model, nn.Module)
  assert isinstance(act_map, dict)
  assert isinstance(grad_map, dict)

  if logger is None:
    logger = module_logger

  # register hooks
  for name, mod in model.named_modules():
    # skip modules unnecessary for pruning
    if not isinstance(mod, nn.Conv2d) and not isinstance(mod, nn.Linear):
      continue

    logger.debug('Register forward hook for module "{}": {}'.format(name, mod))
    mod.register_forward_hook(
        functools.partial(get_act_hook, act_map=act_map, mod_name=name))

    for var_name, var in mod.named_parameters():
      # register hooks for variables to collect gradients
      if re.search(param_name_pattern, var_name):
        assert isinstance(var, Variable)

        # NOTE: name of the parameter is prefixed by module name
        var_name = name + '.' + var_name

        logger.debug('Register hook for parameter "{}" of shape {}'.format(
            var_name, var.shape))
        var.register_hook(
            functools.partial(
                get_grad_hook, var_name=var_name, grad_map=grad_map))
