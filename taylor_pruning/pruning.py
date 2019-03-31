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

from taylor_pruning.transfer import ModelTransfer


def get_act_hook(mod, inp, out, mod_name=None, act_map=None):
  """ The forward hook to collect activation. """
  assert isinstance(mod_name, str)
  assert isinstance(act_map, dict)

  act_map[mod_name] = out


def get_grad_hook(mod, grad_in, grad_out, mod_name=None, grad_map=None):
  """ The hook to collect gradient. """
  assert isinstance(mod_name, str)
  assert isinstance(grad_map, dict)

  grad_map[mod_name] = grad_out


def register_hooks(model, act_map, grad_map, logger=None):
  """ Register hooks on model to collect activations and
    gradients to act_map and grad_map respectively.
    
    Args:
      model(nn.Module):
      act_map(dict): activation map
      grad_map(dict): gradient map
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
    logger.debug('Register backward hook for module "{}": {}'.format(name, mod))
    mod.register_backward_hook(
        functools.partial(get_grad_hook, grad_map=grad_map, mod_name=name))


def compute_taylor_criterion(act, grad):
  """ Compute the taylor criterion Eq. (8) of a single module. """
  assert isinstance(act, torch.Tensor)
  assert isinstance(grad, torch.Tensor)
  assert act.shape == grad.shape

  # turn both into a tensor of (batch, channels, img_h, img_w)
  if len(act.shape) == 2:  # especially for nn.Linear
    act = act.view([*act.shape, 1, 1])
    grad = grad.view([*grad.shape, 1, 1])

  crit = torch.mul(grad, act)
  crit = crit.view([*crit.shape[:2], -1])
  crit = crit.mean(dim=2)  # average spatially
  crit = torch.abs(crit)
  crit = crit.mean(dim=0)  # average across all mini-batches

  return crit


class ModelPruner(ModelTransfer):
  """ Implemented pruning related utilities. """
  pass
