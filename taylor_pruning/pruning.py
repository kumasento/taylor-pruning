""" Pruning related functions.

Reference - https://jacobgil.github.io/deeplearning/pruning-deep-learning
"""
import re
import time
import functools
import logging
module_logger = logging.getLogger('pruning')
module_logger.setLevel(logging.INFO)

import torch
import torch.nn as nn
from torch.autograd import Variable

from taylor_pruning.transfer import ModelTransfer
from taylor_pruning.utils import AverageMeter


def get_act_hook(mod, inp, out, mod_name=None, act_map=None):
  """ The forward hook to collect activation. """
  assert isinstance(mod_name, str)
  assert isinstance(act_map, dict)

  act_map[mod_name] = out


def get_grad_hook(mod, grad_in, grad_out, mod_name=None, grad_map=None):
  """ The hook to collect gradient. """
  assert isinstance(mod_name, str)
  assert isinstance(grad_map, dict)
  assert len(grad_out) == 1

  grad_map[mod_name] = grad_out[0]


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

  hooks = {}

  # register hooks
  for name, mod in model.named_modules():
    # skip modules unnecessary for pruning
    if not isinstance(mod, nn.Conv2d) and not isinstance(mod, nn.Linear):
      continue

    hooks[name] = []
    hooks[name].append(
        mod.register_forward_hook(
            functools.partial(get_act_hook, act_map=act_map, mod_name=name)))
    logger.debug('Registered act hook for module "{}"'.format(name))
    hooks[name].append(
        mod.register_backward_hook(
            functools.partial(get_grad_hook, grad_map=grad_map, mod_name=name)))
    logger.debug('Registered grad hook for module "{}"'.format(name))

  return hooks


def compute_taylor_criterion(act, grad):
  """ Compute the taylor criterion Eq. (8) of a single module. """
  assert isinstance(act, torch.Tensor)
  assert isinstance(grad, torch.Tensor)
  assert act.shape == grad.shape

  # turn both into a tensor of (batch, channels, img_h, img_w)
  if len(act.shape) == 2:  # especially for nn.Linear
    act = act.view([*act.shape, 1, 1])
    grad = grad.view([*grad.shape, 1, 1])

  with torch.no_grad():
    crit = torch.mul(grad, act)
    crit = crit.view([*crit.shape[:2], -1])
    crit = crit.mean(dim=2)  # average spatially
    crit = torch.abs(crit)
    crit = crit.mean(dim=0)  # average across all mini-batches

  return crit


class ModelPruner(ModelTransfer):
  """ Implemented pruning related utilities. """

  def compute_taylor_criterion(self, model, use_cuda=True):
    """ Compute the taylor criterion for every activation in the model. """
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    crit_map = {}

    end = time.time()
    for i, (x, target) in enumerate(self.train_loader):
      data_time.update(time.time() - end)

      grad_map, act_map = {}, {}
      hooks = register_hooks(model, act_map, grad_map)

      if use_cuda:
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

      y = model(x)
      loss = self.criterion(y, target)
      loss.backward()

      # collect criterion
      for key in grad_map:
        act, grad = act_map[key], grad_map[key]
        crit = compute_taylor_criterion(act, grad)
        crit = crit.cpu()

        if key not in crit_map:
          crit_map[key] = [crit]
        else:
          crit_map[key].append(crit)

      # remove hooks
      for hooks_ in hooks.values():
        for hook in hooks_:
          hook.remove()

      del grad_map
      del act_map

      batch_time.update(time.time() - end)
      end = time.time()

      if i % self.args.print_freq == 0:
        print('Batch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i,
                  len(self.train_loader),
                  batch_time=batch_time,
                  data_time=data_time))

    for key in crit_map:
      crit_map[key] = torch.mean(torch.stack(crit_map[key]), dim=0)

    return crit_map
