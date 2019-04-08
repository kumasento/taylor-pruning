""" Pruning related functions.

Reference - https://jacobgil.github.io/deeplearning/pruning-deep-learning
"""
import os
import re
import time
import functools
import logging
import copy
from collections import namedtuple, OrderedDict, Counter
module_logger = logging.getLogger('pruning')
module_logger.setLevel(logging.INFO)

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from taylor_pruning.transfer import ModelTransfer
from taylor_pruning.utils import AverageMeter
from taylor_pruning.mask import ActMask

ActRank = namedtuple('ActRank', ['mod_name', 'channel', 'crit'])


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

  with torch.no_grad():  # this is critical for the memory usage
    crit = torch.mul(grad, act)  # element-wise
    crit = crit.view([*crit.shape[:2], -1])  # flatten spatial dims
    crit = crit.mean(dim=2)  # average spatially
    crit = torch.abs(crit)  # take abs for the final crit value
    crit = crit.mean(dim=0, keepdim=True)  # average across all mini-batches

  return crit


def rank_act(crit_map):
  """ Rank channels in activations by their criterion.

    NOTE: All the modules that are affected by replacement should be
      returned by this function.
  
    Args:
      crit_map(dict): computed criterion
    Returns:
      A list of ActRank(<mod_name, channel_id, crit_val>)
      sorted in increasing order
  """
  ranking = []

  for mod_name in crit_map:
    crit = crit_map[mod_name].detach().cpu().numpy().tolist()
    # generate the list of tuple for every module
    ranking.extend(list([ActRank(mod_name, i, c) for i, c in enumerate(crit)]))

  return list(sorted(ranking, key=lambda k: k.crit))  # sort by crit value


def create_crit_map(act_map, grad_map):
  """ Create crit_map from act_map and grad_map. """
  assert isinstance(act_map, dict)
  assert isinstance(grad_map, dict)

  # compute and store criterion
  crit_map = {}
  for key in grad_map:
    act, grad = act_map[key], grad_map[key]
    crit = compute_taylor_criterion(act, grad)
    crit = crit.detach().cpu()

    if key not in crit_map:
      crit_map[key] = crit
    else:
      crit_map[key] = torch.cat((crit_map[key], crit), dim=0)

  return crit_map


def get_mod_channels(mod, out=False):
  """ Collect the number of channels. """
  if isinstance(mod, nn.Conv2d):
    return mod.out_channels if out else mod.in_channels
  elif isinstance(mod, nn.Linear):
    return mod.out_features if out else mod.in_features
  else:
    raise TypeError('Cannot recognise module with type: {}'.format(type(mod)))


def update_mod_channels(mod, channels, out=False):
  if isinstance(mod, nn.Conv2d):
    if out:
      mod.out_channels = channels
    else:
      mod.in_channels = channels
  elif isinstance(mod, nn.Linear):
    if out:
      mod.out_features = channels
    else:
      mod.in_features = channels
  else:
    raise TypeError('Cannot recognise module with type: {}'.format(type(mod)))


def clone_module(mod, in_channels, out_channels):
  """ Create a new nn.Conv2d or nn.Linear module based on
    the original module provided. """
  if isinstance(mod, nn.Conv2d):
    mod_ = nn.Conv2d(
        in_channels,
        out_channels,
        mod.kernel_size,
        stride=mod.stride,
        padding=mod.padding,
        bias=mod.bias is not None,
        dilation=mod.dilation)
  elif isinstance(mod, nn.Linear):
    mod_ = nn.Linear(in_channels, out_channels, bias=mod.bias is not None)
  else:
    raise TypeError('Cannot recognise module with type: {}'.format(type(mod)))

  return mod_


def get_channels_to_prune(ranking,
                          num_channels,
                          least_num_channels=8,
                          excludes=None):
  """ Return the channels that will be pruned. 
  
  Args:
    ranking(list): a list of ActRank, already SORTED.
    num_channels(int): the amount of channels that will be pruned.
    least_num_channels(int): at least leave each activation
      this amount of channels
    excludes(list): a list of mod names that will be excluded
      for pruning.
  Returns:
    A list of ActRank, representing channels to be pruned.
  """
  assert isinstance(least_num_channels, int) and least_num_channels >= 0

  # compute the number of channels for each activation
  # labelled by mod_name
  nc_map = Counter([rnk.mod_name for rnk in ranking])

  if excludes is None:
    excludes = []

  cnt = 0
  to_prune = []
  for rk in ranking:
    if nc_map[rk.mod_name] <= least_num_channels or rk.mod_name in excludes:
      continue

    to_prune.append(rk)
    nc_map[rk.mod_name] -= 1  # update the remaining number of channels

    cnt += 1
    if cnt >= num_channels:
      break

  return to_prune


def find_act_mask(mod, par):
  """ Find the ActMask followed by mod in its parent children.

  Args:
    mod(nn.Module):
    par(nn.Module): parent module of mod
  Returns:
    The ActMask module or None 
  """
  # locate mod in par.children
  mods = list(par.children())

  for i, mod_ in enumerate(mods):
    if mod_ is mod:
      if i < len(mods) - 1 and isinstance(mods[i + 1], ActMask):
        return mods[i + 1]
      else:
        return None

  raise ValueError('mod {} is not a children of {}.'.format(mod, par))


def insert_or_update_act_mask(mod, par, ouc):
  """ Insert a new ActMask after mod or update its contents.
  
  Args:
    mod(nn.Module):
    par(nn.Module): parent module of mod
    ouc(list): a list of channel indices to be remained
  Returns:
    None
  """
  act_mask = find_act_mask(mod, par)
  if act_mask is None:
    num_channels = get_mod_channels(mod, out=True)
    act_mask = ActMask(num_channels)


def prune_by_taylor_criterion(model,
                              crit_map,
                              num_channels_per_prune=1,
                              logger=None):
  """ Prune the given model by the crit_map.

  NOTE: need update for batch norm
  NOTE: need update for non-sequential connectivity

  Args:
    model(nn.Module): model to be pruned
    crit_map(dict): the computed criterion map
    num_channels_per_prune(int): number of channels will be pruned per
      iteration
  Returns:
    the model, updated in-place 
  """
  assert isinstance(model, nn.Module)
  assert isinstance(crit_map, dict)
  assert isinstance(num_channels_per_prune, int)
  assert num_channels_per_prune >= 1

  if logger is None:
    logger = module_logger

  # compute the ranking
  ranking = rank_act(crit_map)
  chl_map = OrderedDict()
  for tup in ranking:  # initialise every module name
    chl_map[tup.mod_name] = []

  # figure out all the channels to be pruned
  cls_name = list(model.named_modules())[-1][0]  # name of the classifier module

  # NOTE:  we should skip the last classifier
  ranking = [rnk for rnk in ranking if rnk.mod_name != cls_name]
  to_prune = get_channels_to_prune(ranking, num_channels_per_prune)

  for tup in to_prune:
    if tup.mod_name not in chl_map:
      chl_map[tup.mod_name] = [tup.channel]
    else:
      chl_map[tup.mod_name].append(tup.channel)

  # reorder the channel map by sequence in model
  # and build the mod map
  act_chl = []
  mod_map = OrderedDict()
  par_map = OrderedDict()  # record the parent module name
  mod_out = OrderedDict()
  for name, mod in model.named_modules():
    mod_map[name] = mod
    for child_name, child in mod.named_children():
      full_name = child_name if not name else '{}.{}'.format(name, child_name)
      par_map[full_name] = name

    if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
      mod_out[name] = get_mod_channels(mod, out=True)
      if name in chl_map:
        act_chl.append((name, chl_map[name]))
  del chl_map

  # iterate every activation
  # in each iteration, we remove those channels that
  # are marked to be pruned, by creating a new Conv2d
  # or Linear model and migrate the weights to the new
  # model.
  for i, (name, channels_to_prune) in enumerate(act_chl):
    mod = mod_map[name]
    pre = mod_map[act_chl[i - 1][0]] if i > 0 else None
    assert isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear)

    # collect the number of channels from the module
    # that outputs the current activation.
    in_channels = get_mod_channels(mod, out=False)
    out_channels = get_mod_channels(mod, out=True)

    # decide the channels will be remained
    ouc = [x for x in range(out_channels) if x not in act_chl[i][1]]
    if i == 0:
      inc = range(in_channels)
    else:
      # HACK: the interface between CONV-FC is hard to handle
      if isinstance(mod, nn.Linear) and isinstance(pre, nn.Conv2d):
        img_size = in_channels // mod_out[act_chl[i - 1][0]]
        inc = np.arange(in_channels).reshape((-1, img_size))
        inc_ = []
        for x in range(inc.shape[0]):
          if x not in act_chl[i - 1][1]:
            inc_.extend(inc[x, :].tolist())
        inc = inc_
      else:
        inc = [x for x in range(in_channels) if x not in act_chl[i - 1][1]]

    # NOTE: this is an older module replacement approach. We now prefer
    #   adding mask modules
    # mod_ = clone_module(mod, len(inc), len(ouc))
    # mod_.weight.data = mod.weight[ouc, :][:, inc]  # pruned weights
    # if mod_.bias is not None:
    #   mod_.bias.data = mod.bias[ouc]  # pruned biases
    # par_mod = mod_map[par_map[name]]
    # par_mod._modules[name.split('.')[-1]] = mod_
    # del mod  # explicitly remove this replaced module

  return model


class ModelPruner(ModelTransfer):
  """ Implemented pruning related utilities. """

  def __init__(self, args):
    super().__init__(args)

    self.base_dir = args.checkpoint
    if isinstance(self.base_dir, str):
      os.makedirs(self.base_dir, exist_ok=True)

  def update_criterion(self, crit_map, act_map, grad_map, hooks):
    """ Called after forward-backward in one pass.
    
      act_map and grad_map will be recycled
      hooks in `hooks` will be removed
    """
    assert isinstance(crit_map, dict)
    assert isinstance(act_map, dict)
    assert isinstance(grad_map, dict)
    assert isinstance(hooks, dict)  # NOTE: returned from register_hooks

    # compute and store criterion
    crit_map.update(create_crit_map(act_map, grad_map))

    # remove hooks
    for hooks_ in hooks.values():
      for hook in hooks_:
        hook.remove()

    del grad_map
    del act_map

  def compute_taylor_criterion(self, model, use_cuda=True, logger=None):
    """ Compute the taylor criterion for every activation in the model. """
    assert isinstance(model, nn.Module)

    if logger is None:
      logger = module_logger

    batch_time = AverageMeter()
    data_time = AverageMeter()
    crit_time = AverageMeter()

    # switch to train mode
    model.train()

    crit_map = {}

    end = time.time()
    for i, (x, target) in enumerate(self.train_loader):
      data_time.update(time.time() - end)

      # create hooks
      grad_map, act_map = {}, {}
      hooks = register_hooks(model, act_map, grad_map)

      if use_cuda:
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

      y = model(x)
      loss = self.criterion(y, target)
      loss.backward()

      # collect criterion
      # NOTE: here is the most time-consuming part (or maybe not)
      crit_start = time.time()
      self.update_criterion(crit_map, act_map, grad_map, hooks)
      crit_time.update(time.time() - crit_start)

      batch_time.update(time.time() - end)
      end = time.time()

      if i % self.args.print_freq == 0:
        print('[{0:3d}/{1:3d}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Crit {crit_time.val:.3f} ({crit_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i,
                  len(self.train_loader),
                  crit_time=crit_time,
                  batch_time=batch_time,
                  data_time=data_time))

    # summarise result
    logger.info('==> Summarising criterion from all batches ...')
    for key in crit_map:
      crit_map[key] = torch.mean(crit_map[key], dim=0)

    return crit_map

  def prune_loop(self, model, use_cuda=True, logger=None):
    """ Prune for the required number of iterations. """
    if logger is None:
      logger = module_logger

    for prune_iter in range(self.args.num_prune_iters):
      logger.info('==> Running pruning iteration [{:3d}/{:3d}] ...'.format(
          prune_iter, self.args.num_prune_iters))

      # create the checkpoint directory
      checkpoint = os.path.join(self.base_dir,
                                'prune_iter_{}'.format(prune_iter))
      os.makedirs(checkpoint, exist_ok=True)

      # declare the new checkpoint
      self.args.checkpoint = checkpoint
      # NOTE: this logger is only used for the internal logic of training
      self.logger = self.get_logger(self.args)

      model = self.prune(model, use_cuda=use_cuda, logger=logger)

  def prune(self, model, use_cuda=True, logger=None):
    """ A prune round. """
    if logger is None:
      logger = module_logger

    # collect the criterion map
    logger.info('==> Collecting criterion for all modules ...')
    crit_map = self.compute_taylor_criterion(
        model, use_cuda=use_cuda, logger=logger)

    # prune the model based on the criterion
    logger.info('==> Pruning the model ...')
    model = prune_by_taylor_criterion(
        model,
        crit_map,
        num_channels_per_prune=self.args.num_channels_per_prune,
        logger=logger)
    print(model)

    # train and validate this model
    logger.info('==> Fine-tuning the model ...')
    self.train(model, load_optim=False)
    self.validate(model)

    return model
