""" Unit tests for pruning. """

import unittest
import functools
import copy

import numpy as np
import torch
import torch.nn as nn

from taylor_pruning.mask import ActMask
from taylor_pruning.pruning import *


class Flatten(nn.Module):

  def forward(self, x):
    shape = torch.prod(torch.tensor(x.shape[1:])).item()
    return x.view(-1, shape)


class ConvNet(nn.Sequential):
  """ A very small CNN for testing. """

  def __init__(self):
    super().__init__()

    self.add_module('conv1', nn.Conv2d(3, 32, 3, stride=1, padding=1))
    self.add_module('bn1', nn.BatchNorm2d(32))
    self.add_module('relu1', nn.ReLU(inplace=True))
    self.add_module('pool1', nn.MaxPool2d(2, 2))
    self.add_module('conv2', nn.Conv2d(32, 32, 3, stride=1, padding=1))
    self.add_module('bn2', nn.BatchNorm2d(32))
    self.add_module('relu2', nn.ReLU(inplace=True))
    self.add_module('pool2', nn.MaxPool2d(2, 2))
    self.add_module('flatten', Flatten())
    self.add_module('fc', nn.Linear(8 * 8 * 32, 10))


class TestPruning(unittest.TestCase):

  def test_register_hook(self):
    """ Example use case of register_hook. """

    def hook(grad, var_name=None, grad_map=None):
      """ store the grad into grad_map """
      assert grad_map is not None
      assert isinstance(var_name, str)

      # record
      grad_map[var_name] = grad

    grad_map = {}

    x = torch.rand((32, 32))
    x.requires_grad = True
    x.register_hook(functools.partial(hook, var_name='x', grad_map=grad_map))

    y = x.sum()
    y.register_hook(functools.partial(hook, var_name='y', grad_map=grad_map))
    y.backward(torch.ones(1))

    self.assertTrue(torch.allclose(torch.ones(1), grad_map['y']))
    self.assertTrue(torch.allclose(torch.ones((32, 32)), grad_map['x']))

  def test_register_forward_hook(self):
    """ Experiment with the register_forward_hook function. """

    class Square(nn.Module):

      def forward(self, x):
        return x**2

    def forward_hook(mod, inp, out, mod_name=None, act_map=None):
      """ Put the module activation to act_map """
      assert isinstance(mod_name, str)
      assert act_map is not None

      act_map[mod_name] = out

    act_map = {}

    sq = Square()
    sq.register_forward_hook(
        functools.partial(forward_hook, mod_name='sq', act_map=act_map))
    x = torch.rand((4, 4))
    y = sq(x)

    self.assertTrue(torch.allclose(y, act_map['sq']))

  def test_register_hooks(self):
    """ Test the register_hooks function from the pruning module """
    conv_net = ConvNet()
    act_map = {}
    grad_map = {}

    register_hooks(conv_net, act_map, grad_map)

    target = torch.LongTensor(32).random_(0, 10)
    criterion = nn.CrossEntropyLoss()

    x = torch.rand((32, 3, 32, 32))
    y = conv_net(x)
    loss = criterion(y, target)
    loss.backward()

    x = torch.rand((32, 3, 32, 32))
    y = conv_net(x)
    loss = criterion(y, target)
    loss.backward()

    self.assertEqual(set(act_map.keys()), set(['conv1', 'conv2', 'fc']))
    self.assertEqual(set(grad_map.keys()), set(['conv1', 'conv2', 'fc']))
    # even we call twice, the act_map will be refreshed
    self.assertEqual(act_map['conv1'].shape[0], 32)

  def test_compute_taylor_criterion(self):
    """ test the compute function """

    act = torch.rand((32, 3, 28, 28))
    grad = torch.rand((32, 3, 28, 28))

    crit = compute_taylor_criterion(act, grad)
    assert len(crit.shape) == 2
    assert crit.shape[1] == 3  # number of channels

  def test_rank_act(self):
    """ test the rank_act function. """
    crit_map = {  # two modules output 32 and 16 channels
        'mod1': torch.rand(32),
        'mod2': torch.rand(16)
    }

    ranking = rank_act(crit_map)
    self.assertEqual(len(ranking), 32 + 16)
    self.assertLess(ranking[0].crit, ranking[-1].crit)

  def test_prune_by_taylor_criterion(self):
    """ Test the prune_by_taylor_criterion function. """
    conv_net = ConvNet()
    act_map = {}
    grad_map = {}

    register_hooks(conv_net, act_map, grad_map)

    target = torch.LongTensor(32).random_(0, 10)
    criterion = nn.CrossEntropyLoss()

    x = torch.rand((32, 3, 32, 32))
    y = conv_net(x)
    loss = criterion(y, target)
    loss.backward()

    crit_map = create_crit_map(act_map, grad_map)
    prune_by_taylor_criterion(conv_net, crit_map, num_channels_per_prune=16)

    # count number of pruned channels
    nz = 0
    for mod in conv_net.modules():
      if isinstance(mod, ActMask):
        nz += np.count_nonzero(mod.mask.cpu().numpy() == 0)
    self.assertEqual(nz, 16)

    y2 = conv_net(x)
    self.assertFalse(torch.allclose(y, y2))  # results are different

    # Try to prune again
    act_map = {}
    grad_map = {}
    register_hooks(conv_net, act_map, grad_map)
    y2 = conv_net(x)
    loss = criterion(y2, target)
    loss.backward()

    crit_map = create_crit_map(act_map, grad_map)
    prune_by_taylor_criterion(conv_net, crit_map, num_channels_per_prune=16)
    y3 = conv_net(x)
    self.assertFalse(torch.allclose(y2, y3))  # results are different

    # find collect all zeros from all masks
    nz = 0
    for mod in conv_net.modules():
      if isinstance(mod, ActMask):
        nz += np.count_nonzero(mod.mask.cpu().numpy() == 0)
    self.assertEqual(nz, 32)

  def test_get_channels_to_prune(self):
    """ Test whether the get_channels_to_prune function works well. """

    ranking = [
        ActRank('a', 0, 0.1),
        ActRank('a', 1, 0.2),
        ActRank('b', 0, 0.3),
        ActRank('b', 1, 0.35),
        ActRank('c', 0, 0.4),
        ActRank('c', 1, 0.6),
        ActRank('d', 0, 0.7),
        ActRank('d', 1, 0.8),
        ActRank('d', 2, 0.91),
    ]

    # Now, we remove 3 channels, and the threshold is 1, so that only the first
    # act of each module will be removed.
    # also 'c' should be skipped
    pc_map = get_channels_to_prune(
        ranking, 3, least_num_channels=1, excludes=['c'])

    self.assertEqual(pc_map['a'], set([0]))
    self.assertEqual(pc_map['b'], set([0]))
    self.assertEqual(pc_map['c'], set())
    self.assertEqual(pc_map['d'], set([0]))


if __name__ == '__main__':
  unittest.main()