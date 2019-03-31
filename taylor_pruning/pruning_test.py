""" Unit tests for pruning. """

import unittest
import functools

import torch
import torch.nn as nn

from taylor_pruning.pruning import register_hooks


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

    class Flatten(nn.Module):

      def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

    class ConvNet(nn.Sequential):
      """ A very small CNN for testing. """

      def __init__(self):
        super().__init__()

        self.add_module('conv1', nn.Conv2d(3, 32, 3, stride=1, padding=1))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('pool1', nn.MaxPool2d(2, 2))
        self.add_module('conv2', nn.Conv2d(32, 32, 3, stride=1, padding=1))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('pool2', nn.MaxPool2d(2, 2))
        self.add_module('flatten', Flatten())
        self.add_module('fc', nn.Linear(8 * 8 * 32, 10))

    conv_net = ConvNet()
    act_map = {}
    grad_map = {}

    register_hooks(conv_net, act_map, grad_map)

    x = torch.rand((32, 3, 32, 32))
    y = conv_net(x)
    target = torch.LongTensor(32).random_(0, 10)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y, target)
    loss.backward()

    self.assertEqual(set(act_map.keys()), set(['conv1', 'conv2', 'fc']))
    self.assertEqual(set(grad_map.keys()), set(['conv1', 'conv2', 'fc']))


if __name__ == '__main__':
  unittest.main()