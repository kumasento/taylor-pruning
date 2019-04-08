""" Unit test for utils. """

import unittest

import torch
import torch.nn as nn

from taylor_pruning import utils


class TestUtils(unittest.TestCase):
  """ unit-tests for taylor_pruning.utils """

  def test_find_in_parent(self):
    """ test find_in_parent """
    par = nn.Sequential()
    mod1 = nn.Conv2d(32, 32, 3)
    mod2 = nn.Conv2d(32, 32, 3)
    par.add_module('conv1', mod1)

    self.assertEqual(utils.find_in_parent(mod1, par), 0)
    self.assertEqual(utils.find_in_parent(mod2, par), -1)

    par.add_module('conv2', mod2)
    self.assertEqual(utils.find_in_parent(mod2, par), 1)

  def test_insert_after(self):
    """ Test the module for insertion. """
    par = nn.Sequential()
    mod1 = nn.Conv2d(32, 32, 3)
    mod2 = nn.Conv2d(32, 32, 3)

    par.add_module('conv1', mod1)
    utils.insert_after(mod2, mod1, par)
    self.assertEqual(utils.find_in_parent(mod2, par), 1)


if __name__ == '__main__':
  unittest.main()