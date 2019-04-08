""" Implement the mask on activations. """

import torch
import torch.nn as nn


class ActMask(nn.Module):
  """ Apply a 0-1 mask on the activation channels. """

  def __init__(self, num_channels):
    """ CTOR. """
    super().__init__()

    assert isinstance(num_channels, int)
    assert num_channels >= 1

    self.num_channels = num_channels
    self.mask = nn.Parameter(torch.ones(num_channels, requires_grad=False))

  def forward(self, x):
    """ Element-wise multiply mask with x. """
    mask = self.mask.view([1, x.shape[1], 1, 1])
    return torch.mul(x, mask)