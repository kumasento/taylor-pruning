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
    self.mask = nn.Parameter(torch.ones(num_channels), requires_grad=False)

  def forward(self, x):
    """ Element-wise multiply mask with x. """
    mask = self.mask
    if len(x.shape) == 4:
      mask = mask.view([x.shape[1], 1, 1])
    return torch.mul(x, mask)

  def extra_repr(self):
    s = '{num_channels}'
    return s.format(**self.__dict__)