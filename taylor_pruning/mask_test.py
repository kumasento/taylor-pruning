import unittest

import torch

from taylor_pruning.mask import ActMask


class MaskTest(unittest.TestCase):
  """ Test the correctness of the ActMask module """

  def test_forward(self):
    mask = ActMask(32)
    x = torch.rand((32, 32, 32, 32))
    y = mask(x)
    self.assertTrue(torch.allclose(x, y))

    mask.mask[1] = 0.0
    y = mask(x)
    self.assertFalse(torch.allclose(x, y))
    x[:, 1, :, :] = 0.0
    self.assertTrue(torch.allclose(x, y))


if __name__ == '__main__':
  unittest.main()
