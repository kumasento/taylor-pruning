""" CLI for pruning """
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)

import torch

from taylor_pruning.parser import create_cli_parser
from taylor_pruning.pruning import ModelPruner

parser = create_cli_parser(prog='Transfer pre-trained models')
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def main():
  logging.info('==> Loading model ...')
  pruner = ModelPruner(args)
  model = pruner.load_model()

  # logging.info('==> Validating the loaded model ...')
  # pruner.validate(model)

  logging.info('==> Collecting gradients ...')
  crit_map = pruner.compute_taylor_criterion(
      model, use_cuda=torch.cuda.is_available())
  print(crit_map)


if __name__ == '__main__':
  main()