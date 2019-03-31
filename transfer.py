""" Transfer an ImageNet pre-trained model to 
    a target fine-grained recognition dataset.
"""
import os

from taylor_pruning.parser import create_cli_parser
from taylor_pruning.transfer import ModelTransfer

parser = create_cli_parser(prog='Transfer pre-trained models')
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def main():
  transfer = ModelTransfer(args)
  model = transfer.load_model()
  transfer.train(model)
  transfer.validate(model)


if __name__ == '__main__':
  main()