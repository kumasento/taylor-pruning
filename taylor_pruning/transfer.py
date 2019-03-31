""" Utilities for transferring pre-trained models. """

import random
import shutil
import math
import time
import json
import warnings
import sys
import os
import logging
logger = logging.getLogger('transfer')
logger.setLevel(logging.INFO)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import taylor_pruning.utils as utils
from taylor_pruning.model_runner import ModelRunner


def set_requires_grad(model, requires_grad):
  """ A helper function that sets the requires_grad field
    of parameters of the given model. """
  for param in model.parameters():
    param.requires_grad = requires_grad


# def init_module(mod):
#   """ init the parameters of a given module """
#   if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
#     nn.init.kaiming_normal_(mod.weight)
#     if mod.bias is not None:
#       nn.init.zeros_(mod.bias)
#   set_require_grad(mod, True)  # HACK: we need to restore the trainability


def replace_classifier(arch, model, dataset, freeze=False):
  """ Replace the final classifier based on the dataset.
  
  Args:
    arch(str): network architecture
    model(nn.Module): the model to be updated
    dataset(str): name of the dataset, to get the number of classes
    freeze(bool): freeze all CONV feature layers
    init(bool): whether to initialise weights by kaiming normal
  """
  if dataset == 'imagenet':
    logger.warn('Given dataset is imagenet, no classifier will be replaced.')
    return

  # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
  num_classes = utils.get_num_classes(dataset)
  if freeze:
    set_requires_grad(model, False)

  mod = None  # the module of the replaced classifier
  if arch.startswith('resnet'):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    mod = model.fc

  elif arch == 'alexnet' or arch.startswith('vgg'):
    model.classifier[6] = nn.Linear(4096, num_classes)
    mod = model.classifier
    # HACK
    for param in model.classifier.parameters():
      param.requires_grad = True

  elif arch.startswith('squeezenet'):
    model.classifier[1] = nn.Conv2d(
        512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    mod = model.classifier[1]

  elif arch.startswith('densenet'):
    model.classifier = nn.Linear(1024, num_classes)
    mod = model.classifier

  elif arch.startswith('mobilenet'):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    mod = model.fc

  else:
    raise ValueError('ARCH={} cannot be recognized.'.format(arch))


class ModelTransfer(ModelRunner):
  """ Inherited model runner that focuses on transfer pre-trained models. """

  def create_update_model_fn(self):
    """ Create the hook function. """

    def update_model(model):
      """ Simply call the replace_classifier function. """
      replace_classifier(
          self.args.arch, model, self.args.dataset, freeze=self.args.freeze)

      return model

    return update_model

  def load_model(self, **kwargs):
    """ By default, creates an update_model_fn to replace
      the classifiers. """
    update_model_fn = self.create_update_model_fn()

    return super().load_model(update_model_fn=update_model_fn, **kwargs)