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


def set_require_grad(model, require_grad):
  """ A helper function that sets the require_grad field
    of parameters of the given model. """
  for param in model.parameters():
    param.require_grad = require_grad


def replace_classifier(arch, model, dataset, freeze=False, init=True):
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
    set_require_grad(model, False)

  mod = None  # the module of the replaced classifier
  if arch.startswith('resnet'):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    mod = model.fc
  elif arch == 'alexnet' or arch.startswith('vgg'):
    model.classifier[6] = nn.Linear(4096, num_classes)
    mod = model.classifier[6]
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

  if init:
    nn.init.kaiming_normal_(mod.weight)


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