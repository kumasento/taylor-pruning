""" Utility functions. 

Most of the functions are borrowed from pytorch/examples.
"""

import random
import shutil
import math
import time
import json
import warnings
import sys
import os
import logging
from collections import OrderedDict
logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and
                     callable(models.__dict__[name]))

#######################################
# Model                             #
#######################################


def load_model(arch,
               dataset,
               resume=None,
               pretrained=False,
               update_model_fn=None,
               update_state_dict_fn=None,
               use_cuda=True,
               data_parallel=True,
               checkpoint_file_name='checkpoint.pth.tar',
               **kwargs):
  """ Load a model.
  
    You can use update_model_fn and update_state_dict_fn
    to configure those models undesirable.

  Args:
    arch(str): model architecture
    dataset(str): dataset name
    resume(str): initialise model from a checkpoint file
    pretrained(bool): directly load pre-trained models from
      TorchVision
    update_model_fn(func): a hook to update the model
    update_state_dict_fn(func): a hook to update the state_dict
    use_cuda(bool): put the model on GPU
    data_parallel(bool): use the DataParallel module
  
  Returns:
    A nn.Module represents the loaded model.
  """
  # construct the model
  num_classes = get_num_classes(dataset)
  model = models.__dict__[arch](pretrained=pretrained)

  # update model if required
  if update_model_fn:
    logging.debug('update_model_fn is provided.')
    model = update_model_fn(model)

  # initialise the model on GPU if necessary
  if use_cuda:
    if data_parallel:
      model = torch.nn.DataParallel(model)
    model = model.cuda()

  if resume:  # load from checkpoint
    if pretrained:
      raise ValueError(
          'You cannot specify pretrained to True and resume not None.')

    assert isinstance(resume, str)

    # update the resume if it points to a directory
    if os.path.isdir(resume):
      resume = os.path.join(resume, checkpoint_file_name)
      logging.debug(
          'Resume was given as a directory, updated to: {}'.format(resume))

    # now resume should be a valid file.
    assert os.path.isfile(resume)

    checkpoint = torch.load(resume)  # load

    # get the state dict
    state_dict = checkpoint['state_dict']
    if update_state_dict_fn:
      state_dict = update_state_dict_fn(state_dict)

    # initialize model
    model.load_state_dict(state_dict)

  return model


#######################################
# Module                              #
#######################################


def find_in_parent(mod, par):
  """ Get the index of mod in par.children.
  
  Args:
    mod(nn.Module)
    par(nn.Module)
  Returns:
    An index, -1 indicates not found.
  """
  assert isinstance(mod, nn.Module)
  assert isinstance(par, nn.Module)

  # locate mod in par.children
  mods = list(par.children())

  for i, mod_ in enumerate(mods):
    if mod_ is mod:
      return i

  return -1


def insert_after(ins, mod, par, name=None):
  """ Insert a module after a given module.
  
  Args:
    ins(nn.Module): module to insert
    mod(nn.Module):
    par(nn.Module): parent module of mod
  Returns:
    None
  """
  idx = find_in_parent(mod, par)
  if idx == -1:
    raise ValueError('module {} (mod) is not a child of {} (par).'.format(
        mod, par))

  mods = list(par._modules.items())
  if name is None:  # generate a new name
    name = mods[idx][0] + '_ins'
  assert isinstance(name, str)

  mods.insert(idx + 1, (name, ins))  # insert after idx
  par._modules = OrderedDict(mods)


#######################################
# Dataset                             #
#######################################


def get_dataset(is_training=False, dataset_dir=None):
  """ Create dataset object with ImageNet augmentation/normalisation. """
  assert isinstance(dataset_dir, str) and os.path.isdir(dataset_dir)

  img_size = 224
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  # convention
  train_dir = os.path.join(dataset_dir, 'train')
  val_dir = os.path.join(dataset_dir, 'val')

  logger.debug('Creating dataset loader for {} ...'.format(
      'training' if is_training else 'validation'))

  if is_training:
    return datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
  else:
    return datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]))


def get_num_classes(dataset):
  """ NOTE: Need to update this when adding a new dataset. """
  if dataset == 'imagenet':
    return 1000
  if dataset == 'cub200':
    return 200

  raise ValueError('dataset cannot be recognised: {}'.format(dataset))


def get_data_loader(dataset_dir, batch_size, workers=12, is_training=False):
  """ Create data loader.
  
  Args:
    dataset_dir(str): path to a root directory for dataset, must has
      'train' and 'val' sub directories.
    batch_size(int): batch size
    worker(int): number of data loaders
    is_training(bool): whether in training mode or not
  """
  return data.DataLoader(
      get_dataset(is_training=is_training, dataset_dir=dataset_dir),
      batch_size=batch_size,
      shuffle=is_training,
      num_workers=workers,
      pin_memory=True  # for better performance
  )


#######################################
# Logging                             #
#######################################


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


#######################################
#  Training and Eval                  #
#######################################


def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          print_freq=100,
          gpu=None,
          **kwargs):
  """ Major training function. """
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    lr = adjust_learning_rate(
        epoch, optimizer, batch=i, batches=len(train_loader), **kwargs)

    if gpu is not None:
      input = input.cuda(gpu, non_blocking=True)
    target = target.cuda(gpu, non_blocking=True)

    # compute output
    output = model(input)
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'LR: {lr:.4f}\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch,
                i,
                len(train_loader),
                lr=lr,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5))

  return losses.avg, top1.avg


def validate(val_loader, model, criterion, print_freq=100, gpu=None):
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
      # if gpu is not None:
      input = input.cuda(gpu, non_blocking=True)
      target = target.cuda(gpu, non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(acc1[0], input.size(0))
      top5.update(acc5[0], input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % print_freq == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  i,
                  len(val_loader),
                  batch_time=batch_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

  return losses.avg, top1.avg


def save_checkpoint(state,
                    is_best,
                    checkpoint_dir,
                    file_name='checkpoint.pth.tar'):
  path = os.path.join(checkpoint_dir, file_name)
  torch.save(state, path)

  if is_best:
    shutil.copyfile(path, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def adjust_learning_rate(epoch,
                         optimizer,
                         state=None,
                         schedule=None,
                         epochs=None,
                         batch=None,
                         batches=None,
                         base_lr=None,
                         gamma=None,
                         lr_type=None):
  """ Adjust the LR value in state. """
  assert state is not None

  if lr_type == 'cosine':
    assert isinstance(epochs, int)
    assert isinstance(batches, int)
    assert isinstance(batch, int)
    assert isinstance(base_lr, float)

    tot = epochs * batches
    cur = (epoch % epochs) * batches + batch  # TODO why mod?
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * cur / tot))

  elif lr_type is None or lr_type == 'multistep':
    assert state is not None
    assert schedule is not None
    assert isinstance(gamma, float)
    assert isinstance(batch, int)

    lr = state['lr']
    if epoch in schedule and batch == 0:
      lr *= gamma

  else:
    raise ValueError('lr_type {} cannot be recognised'.format(lr_type))

  state['lr'] = lr
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  return lr
