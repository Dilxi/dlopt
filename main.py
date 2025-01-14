#!/usr/bin/env python

import argparse
import logging
import os
from typing import Type

from torch.utils.data import BatchSampler, Dataset
from torchvision import transforms as transforms

from sampler import *
from setup import Setup
from vgg import VGG

DATA_DIR = 'datasets'
os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
  parser = argparse.ArgumentParser(
    description='Deep Learning Dataset Pruning Experiment',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # basic
  parser.add_argument('--batch-size', type=int, default=128, help='number of samples per batch (default=1)')
  parser.add_argument('--iters', type=int, default=1000, help='number of iterations (deafult=1000)')
  parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device to use')
  parser.add_argument('--runs', type=int, default=5, help='number of runs for each sampler (default=5)')

  return parser.parse_args()


def main():
  args = parse_args()
  device = torch.device(args.device)
  train_set, test_set = get_mnist()

  # hyperparams
  lr = 0.1
  decay = 1e-5
  gamma = 0.1

  # helper function that wraps a sampler
  batched = lambda sampler: BatchSampler(sampler, batch_size=args.batch_size)

  results = []
  for i in range(args.runs):
    samplers = [
      batched(RandomSampler(train_set, args.batch_size)),
      batched(ImportanceSampler(train_set, gamma, args.batch_size)),
      batched(ShuffledEpochSampler(train_set, args.batch_size)),
      batched(SequentialEpochSampler(train_set, args.batch_size)),
      # batched(ClasswiseSampler(train_set, args.batch_size)),
    ]

    for sampler in samplers:
      name = sampler.sampler.__class__.__name__
      logger.info(f'Running sampler: {name}, run_id: {i}')

      model = get_vgg()
      model = model.to(device)

      experiment = Setup(train_set, test_set, model, sampler, lr, decay, args.iters, device)
      res = experiment.run()
      res['sampler'] = name
      res['run_id'] = i
      results.append(res)

  import json
  json.dump(results, open(os.path.join('results.json'), 'w'))


def indexed(cls: Type[Dataset]):
  """helper function that adds indices to the samples of a dataset"""

  def __getitem__(self, index):
    data, target = cls.__getitem__(self, index)
    return data, target, index

  return type(cls.__name__, (cls,), dict(__getitem__=__getitem__))


def get_mnist():
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Pad(2)  # pad images to minimum vgg11 dimensions
  ])
  mnist_train = indexed(MNIST)(root=DATA_DIR, train=True, download=True, transform=transform)
  mnist_test = indexed(MNIST)(root=DATA_DIR, train=False, download=True, transform=transform)
  return mnist_train, mnist_test


def get_vgg():
  return VGG('VGG11')


if __name__ == '__main__':
  main()
