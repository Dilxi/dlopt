"""Experiment setups for MNIST and CIFAR10 datasets"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, vmap, grad
from torch.optim import SGD
from torch.utils.data import DataLoader

from sampler import BatchSampler

logger = logging.getLogger(__name__)


class Trainer:
  def __init__(self, model, optimizer, criterion, sampler: BatchSampler, device):
    self.model = model
    self.optimizer = optimizer
    self.criterion = criterion
    self.sampler = sampler
    self.device = device
    self.metrics = dict(training_loss=[], test_accuracy=[])

    if self.sampler.requires_grad():
      params = {k: v.detach() for k, v in self.model.named_parameters()}
      buffers = {k: v.detach() for k, v in self.model.named_buffers()}
      ft_compute_sample_grads = vmap(grad(self._loss_func), in_dims=(None, None, 0, 0))
      self.get_grads = lambda data, target: ft_compute_sample_grads(params, buffers, data, target).values()

  def _loss_func(self, params, buffers, data, target):
    """Returns a function call to compute loss for a single sample."""
    # add an extra dimension to separate different samples in the batch
    batch = data.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(self.model, (params, buffers), (batch,))
    loss = F.cross_entropy(predictions, targets)
    return loss

  def train_step(self, data, target):
    self.model.train()
    self.optimizer.zero_grad()

    output = self.model(data)
    loss = self.criterion(output, target)
    loss.backward()

    grad_norm = None
    if self.sampler.requires_grad():
      # Get weight gradients before optimizer step
      grads = [param.grad for _, param in self.model.named_parameters() if param.grad is not None]
      grad_norm = torch.sqrt(torch.sum(torch.stack([torch.norm(g, p='fro') ** 2 for g in grads]))).cpu()

    self.optimizer.step()
    return loss, grad_norm

  def train(self, train_loader, test_loader, num_iterations: int):
    for step, (data, target, indices) in enumerate(train_loader, 1):
      data, target = data.to(self.device), target.to(self.device)
      loss, grad_norm = self.train_step(data, target)
      self.metrics['training_loss'].append(loss.item())

      if self.sampler.requires_grad():
        self.sampler.update(grad_norm, indices)

      if step % 100 == 0:
        logger.info(f'Iteration {step}: Loss {loss.item():.6f}')

      if step % 1000 == 0:
        self.metrics[f'test_accuracy'].append(self.test_accuracy(test_loader))
        logger.info(f'Test accuracy {self.metrics["test_accuracy"][-1]:.6f}')

      if step == num_iterations:
        break

    self.metrics['final_accuracy'] = self.test_accuracy(test_loader)
    logger.info(f'Final test accuracy: {self.metrics["final_accuracy"]:.6f}')

  def test_accuracy(self, test_loader):
    self.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for data, target, _ in test_loader:
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    self.model.train()

    return correct / total


class Setup:
  CRITERION = nn.CrossEntropyLoss()
  OPTIMIZER_CLASS = SGD

  def __init__(self, train_set, test_set, model, sampler: BatchSampler, lr, decay, iters, device):
    self.train_set = train_set
    self.test_set = test_set
    self.model = model
    self.sampler = sampler
    self.lr = lr
    self.decay = decay
    self.iters = iters
    self.device = device

  def run(self):
    optimizer = self.OPTIMIZER_CLASS(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.decay)

    train_loader = DataLoader(self.train_set,
                              batch_sampler=self.sampler,
                              num_workers=2,
                              pin_memory=True if self.device == 'cuda' else False)
    test_loader = DataLoader(self.test_set,
                             batch_size=self.sampler.batch_size,
                             num_workers=2,
                             pin_memory=True if self.device == 'cuda' else False)

    trainer = Trainer(self.model, optimizer, self.CRITERION, self.sampler, self.device)
    trainer.train(train_loader, test_loader, self.iters)
    return trainer.metrics
