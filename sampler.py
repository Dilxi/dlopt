import torch
from torch.utils.data import Sampler as TorchSampler
from torchvision.datasets import MNIST, CIFAR10


class Sampler(TorchSampler):
  def __init__(self, data_source: MNIST | CIFAR10, batch_size: int):
    self.data_source = data_source
    self.num_samples = len(data_source)
    # this is useful when we want to yield multiple indices at once, see: RandomSampler, ImportanceSampler
    self.batch_size = batch_size

  def __iter__(self):
    raise NotImplementedError

  def __len__(self):
    return self.num_samples

  def requires_grad(self):
    return False

  def update(self, values: torch.Tensor, indices: torch.Tensor):
    pass


class RandomSampler(Sampler):
  """Samples indices randomly with replacement"""

  def __iter__(self):
    yield from torch.randint(high=self.num_samples, size=(self.batch_size,)).tolist()


class ImportanceSampler(Sampler):
  """Samples indices with probability updated every iteration based on the norm of the loss gradients"""

  def __init__(self, data_source: MNIST | CIFAR10, gamma: float, batch_size: int):
    super().__init__(data_source, batch_size)
    self.weights = torch.ones(self.num_samples)
    self.weight_sum = torch.sum(self.weights).item()
    self.gamma = gamma

  def requires_grad(self):
    return True

  def __iter__(self):
    probs = self.weights / self.weight_sum
    yield from torch.multinomial(probs, self.batch_size, replacement=True).tolist()

  def update(self, values: torch.Tensor, indices: torch.Tensor):
    delta = (1 - self.gamma) * (values - self.weights[indices])
    self.weight_sum += torch.sum(delta).item()
    self.weights[indices] += delta


class EpochSampler(Sampler):
  """Base class for samplers that iterate over the entire dataset before using the same samples again"""

  def __init__(self, data_source: MNIST | CIFAR10, batch_size: int, shuffle=True):
    super().__init__(data_source, batch_size)
    self.shuffle = shuffle

  def __iter__(self):
    while True:
      if self.shuffle:
        yield from torch.randperm(self.num_samples).tolist()
      else:
        yield from range(self.num_samples)


class ShuffledEpochSampler(EpochSampler):
  """Shuffles the dataset before each epoch"""

  def __init__(self, data_source: MNIST | CIFAR10, batch_size: int):
    super().__init__(data_source, batch_size, shuffle=True)


class SequentialEpochSampler(EpochSampler):
  """Iterates over the dataset in the same order every epoch"""

  def __init__(self, data_source: MNIST | CIFAR10, batch_size: int):
    super().__init__(data_source, batch_size, shuffle=False)


class ClasswiseSampler(EpochSampler):
  """Selects only samples from the same class before moving to the next class"""

  def __init__(self, data_source: MNIST | CIFAR10, batch_size: int, shuffle: bool = True):
    super().__init__(data_source, batch_size, shuffle)

    # Group indices by class
    self.class_indices = {}
    for idx, (_, label, _) in enumerate(data_source):
      if label not in self.class_indices:
        self.class_indices[label] = []
      self.class_indices[label].append(idx)

    self.class_labels = list(self.class_indices.keys())
    self.label_index = None

  def __iter__(self):
    # start a new epoch if needed
    if not self.label_index or self.label_index == len(self.class_indices):
      self.label_index = 0
      if self.shuffle:
        self.class_labels = [self.class_labels[i] for i in torch.randperm(len(self.class_labels)).tolist()]

    indices = self.class_indices.get(self.class_labels[self.label_index])
    if self.shuffle:
      yield from (indices[i] for i in torch.randperm(len(indices)).tolist())
    else:
      yield from indices
    self.label_index += 1


class BatchSampler(Sampler):
  r"""Wraps another sampler to yield a mini-batch of indices."""

  def __init__(self, sampler, batch_size: int):
    self.sampler = sampler
    self.batch_size = batch_size

  def __iter__(self):
    while True:
      batch = [0] * self.batch_size
      idx_in_batch = 0
      for idx in self.sampler:
        batch[idx_in_batch] = idx
        idx_in_batch += 1
        if idx_in_batch == self.batch_size:
          yield batch
          idx_in_batch = 0
          batch = [0] * self.batch_size
      if idx_in_batch > 0:
        yield batch[:idx_in_batch]

  def update(self, values, indices):
    self.sampler.update(values, indices)

  def requires_grad(self):
    return self.sampler.requires_grad()
