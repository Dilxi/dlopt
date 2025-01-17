#!/usr/bin/env python

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

with open('results.json', 'r') as f:
  results = json.load(f)

# Group by sampler
sampler_runs = defaultdict(list)
sampler_accuracies = defaultdict(list)
for run in results:
  sampler_runs[run['sampler']].append(run['training_loss'])
  sampler_accuracies[run['sampler']].append(run['test_accuracy'])

# Print mean accuracies
print("\nFinal Test Accuracy Summary:")
print("-" * 40)
print(f"{'Sampler':<20} {'Mean Accuracy':<10}")
print("-" * 40)
for sampler, accuracies in sampler_accuracies.items():
  mean_acc = np.mean(accuracies)
  print(f"{sampler:<20} {mean_acc:.4f}")
print("-" * 40)

colors = ['blue', 'red', 'green', 'orange', 'purple']  # Add more if needed

for (sampler, losses), color in zip(sampler_runs.items(), colors):
  assert all(len(loss) == len(losses[0]) for loss in losses)

  window_size = 500
  n_windows = len(losses[0]) // window_size

  windowed_means = []
  windowed_lower = []
  windowed_upper = []

  for i in range(n_windows):
    start_idx = i * window_size
    end_idx = (i + 1) * window_size
    window_losses = [loss[start_idx:end_idx] for loss in losses]

    # Calculate statistics for this window
    window_means = [np.mean(run) for run in window_losses]
    windowed_means.append(np.mean(window_means))
    windowed_lower.append(np.percentile(window_means, 10))
    windowed_upper.append(np.percentile(window_means, 90))

  # Plot with windowed statistics
  x_points = np.arange(window_size, window_size * (n_windows + 1), window_size) - window_size // 2

  plt.fill_between(x_points, windowed_lower, windowed_upper,
                  color=color, alpha=0.2)

  plt.plot(x_points, windowed_means,
          color=color,
          marker='o',
          markersize=4,
          linewidth=2,
          label=f'{sampler}')

plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss by Sampler Type')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.yscale('log')
plt.savefig('loss.png', dpi=300)

# Create another figure for accuracy plots
plt.figure(figsize=(10, 6))

for (sampler, accuracies), color in zip(sampler_accuracies.items(), colors):
    # Calculate statistics
    mean_acc = np.mean(accuracies, axis=0)
    lower_bound = np.percentile(accuracies, 10, axis=0)
    upper_bound = np.percentile(accuracies, 90, axis=0)

    x_points = np.arange(1000, (len(mean_acc) + 1) * 1000, 1000)

    # Plot confidence interval
    plt.fill_between(x_points, lower_bound, upper_bound,
                    color=color, alpha=0.2)

    # Plot mean line
    plt.plot(x_points, mean_acc,
             color=color,
             marker='o',
             markersize=4,
             linewidth=2,
             label=f'{sampler}')

plt.xlabel('Iterations')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy by Sampler Type')
plt.legend()
plt.grid(True)
plt.savefig('accuracy.png', dpi=300)

# Show all plots
plt.show()
