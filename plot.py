#!/usr/bin/env python

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

with open('results.json', 'r') as f:
  results = json.load(f)

# Group by sampler
sampler_runs = defaultdict(list)
for run in results:
  sampler_runs[run['sampler']].append(run['training_loss'])

# Setup plot
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple']  # Add more if needed

# Plot for each sampler
for (sampler, losses), color in zip(sampler_runs.items(), colors):
  # Plot individual runs with transparency
  for loss in losses:
    plt.plot(loss, alpha=0.2, color=color, linewidth=0.5)

  # Calculate and plot mean every 100 steps
  assert all(len(loss) == len(losses[0]) for loss in losses)
  mean_loss = np.mean(losses, axis=0)

  # Sample every 100 points
  x_points = np.arange(1, len(mean_loss), 100)
  y_points = mean_loss[x_points]

  plt.plot(x_points, y_points,
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
plt.savefig('plot.png', dpi=300)
plt.show()
