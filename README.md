# Sampling Strategy for SGD

## Overview
Experiment with 5 different sampling strategies for Stochastic Gradient Descent (SGD) training:
- Uniform random sampling with replacement
- Importance sampling weighted by gradient norms
- Shuffled/Sequential epoch sampling
- Classwise sampling

## Project Structure
```
.
├── main.py          # Entry point
├── sampler.py       # Sampling strategy implementations
├── setup.py         # Training setup and utilities
├── vgg.py           # VGG model
└── plot.py          # Plotting
```

## Running Experiments
To run the experiment, execute:
```bash
./main.py --batch-size=128 --iters=5000 --runs=5
```

Run `./main.py --help` to see all options.

## Viewing Results
Results are saved in `result.json` with the following metrics of each individual run:

- Sampling strategy name
- Run id
- Training loss after each iteration
- Final test accuracy

Run `./plot.py` to plot the results in `result.json`. The two plots will be saved as `accuracy.png` and `loss.png`.

## Acknowledgements

- VGG model implementation from [torchvision.models.vgg](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
- BatchSampler implementation from [torch.utils.data.sampler](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py)
