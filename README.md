# MNIST Classification with CNN

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.43%25-success.svg)]()
[![Parameters](https://img.shields.io/badge/Parameters-13.2K-informational.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Tests](https://img.shields.io/github/actions/workflow/status/username/repo/tests.yml?label=tests)](https://github.com/username/repo/actions)

This repository contains a highly optimized CNN model for MNIST digit classification that achieves 99.4% test accuracy in less than 10 epochs with under 20K parameters.

## Model Architecture

The model uses a carefully designed CNN architecture with the following key features:

- **Parameter Efficient**: Only 13,272 trainable parameters
- **Optimized Depth**: 7 convolutional layers with strategic channel progression
- **Regularization**: Employs Batch Normalization and Dropout (0.1)
- **Modern Architecture**: Uses skip connections and proper gradient flow

## Performance

- **Test Accuracy**: 99.43% (achieved in epoch 10)
- **Training Time**: ~10 epochs
- **Parameters**: 13,272 (well under 20K limit)

## Key Features

1. **Efficient Architecture**
   - Strategic use of channels (8→16→32)
   - Proper gradient flow
   - Balanced depth and width

2. **Training Optimizations**
   - Batch size of 28 for better generalization
   - Adam optimizer with learning rate of 0.0001
   - OneCycleLR scheduler for optimal convergence

3. **Data Augmentation**
   - Random rotation (-15°, 15°)
   - Random affine transformations
   - Proper normalization (0.1307, 0.3081)

## Usage
Train the model : python Session6_assignment.py
Run tests : pytest tests/test_model.py

## Requirements

- PyTorch
- torchvision
- numpy
- tqdm
- pytest (for testing)

## Model Testing

The model includes automated tests that verify:
- Output shape correctness
- Parameter count constraints
- Forward pass stability
- Performance metrics

These tests are automatically run on every push and pull request through GitHub Actions.

## Results

The model achieves:
- Training accuracy: >99.4%
- Test accuracy: 99.43%
- Convergence: 10 epochs
- Parameters: 13,272

