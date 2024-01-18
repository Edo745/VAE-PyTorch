# Variational Autoencoder (VAE)

This project implements a (Linear) Variational Autoencoder (VAE) following the paper "Auto-Encoding Variational Bayes" by Diederik P. Kingma et al. The model is trained on the MNIST dataset and can be used to generate new samples.

## Overview

The Variational Autoencoder is a generative model that learns a probabilistic mapping between input data and a latent space. It is trained to encode input data into a distribution and decode samples from that distribution back into the input space. This project provides a basic implementation of a VAE using PyTorch and demonstrates how to train the model on the MNIST dataset. 

## Requirements

- PyTorch
- torchvision
- Matplotlib (for visualization)

## Usage

### Training

To train the linear VAE model, run the `train.py` script. You can customize training parameters such as learning rate, epochs, and batch size using command-line arguments. Example:

```bash
python train.py --learning_rate 0.001 --epochs 20 
```

### Digit-Specific Generation:

Users can specify a digit to generate, and the model will generate samples based on the learned characteristics of that digit.
```bash
generate.py  --checkpoint_path  <path/to/checkpoints.pth> --digit_to_generate 0
```
### Interpolation:
The project supports interpolating between two specified digits in the latent space.
Users can visualize the transition between two digits by interpolating their latent representations.
```bash
generate.py  --checkpoint_path  <path/to/checkpoints.pth> --from_digit 0 --to_digit 1
```
