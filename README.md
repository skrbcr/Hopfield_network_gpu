# Hopfield Network GPU

[![Python package](https://github.com/skrbcr/Hopfield_network_gpu/actions/workflows/python-package.yml/badge.svg)](https://github.com/skrbcr/Hopfield_network_gpu/actions/workflows/python-package.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skrbcr/Hopfield_network_gpu/blob/main/Hopfield_Tutorial.ipynb)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat-square&logo=github)](https://skrbcr.github.io/Hopfield_network_gpu/)

GPU version of [Hopfield_network](https://github.com/skrbcr/Hopfield_network). I also revised source code and rewrite it in Python.


## About

Hopfield network is a classical neural network model.
It is capable of storing patterns e.g., images and recalling them even when presented with noisy or incomplete input.

**Demonstration**:

https://github.com/user-attachments/assets/9209d37e-a18a-4ba0-ad60-7b2fbdb47c83

## Usage

If you don't have graphics card, please execute it with [Google Colab](https://colab.research.google.com/).

There are tutorial and example in this repository:

- [`AssocMem_Tutorial.ipynb`](https://colab.research.google.com/github/skrbcr/Hopfield_network_gpu/blob/main/Hopfield_Tutorial.ipynb): Tutorial of Hopfield network and this module. When you run it, please put `AssocMem.py` in the same directory.
- [`compare_cpu_gpu.ipynb`](https://colab.research.google.com/github/skrbcr/Hopfield_network_gpu/blob/main/compare_cpu_gpu.ipynb): Comparison of exection time between CPU and GPU. You can experience the benefit of GPU.

