
## Contrastive Normalizing Flow: Robust Uncertainty Estimation Under Distributional Shifts
This repository provides the official PyTorch implementation code of the paper.

> Juhong Song, Jaewoong Choi, Myungjoo Kang, Contrastive Normalizing Flow: Robust Uncertainty Estimation Under Distributional Shifts, arXiv, 2021

## Dependencies

	conda env create -f environment.yml

## Data

The base dataset of the experiments use MNIST and CIFAR.

For distributional shift datasets, MNIST rotation and [CIFAR-10-C](https://zenodo.org/record/2535967) are used.

For OOD shift datasets, SVHN, LSUN, [TinyImageNet](https://www.kaggle.com/c/tiny-imagenet/data), F-MNIST, NotMNIST, and EMNIST-letters are used.

## Train the proposed model

**Train CTNF for CIFAR dataset**

	python cifar_train.py

**Train CTNF for MNIST dataset**

	python mnist_train.py

## Experiment using pre-trained model

Assume that the pre-trained model is downloaded on ./data/weights folder.

**CIFAR-10-Corruption and OOD Detection for SVHN, LSUN, and TinyImageNet**

	python experiments/cifar_c_main.py --ood svhn

**MNIST Rotation and OOD detection for F-MNIST, NotMNIST, and EMNIST-letters**

	python experiments/mnist_r_main.py --ood fmnist


