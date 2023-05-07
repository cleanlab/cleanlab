# Useful methods/models adapted for use with cleanlab 

Methods in this `experimental` module are bleeding edge and may have sharp edges. They are not guaranteed to be stable between different cleanlab versions.

Some of these files include various models that can be used with cleanlab to find issues in specific types of data. These require dependencies on deep learning and other machine learning packages that are not official cleanlab dependencies. You must install these dependencies on your own if you wish to use them.

The modules and required dependencies are as follows:
* mnist_pytorch.py - a cleanlab-compatible simplified AlexNet for MNIST using PyTorch
	- torch
	- torchvision
* cifar_cnn.py - a cleanlab-compatible Convolutional Neural Network for CIFAR using PyTorch, trainable via CoTeaching
	- torch
	- torchvision
* coteaching.py - an algorithm to train neural networks with noisy labels
	- torch

