# Experimental methods

Methods in this `experimental` module are bleeding edge and may have sharp edges. They are not guaranteed to be stable between different cleanlab versions.

Some of these files include various models that can be used with cleanlab to find issues in specific types of data. These require dependencies on deep learning and other machine learning packages that are not official cleanlab dependencies. You must install these dependencies on your own if you wish to use them.

The dependencies are as follows:
* fasttext.py - a FastText classifier for text data
	- fasttext
* mnist_pytorch.py - training a simplified AlexNet on MNIST using PyTorch
	- pytorch
	- torchvision
* cifar_cnn.py - training on CIFAR using PyTorch via CoTeaching method
	- pytorch
	- torchvision
* coteaching - an algorithm to train neural networks with noisy labels
	- pytorch

