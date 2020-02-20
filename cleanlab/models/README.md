# cleanlab Models

This submodule in cleanlab requires dependencies on deep learning and other machine learning frameworks that are not directly support in cleanlab.
The dependencies are as follows

* cifar_cnn.py - used training on CIFAR using PyTorch via CoTeaching method
   - pytorch
   - torchvision
* mnist_pytorch.py - used for training a simplified AlexNet on MNIST using PyTorch
   - pytorch
   - torchvision
* fastext.py - used for supervised learning on text classification using FastText
   - fastext

You must install these dependencies on your own if you wish to use them. cleanlab will not force you to install these large deep learning frameworks because that may potentially clobber your code base. For example if you have both TensorFlow and PyTorch installed, issues sometims arise.
