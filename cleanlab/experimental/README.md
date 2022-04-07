# Experimental cleanlab functionality

Methods in this `experimental` module are bleeding edge and may have sharp edges.
They are not guaranteed to be stable between different `cleanlab` versions.

This submodule in cleanlab requires dependencies on deep learning and other machine learning
frameworks that are not directly supported in cleanlab.
You must install these dependencies on your own if you wish to use them.

The dependencies are as follows:
* fasttext.py - text classification with FastText models (allows you to find label issues in your text datasets)
	- fasttext
* coteaching.py - the Co-teaching algorithm for training neural networks on noisily-labeled data 
	- pytorch
