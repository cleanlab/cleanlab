# Useful models adapted for use with cleanlab 

Methods in this ``models`` module are not guaranteed to be stable between different ``cleanlab`` versions.

Some of these files include various models that can be used with cleanlab to find issues in specific types of data. These require dependencies on deep learning and other machine learning packages that are not official cleanlab dependencies. You must install these dependencies on your own if you wish to use them.

The dependencies are as follows:
* keras.py - a wrapper to make any Keras model compatible with cleanlab and sklearn
    - tensorflow
* fasttext.py - a cleanlab-compatible FastText classifier for text data
	- fasttext
