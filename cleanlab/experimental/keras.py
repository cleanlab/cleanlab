# Copyright (C) 2017-2022  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.

"""
A wrapper class you can use to make any Keras model compatible with cleanlab and sklearn.
Most of the instance methods of this class are the same as the ones for any Keras model, 
see the Keras documentation for details.

This is a good example of making any bespoke neural network compatible with cleanlab.

You must have Tensorflow installed: https://www.tensorflow.org/install

Note: Tensorflow is only compatible with Python versions >= 3.7: https://www.tensorflow.org/install/pip#software_requirements

Tips:
* If this class lacks certain functionality, you can alternatively try scikeras: https://github.com/adriangb/scikeras 
* To call ``fit()`` on a Tensorflow Dataset object with a Keras model, the Dataset should already be batched.
"""

import tensorflow as tf
import numpy as np
from typing import Callable


class KerasWrapperModel:
    def __init__(
        self,
        model: Callable,
        model_kwargs: dict = {},
        compile_kwargs: dict = {
            "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
    ):
        self.model = model
        self.model_kwargs = model_kwargs
        self.compile_kwargs = compile_kwargs
        self.net = None

    def get_params(self, deep=True):
        return {
            "model": self.model,
            "model_kwargs": self.model_kwargs,
            "compile_kwargs": self.compile_kwargs,
        }

    def fit(self, X, y=None, **kwargs):
        """Note that ``X`` dataset object must already contain the labels as is required for standard Keras fit.
        You can provide the labels again here as argument ``y`` to be compatible with sklearn, but they are ignored.
        """
        self.net = self.model(**self.model_kwargs)
        self.net.compile(**self.compile_kwargs)
        self.net.fit(X, **kwargs)

    def predict_proba(self, X, apply_softmax=True, **kwargs):
        """Set `apply_softmax` to True to indicate your network only outputs logits not probabilities"""
        if self.net is None:
            raise ValueError("must call fit() before predict()")
        pred_probs = self.net.predict(X, **kwargs)
        if apply_softmax:
            pred_probs = tf.nn.softmax(pred_probs, axis=1)
        return pred_probs

    def predict(self, X, **kwargs):
        pred_probs = self.predict_proba(X, **kwargs)
        return np.argmax(pred_probs, axis=1)

    def summary(self, **kwargs):
        self.net.summary(**kwargs)


class KerasWrapperSequential:
    """
    KerasWrapperSequential is instantiated in the same way as a ``tf.keras.models.Sequential`` object,
    except for extra argument:
    * *compile_kwargs*: dict of args to pass into ``model.compile()``
    """

    def __init__(
        self,
        layers: list = None,
        name: str = None,
        compile_kwargs: dict = {
            "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
    ):
        self.layers = layers
        self.name = name
        self.compile_kwargs = compile_kwargs
        self.net = None

    def get_params(self, deep=True):
        return {
            "layers": self.layers,
            "name": self.name,
            "compile_kwargs": self.compile_kwargs,
        }

    def fit(self, X, y=None, **kwargs):
        """Note that ``X`` dataset object must already contain the labels as is required for standard Keras fit.
        You can provide the labels again here as argument ``y`` to be compatible with sklearn, but they are ignored.
        """
        self.net = tf.keras.models.Sequential(self.layers, self.name)
        self.net.compile(**self.compile_kwargs)
        self.net.fit(X, **kwargs)

    def predict_proba(self, X, apply_softmax=True, **kwargs):
        """Set `apply_softmax` to True to indicate your network only outputs logits not probabilities"""
        if self.net is None:
            raise ValueError("must call fit() before predict()")
        pred_probs = self.net.predict(X, **kwargs)
        if apply_softmax:
            pred_probs = tf.nn.softmax(pred_probs, axis=1)
        return pred_probs

    def predict(self, X, **kwargs):
        pred_probs = self.predict_proba(X, **kwargs)
        return np.argmax(pred_probs, axis=1)

    def summary(self, **kwargs):
        self.net.summary(**kwargs)
