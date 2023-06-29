# Copyright (C) 2017-2023  Cleanlab Inc.
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
Wrapper class you can use to make any Keras model compatible with :py:class:`CleanLearning <cleanlab.classification.CleanLearning>` and sklearn.
Use :py:class:`KerasWrapperModel<cleanlab.experimental.keras.KerasWrapperModel>` to wrap existing functional API code for ``keras.Model`` objects,
and :py:class:`KerasWrapperSequential<cleanlab.experimental.keras.KerasWrapperSequential>` to wrap existing ``tf.keras.models.Sequential`` objects.
Most of the instance methods of this class work the same as the ones for the wrapped Keras model,
see the `Keras documentation <https://keras.io/>`_ for details.

This is a good example of making any bespoke neural network compatible with cleanlab.

You must have `Tensorflow 2 installed <https://www.tensorflow.org/install>`_ (only compatible with Python versions >= 3.7).
This wrapper class is only fully compatible with ``tensorflow<2.11``, if using ``tensorflow>=2.11``, 
please replace your Optimizer class with the legacy Optimizer `here <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/legacy/Optimizer>`_.

Tips:

* If this class lacks certain functionality, you can alternatively try `scikeras <https://github.com/adriangb/scikeras>`_.
* Unlike scikeras, our `KerasWrapper` classes can operate directly on ``tensorflow.data.Dataset`` objects (like regular Keras models).
* To call ``fit()`` on a tensorflow ``Dataset`` object with a Keras model, the ``Dataset`` should already be batched.
* Check out our example using this class: `huggingface_keras_imdb <https://github.com/cleanlab/examples/blob/master/huggingface_keras_imdb/huggingface_keras_imdb.ipynb>`_
* Our `unit tests <https://github.com/cleanlab/cleanlab/blob/master/tests/test_frameworks.py>`_ also provide basic usage examples.

"""

import tensorflow as tf
import keras  # type: ignore
import numpy as np
from typing import Callable, Optional


class KerasWrapperModel:
    """Takes in a callable function to instantiate a Keras Model (using Keras functional API)
    that is compatible with :py:class:`CleanLearning <cleanlab.classification.CleanLearning>` and sklearn.

    The instance methods of this class work in the same way as those of any ``keras.Model`` object, see the `Keras documentation <https://keras.io/>`_ for details.
    For using Keras sequential instead of functional API, see the :py:class:`KerasWrapperSequential<cleanlab.experimental.keras.KerasWrapperSequential>` class.

    Parameters
    ----------
    model: Callable
        A callable function to construct the Keras Model (using functional API). Pass in the function here, not the constructed model!

        For example::

            def model(num_features, num_classes):
                inputs = tf.keras.Input(shape=(num_features,))
                outputs = tf.keras.layers.Dense(num_classes)(inputs)
                return tf.keras.Model(inputs=inputs, outputs=outputs, name="my_keras_model")

    model_kwargs: dict, default = {}
        Dict of optional keyword arguments to pass into ``model()`` when instantiating the ``keras.Model``.

    compile_kwargs: dict, default = {"loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
        Dict of optional keyword arguments to pass into ``model.compile()`` for declaring loss, metrics, optimizer, etc.
    """

    def __init__(
        self,
        model: Callable,
        model_kwargs: dict = {},
        compile_kwargs: dict = {
            "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        params: Optional[dict] = None,
    ):
        if params is None:
            params = {}

        self.model = model
        self.model_kwargs = model_kwargs
        self.compile_kwargs = compile_kwargs
        self.params = params
        self.net = None

    def get_params(self, deep=True):
        """Returns the parameters of the Keras model."""
        return {
            "model": self.model,
            "model_kwargs": self.model_kwargs,
            "compile_kwargs": self.compile_kwargs,
            "params": self.params,
        }

    def set_params(self, **params):
        """Set the parameters of the Keras model."""
        self.params.update(params)
        return self

    def fit(self, X, y=None, **kwargs):
        """Trains a Keras model.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            If ``X`` is a tensorflow dataset object, it must already contain the labels as is required for standard Keras fit.

        y : np.array or pd.DataFrame, default = None
            If ``X`` is a tensorflow dataset object, you can optionally provide the labels again here as argument `y` to be compatible with sklearn,
            but they are ignored.
            If ``X`` is a numpy array or pandas dataframe, the labels have to be passed in using this argument.
        """
        if self.net is None:
            self.net = self.model(**self.model_kwargs)
            self.net.compile(**self.compile_kwargs)

        # TODO: check for generators
        if y is not None and not isinstance(X, (tf.data.Dataset, keras.utils.Sequence)):
            kwargs["y"] = y

        self.net.fit(X, **{**self.params, **kwargs})

    def predict_proba(self, X, *, apply_softmax=True, **kwargs):
        """Predict class probabilities for all classes using the wrapped Keras model.
        Set extra argument `apply_softmax` to True to indicate your network only outputs logits not probabilities.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            Data in the same format as the original ``X`` provided to ``fit()``.
        """
        if self.net is None:
            raise ValueError("must call fit() before predict()")
        pred_probs = self.net.predict(X, **kwargs)
        if apply_softmax:
            pred_probs = tf.nn.softmax(pred_probs, axis=1)
        return pred_probs

    def predict(self, X, **kwargs):
        """Predict class labels using the wrapped Keras model.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            Data in the same format as the original ``X`` provided to ``fit()``.

        """
        pred_probs = self.predict_proba(X, **kwargs)
        return np.argmax(pred_probs, axis=1)

    def summary(self, **kwargs):
        """Returns the summary of the Keras model."""
        if self.net is None:
            self.net = self.model(**self.model_kwargs)
            self.net.compile(**self.compile_kwargs)

        return self.net.summary(**kwargs)


class KerasWrapperSequential:
    """Makes any ``tf.keras.models.Sequential`` object compatible with :py:class:`CleanLearning <cleanlab.classification.CleanLearning>` and sklearn.

    `KerasWrapperSequential` is instantiated in the same way as a keras ``Sequential``  object, except for optional extra `compile_kwargs` argument.
    Just instantiate this object in the same way as your ``tf.keras.models.Sequential`` object (rather than passing in an existing ``Sequential`` object).
    The instance methods of this class work in the same way as those of any keras ``Sequential`` object, see the `Keras documentation <https://keras.io/>`_ for details.

    Parameters
    ----------
    layers: list
        A list containing the layers to add to the keras ``Sequential`` model (same as for ``tf.keras.models.Sequential``).

    name: str, default = None
        Name for the Keras model (same as for ``tf.keras.models.Sequential``).

    compile_kwargs: dict, default = {"loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
        Dict of optional keyword arguments to pass into ``model.compile()`` for declaring loss, metrics, optimizer, etc.
    """

    def __init__(
        self,
        layers: Optional[list] = None,
        name: Optional[str] = None,
        compile_kwargs: dict = {
            "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        params: Optional[dict] = None,
    ):
        if params is None:
            params = {}

        self.layers = layers
        self.name = name
        self.compile_kwargs = compile_kwargs
        self.params = params
        self.net = None

    def get_params(self, deep=True):
        """Returns the parameters of the Keras model."""
        return {
            "layers": self.layers,
            "name": self.name,
            "compile_kwargs": self.compile_kwargs,
            "params": self.params,
        }

    def set_params(self, **params):
        """Set the parameters of the Keras model."""
        self.params.update(params)
        return self

    def fit(self, X, y=None, **kwargs):
        """Trains a Sequential Keras model.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            If ``X`` is a tensorflow dataset object, it must already contain the labels as is required for standard Keras fit.

        y : np.array or pd.DataFrame, default = None
            If ``X`` is a tensorflow dataset object, you can optionally provide the labels again here as argument `y` to be compatible with sklearn,
            but they are ignored.
            If ``X`` is a numpy array or pandas dataframe, the labels have to be passed in using this argument.
        """
        if self.net is None:
            self.net = tf.keras.models.Sequential(self.layers, self.name)
            self.net.compile(**self.compile_kwargs)

        # TODO: check for generators
        if y is not None and not isinstance(X, (tf.data.Dataset, keras.utils.Sequence)):
            kwargs["y"] = y

        self.net.fit(X, **{**self.params, **kwargs})

    def predict_proba(self, X, *, apply_softmax=True, **kwargs):
        """Predict class probabilities for all classes using the wrapped Keras model.
        Set extra argument `apply_softmax` to True to indicate your network only outputs logits not probabilities.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            Data in the same format as the original ``X`` provided to ``fit()``.
        """
        if self.net is None:
            raise ValueError("must call fit() before predict()")
        pred_probs = self.net.predict(X, **kwargs)
        if apply_softmax:
            pred_probs = tf.nn.softmax(pred_probs, axis=1)
        return pred_probs

    def predict(self, X, **kwargs):
        """Predict class labels using the wrapped Keras model.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            Data in the same format as the original ``X`` provided to ``fit()``.
        """
        pred_probs = self.predict_proba(X, **kwargs)
        return np.argmax(pred_probs, axis=1)

    def summary(self, **kwargs):
        """Returns the summary of the Keras model."""
        if self.net is None:
            self.net = tf.keras.models.Sequential(self.layers, self.name)
            self.net.compile(**self.compile_kwargs)

        return self.net.summary(**kwargs)
