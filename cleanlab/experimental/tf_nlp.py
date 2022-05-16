"""
A Scikeras classifier (adapted for huggingface models) which
can be used for finding label issues in any text datasets.
"""

from typing import Dict
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf
import numpy as np


class HuggingfaceKerasClassifier(KerasClassifier):
    def __init__(self, train_input: Dict, seq_len: int, **kwargs):
        """
        Basic scikeras/scikeras does not directly provide the option of having
        a multi-input model, i.e the input must be in the form (num_samples, feature_size). However, you can work around this problem by
        adding the `feature_encoder` property to the class which extends
        Scikitlearn's BaseEstimator.

        Example of use:

        * ``model = HuggingfaceKerasClassifier(``
        * ``    # --- model function parameters ---``
        * ``    model=model_fn,``
        * ``    n_classes=2,``
        * ``    # --- HuggingfaceKerasClassifier Parameters ---``
        * ``    train_input=dict(train_input),``
        * ``    seq_len=20,``
        * ``    #   --- TF training Parameters ---``
        * ``    optimizer=tf.keras.optimizers.Adam(2e-5),``
        * ``    loss=tf.keras.losses.BinaryCrossentropy(),``
        * ``    metrics=['accuracy'],``
        * ``    epochs=10,``
        * ``    batch_size=64,``
        * ``    shuffle=True,``
        * ``    callbacks=[early_stopping],``
        * ``    verbose=True``
        * ``)``
        * `` ``
        * ``lnl = CleanLearning(clf=model, cv_n_folds=3)``
        * ``lnl.fit(training_ids, train_labels, clf_kwargs={'validation_data': val_dataset})``
        * ``lnl.score(test_ids, test_labels)``

        References:
        - https://towardsdatascience.com/scikeras-tutorial-a-multi-input-multi-output-wrapper-for-capsnet-hyperparameter-tuning-with-keras-3127690f7f28
        - https://www.adriangb.com/scikeras/stable/notebooks/DataTransformers.html#4.-Multiple-inputs

        Parameters
        ----------
        train_input : dictionary or pandas DataFrame,
        Tokenized input data. Must contains the following keys/columns (For a more detailed explanation refers to `Huggingface documentation <https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__>`):
        - `input_ids`: a list or tensorflow tensor of token ids to be fed to the model. Shape = (num_sample, sequence_len).
        - `attention_mask` a list or tensorflow tensor of indices specifying which tokens should be attended to by the model. Shape = (num_sample, sequence_len).

        sequence_len : int,
        Tokenized sentences length.

        kwargs : optional,
        Optional arguments useful to fit a `Tensorflow model`. Refers to Tensorflow documentation for a detailed list of attributes `<https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`
        """
        super().__init__(**kwargs)
        self.train_input = train_input
        self.seq_len = seq_len

    def split_input(self, X):
        splitted_X = [
            X[:, : self.seq_len],  # input_ids
            X[:, self.seq_len :],  # attention_mask
        ]
        return splitted_X

    @property
    def feature_encoder(self):
        return FunctionTransformer(
            func=self.split_input,
        )

    def _get_tf_input(self, ids, train_input):
        indexed_input_ids = np.array(tf.gather(train_input["input_ids"], indices=ids))

        indexed_attention_mask = np.array(tf.gather(train_input["attention_mask"], indices=ids))

        return np.hstack([indexed_input_ids, indexed_attention_mask])

    def fit(self, ids, y, sample_weight=None, **kwargs):
        """
        Constructs and fit a new model using the given data.
        Parameters
        ----------
        ids : array-like of shape (n_samples,)
            Ids of training samples to be used to train the model.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        **kwargs : Dict[str, Any]
            Extra arguments to route to Tensorflow `Model.fit`.
        """
        X = self._get_tf_input(ids, self.train_input)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    def predict_proba(self, ids, **kwargs):
        """
        Returns class probability estimates for the given data.
        Parameters
        ----------
        ids : array-like of shape (n_samples,)
            Ids of training samples to be used to train the model.
        **kwargs : Dict[str, Any]
            Extra arguments to route to Tensorflow ``Model.predict``.
        """

        X = self._get_tf_input(ids, self.train_input)
        return super().predict_proba(X, **kwargs)

    def predict(self, ids, **kwargs):
        """
        Returns predictions for the given data.
        Parameters
        ----------
        ids : array-like of shape (n_samples,)
            Ids of training samples to be used to train the model.
        **kwargs : Dict[str, Any]
            Extra arguments to route to Tensorflow `Model.predict`.
        """
        X = self._get_tf_input(ids, self.train_input)
        return super().predict(X, **kwargs)
