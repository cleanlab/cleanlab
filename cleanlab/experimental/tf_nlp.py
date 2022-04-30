"""
A Scikeras classifier (adapted for huggingface models) which
can be used for finding label issues in any nlp dataset.
"""

from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf
import numpy as np


class MultiInputKerasClassifier(KerasClassifier):
    def __init__(self, train_input, seq_len, valid_input=None, y_val=None, **kwargs):
        """
        Basic scikeras/scikeras does not directly provide the option of having
        a multi-input model. However, you can work around this problem by
        adding the `feature_encoder` property to the class which extends
        Scikitlearn's BaseEstimator.

        References:
        - https://towardsdatascience.com/scikeras-tutorial-a-multi-input-multi-output-wrapper-for-capsnet-hyperparameter-tuning-with-keras-3127690f7f28
        - https://www.adriangb.com/scikeras/stable/notebooks/DataTransformers.html#4.-Multiple-inputs

        """
        super().__init__(**kwargs)
        self.train_input = train_input
        self.valid_input = valid_input
        self.y_val = y_val
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

    def fit(self, ids, y, sample_weight=None):
        X = self._get_tf_input(ids, self.train_input)
        val_dataset = None

        if self.valid_input:
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "input_ids": self.valid_input["input_ids"],
                        "attention_mask": self.valid_input["attention_mask"],
                    },
                    self.y_val,
                )
            ).batch(32)

        return super().fit(X, y, validation_data=val_dataset)

    def predict_proba(self, ids):
        X = self._get_tf_input(ids, self.train_input)
        return super().predict_proba(X)

    def predict(self, ids):
        X = self._get_tf_input(ids, self.train_input)
        return super().predict(X)
