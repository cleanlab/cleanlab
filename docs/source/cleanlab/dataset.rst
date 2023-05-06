dataset
=======

.. automodule:: cleanlab.dataset
   :autosummary:
   :members:
   :undoc-members:
   :show-inheritance:

.. testsetup:: *

   import cleanlab
   import numpy as np
   from cleanlab.benchmarking import noise_generation

   SEED = 0

   def get_data_labels_from_dataset(
      means=[[3, 2], [7, 7], [0, 8], [0, 10]],
      covs=[
         [[5, -1.5], [-1.5, 1]],
         [[1, 0.5], [0.5, 4]],
         [[5, 1], [1, 5]],
         [[3, 1], [1, 1]],
      ],
      sizes=[100, 50, 50, 50],
      avg_trace=0.8,
      seed=SEED,  # set to None for non-reproducible randomness
   ):
      np.random.seed(seed=SEED)

      K = len(means)  # number of classes
      data = []
      labels = []
      test_data = []
      test_labels = []

      for idx in range(K):
         data.append(
               np.random.multivariate_normal(
                  mean=means[idx], cov=covs[idx], size=sizes[idx]
               )
         )
         test_data.append(
               np.random.multivariate_normal(
                  mean=means[idx], cov=covs[idx], size=sizes[idx]
               )
         )
         labels.append(np.array([idx for i in range(sizes[idx])]))
         test_labels.append(np.array([idx for i in range(sizes[idx])]))
      X_train = np.vstack(data)
      y_train = np.hstack(labels)
      X_test = np.vstack(test_data)
      y_test = np.hstack(test_labels)

      # Compute p(y=k) the prior distribution over true labels.
      py_true = np.bincount(y_train) / float(len(y_train))

      noise_matrix_true = noise_generation.generate_noise_matrix_from_trace(
         K,
         trace=avg_trace * K,
         py=py_true,
         valid_noise_matrix=True,
         seed=SEED,
      )

      # Generate our noisy labels using the noise_marix.
      s = noise_generation.generate_noisy_labels(y_train, noise_matrix_true)
      s_test = noise_generation.generate_noisy_labels(y_test, noise_matrix_true)
      ps = np.bincount(s) / float(len(s))  # Prior distribution over noisy labels

      return X_train, s