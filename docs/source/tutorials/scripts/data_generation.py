import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from cleanlab.benchmarking.noise_generation import (
    generate_noise_matrix_from_trace,
    generate_noisy_labels,
)

SEED = 123
np.random.seed(SEED)

BINS = {
    "low": [-np.inf, 3.3],
    "mid": [3.3, 6.6],
    "high": [6.6, +np.inf],
}

BINS_MAP = {
    "low": 0,
    "mid": 1,
    "high": 2,
}


def create_data():

    X = np.random.rand(250, 2) * 5
    y = np.sum(X, axis=1)
    # Map y to bins based on the BINS dict
    y_bin = np.array([k for y_i in y for k, v in BINS.items() if v[0] <= y_i < v[1]])
    y_bin_idx = np.array([BINS_MAP[k] for k in y_bin])

    # Split into train and test
    X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
        X, y_bin, y_bin_idx, test_size=0.5, random_state=SEED
    )

    # Add several (5) out-of-distribution points. Sliding them along the decision boundaries
    # to make them look like they are out-of-frame
    X_out = np.array(
        [
            [-1.5, 3.0],
            [-1.75, 6.5],
            [1.5, 7.2],
            [2.5, -2.0],
            [5.5, 7.0],
        ]
    )
    # Add a near duplicate point to the last outlier, with some tiny noise added
    near_duplicate = X_out[-1:] + np.random.rand(1, 2) * 1e-6
    X_out = np.concatenate([X_out, near_duplicate])

    y_out = np.sum(X_out, axis=1)
    y_out_bin = np.array([k for y_i in y_out for k, v in BINS.items() if v[0] <= y_i < v[1]])
    y_out_bin_idx = np.array([BINS_MAP[k] for k in y_out_bin])

    # Add to train
    X_train = np.concatenate([X_train, X_out])
    y_train = np.concatenate([y_train, y_out])
    y_train_idx = np.concatenate([y_train_idx, y_out_bin_idx])

    # Add an exact duplicate example to the training set
    exact_duplicate_idx = np.random.randint(0, len(X_train))
    X_duplicate = X_train[exact_duplicate_idx, None]
    y_duplicate = y_train[exact_duplicate_idx, None]
    y_duplicate_idx = y_train_idx[exact_duplicate_idx, None]

    # Add to train
    X_train = np.concatenate([X_train, X_duplicate])
    y_train = np.concatenate([y_train, y_duplicate])
    y_train_idx = np.concatenate([y_train_idx, y_duplicate_idx])

    py = np.bincount(y_train_idx) / float(len(y_train_idx))
    m = len(BINS)

    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=0.9 * m,
        py=py,
        valid_noise_matrix=True,
        seed=SEED,
    )

    noisy_labels_idx = generate_noisy_labels(y_train_idx, noise_matrix)

    # TODO: Add noise to test set when we support extra splits in Datalab

    noisy_labels = np.array([list(BINS_MAP.keys())[i] for i in noisy_labels_idx])

    return X_train, y_train_idx, noisy_labels, noisy_labels_idx, X_out, X_duplicate


def plot_data(X_train, y_train_idx, noisy_labels_idx, X_out, X_duplicate):
    # Plot data with clean labels and noisy labels, use BINS_MAP for the legend
    fig, ax = plt.subplots(figsize=(8, 6))
    for k, v in BINS_MAP.items():
        ax.scatter(X_train[noisy_labels_idx == v, 0], X_train[noisy_labels_idx == v, 1], label=k)
    ax.set_title("Noisy labels")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # Plot true boundaries (x+y=3.3, x+y=6.6)
    ax.set_xlim(-3.5, 8.5)
    ax.set_ylim(-3.5, 8.5)
    ax.plot([-0.7, 4.0], [4.0, -0.7], color="k", linestyle="--", alpha=0.5)
    ax.plot([-0.7, 7.3], [7.3, -0.7], color="k", linestyle="--", alpha=0.5)

    # Draw red circles around the points that are misclassified (i.e. the points that are in the wrong bin)
    for i, (X, y) in enumerate(zip([X_train, X_train], [y_train_idx, noisy_labels_idx])):
        for j, (k, v) in enumerate(BINS_MAP.items()):
            ax.plot(
                X[(y == v) & (y != y_train_idx), 0],
                X[(y == v) & (y != y_train_idx), 1],
                "o",
                markerfacecolor="none",
                markeredgecolor="red",
                markersize=14,
                markeredgewidth=2.5,
                alpha=0.5,
                **{"label": "Label error" if i == 1 and j == 0 else None}
            )

    ax.scatter(X_out[:, 0], X_out[:, 1], color="k", marker="x", s=100, linewidth=2, label="Outlier")

    # Plot the exact duplicate
    ax.scatter(
        X_duplicate[:, 0],
        X_duplicate[:, 1],
        color="c",
        marker="x",
        s=100,
        linewidth=2,
        label="Exact Duplicate",
    )
    ax.legend()
    plt.tight_layout()
