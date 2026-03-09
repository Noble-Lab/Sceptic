"""
Example: run Sceptic on a two-timepoint subset of the bundled scGEM dataset.

This example uses only the first two developmental time points from scGEM to
demonstrate the supported two-class classification workflow. LOTO is not used
here because it is not meaningful with only two time points.
"""

from __future__ import annotations

import numpy as np

from sceptic import run_sceptic_and_evaluate


def load_two_timepoint_scgem():
    """Load the bundled scGEM data and keep only two time points."""
    data = np.loadtxt("example_data/scGEM/expression.txt")
    raw_labels = np.loadtxt("example_data/scGEM/expression_type.txt")

    time_lookup = {
        0.0: 0.0,
        1.0: 8.0,
    }
    keep_mask = np.isin(raw_labels, list(time_lookup))

    subset_data = data[keep_mask]
    subset_labels = np.array([time_lookup[value] for value in raw_labels[keep_mask]])
    label_list = np.array(sorted(time_lookup.values()), dtype=float)
    return subset_data, subset_labels, label_list


def run_example(method, data, labels, label_list):
    """Run Sceptic with a method-specific parameter grid."""
    if method == "svm":
        parameters = {
            "C": [1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale"],
        }
    else:
        parameters = {
            "max_depth": [3, 5],
            "learning_rate": [0.1, 0.3],
            "n_estimators": [100],
            "subsample": [0.8],
        }

    cm, predicted_labels, pseudotime, probabilities = run_sceptic_and_evaluate(
        data=data,
        labels=labels,
        label_list=label_list,
        parameters=parameters,
        method=method,
        use_gpu=False,
    )

    print(f"\n=== {method.upper()} ===")
    print("Confusion matrix:")
    print(cm)
    print("Predicted labels (first 10):", predicted_labels[:10])
    print("Pseudotime (first 10):", pseudotime[:10])
    print("Probabilities (first row):", probabilities[0])


def main():
    data, labels, label_list = load_two_timepoint_scgem()

    print("Two-timepoint scGEM subset loaded")
    print(f"Data shape: {data.shape}")
    print(f"Time points: {label_list}")
    print(
        "Cells per time point:",
        [int(np.sum(labels == time_point)) for time_point in label_list],
    )
    print(
        "Note: two-timepoint classification is supported, but this setting "
        "was not benchmarked in the original Sceptic study."
    )
    print("Note: use k-fold CV for two time points; LOTO is not supported.")

    for method in ("svm", "xgboost"):
        run_example(method, data, labels, label_list)


if __name__ == "__main__":
    main()
