"""
Manual smoke test for Sceptic LOTO and regression workflows.
"""

from __future__ import annotations

import numpy as np
from sklearn import preprocessing

from sceptic import run_sceptic_and_evaluate
from sceptic.evaluation import (
    compute_correlation_metrics,
    compute_regression_metrics,
    evaluate_sceptic_results,
)


def _load_scgem():
    """Load scGEM example data and derive encoded and continuous labels."""
    data = np.loadtxt("example_data/scGEM/expression.txt")
    raw_labels = np.loadtxt("example_data/scGEM/expression_type.txt")

    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(raw_labels)

    time_lookup = {
        0.0: 0,
        1.0: 8,
        2.0: 16,
        3.0: 24,
        4.0: 30,
    }
    time_labels = np.array([time_lookup[value] for value in raw_labels], dtype=float)
    ordered_times = np.array(sorted(time_lookup.values()), dtype=float)
    return data, encoded_labels, time_labels, ordered_times


def _print_classification_summary(name, cm, y_true, y_pred, pseudotime, time_axis):
    true_time = np.array([time_axis[idx] for idx in y_true], dtype=float)
    metrics = evaluate_sceptic_results(
        confusion_matrix=cm,
        y_true=y_true,
        y_pred=y_pred.astype(int),
        pseudotime=pseudotime,
        true_time=true_time,
        include_regression=True,
        verbose=False,
    )

    print(f"\n{name}")
    print(f"  Confusion Matrix shape: {cm.shape}")
    print(f"  Label predicted shape: {y_pred.shape}")
    print(f"  Pseudotime shape: {pseudotime.shape}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Spearman correlation: {metrics['spearman'][0]:.4f}")
    print(f"  Pearson correlation: {metrics['pearson'][0]:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")


def _print_regression_summary(name, pseudotime, true_time):
    corr_metrics = compute_correlation_metrics(pseudotime, true_time)
    reg_metrics = compute_regression_metrics(pseudotime, true_time)

    print(f"\n{name}")
    print(f"  Pseudotime shape: {pseudotime.shape}")
    print(f"  Pseudotime range: [{pseudotime.min():.2f}, {pseudotime.max():.2f}]")
    print(f"  True time range: [{true_time.min():.2f}, {true_time.max():.2f}]")
    print(f"  Spearman correlation: {corr_metrics['spearman'][0]:.4f}")
    print(f"  Pearson correlation: {corr_metrics['pearson'][0]:.4f}")
    print(f"  MAE: {reg_metrics['mae']:.4f}")
    print(f"  MSE: {reg_metrics['mse']:.4f}")
    print(f"  RMSE: {reg_metrics['rmse']:.4f}")


def main():
    print("=" * 80)
    print("Testing New Sceptic Features: LOTO and Regression")
    print("=" * 80)

    data, encoded_labels, time_labels, label_list = _load_scgem()
    inverse_counts = [int(np.sum(time_labels == time_point)) for time_point in label_list]

    print("\nLoading scGEM example data...")
    print(f"Data shape: {data.shape}")
    print(f"Number of cells: {data.shape[0]}")
    print(f"Number of features: {data.shape[1]}")
    print(f"Time points: {label_list}")
    print(f"Cells per time point: {inverse_counts}")

    parameters = {
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.3],
        "n_estimators": [100],
    }

    print("\n" + "=" * 80)
    print("Test 1: Classification + k-fold")
    print("=" * 80)
    cm, y_pred, pseudotime, _ = run_sceptic_and_evaluate(
        data=data,
        labels=encoded_labels,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False,
        cv_strategy="kfold",
        model_type="classification",
    )
    _print_classification_summary(
        "Classification + k-fold results",
        cm,
        encoded_labels,
        y_pred,
        pseudotime,
        label_list,
    )

    print("\n" + "=" * 80)
    print("Test 2: Classification + LOTO")
    print("=" * 80)
    cm, y_pred, pseudotime, _ = run_sceptic_and_evaluate(
        data=data,
        labels=encoded_labels,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False,
        cv_strategy="loto",
        model_type="classification",
    )
    _print_classification_summary(
        "Classification + LOTO results",
        cm,
        encoded_labels,
        y_pred,
        pseudotime,
        label_list,
    )

    print("\n" + "=" * 80)
    print("Test 3: Regression + k-fold")
    print("=" * 80)
    cm, y_pred, pseudotime, probabilities = run_sceptic_and_evaluate(
        data=data,
        labels=time_labels,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False,
        cv_strategy="kfold",
        model_type="regression",
    )
    print(f"  Confusion Matrix: {cm}")
    print(f"  Label predicted: {y_pred}")
    print(f"  Probabilities: {probabilities}")
    _print_regression_summary("Regression + k-fold results", pseudotime, time_labels)

    print("\n" + "=" * 80)
    print("Test 4: Regression + LOTO")
    print("=" * 80)
    cm, y_pred, pseudotime, probabilities = run_sceptic_and_evaluate(
        data=data,
        labels=time_labels,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False,
        cv_strategy="loto",
        model_type="regression",
    )
    print(f"  Confusion Matrix: {cm}")
    print(f"  Label predicted: {y_pred}")
    print(f"  Probabilities: {probabilities}")
    _print_regression_summary("Regression + LOTO results", pseudotime, time_labels)

    print("\n" + "=" * 80)
    print("Test 5: Backward compatibility")
    print("=" * 80)
    cm, y_pred, pseudotime, _ = run_sceptic_and_evaluate(
        data=data,
        labels=encoded_labels,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False,
    )
    print("  Backward compatibility maintained.")
    print(f"  Confusion Matrix shape: {cm.shape}")
    print(f"  Pseudotime shape: {pseudotime.shape}")
    print(f"  Label predicted shape: {y_pred.shape}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
