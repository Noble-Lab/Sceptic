"""
Internal comparison script for scGEM regression experiments.

This utility runs both the legacy-style regression configuration and the
enhanced pipeline against the scGEM example data, printing summary metrics for
manual inspection. Keep outputs under version control only in the development
branch.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn import preprocessing

MODULE_DIR = Path(__file__).resolve().parents[2] / "src" / "sceptic"


def _load_module(module_name: str, file_name: str):
    spec = spec_from_file_location(module_name, MODULE_DIR / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_name}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sceptic = _load_module("_sceptic_impl", "sceptic.py")
_evaluation = _load_module("_eval_impl", "evaluation.py")

run_sceptic_and_evaluate = _sceptic.run_sceptic_and_evaluate
compute_regression_metrics = _evaluation.compute_regression_metrics
compute_correlation_metrics = _evaluation.compute_correlation_metrics
evaluate_sceptic_results = _evaluation.evaluate_sceptic_results


DATA_DIR = Path("example_data/scGEM")
LEGACY_REGRESSION_GRID: Dict[str, Any] = {
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.3],
    "n_estimators": [100],
}


def _load_scgem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load scGEM expression matrix and time labels."""
    data_path = DATA_DIR / "expression.txt"
    label_path = DATA_DIR / "expression_type.txt"
    X = np.loadtxt(data_path)
    raw_labels = np.loadtxt(label_path)
    encoder = preprocessing.LabelEncoder()
    encoded = encoder.fit_transform(raw_labels)
    time_lookup = {
        0.0: 0,
        1.0: 8,
        2.0: 16,
        3.0: 24,
        4.0: 30,
    }
    continuous = np.array([time_lookup[val] for val in raw_labels], dtype=float)
    ordered_times = np.array(sorted(time_lookup.values()), dtype=float)
    return X, encoded, continuous, ordered_times


def _summarise_regression(
    X: np.ndarray,
    y_cont: np.ndarray,
    time_axis: np.ndarray,
    *,
    scale_features: bool,
    param_grid: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fit the regression workflow and return metrics."""
    _, _, pseudotime, _ = run_sceptic_and_evaluate(
        data=X,
        labels=y_cont,
        label_list=time_axis,
        parameters=param_grid,
        method="xgboost",
        model_type="regression",
        cv_strategy="kfold",
        scale_features=scale_features,
    )

    metrics = compute_regression_metrics(pseudotime, y_cont)
    correlations = compute_correlation_metrics(pseudotime, y_cont)
    summary = {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mse": metrics["mse"],
        "spearman_r": correlations["spearman"][0],
        "pearson_r": correlations["pearson"][0],
        "kendall_tau": correlations["kendall"][0],
    }
    return summary


def _summarise_classification(
    X: np.ndarray,
    y_encoded: np.ndarray,
    time_axis: np.ndarray,
) -> Dict[str, Any]:
    """Fit the classification workflow and compute pseudotime scores."""
    cm, y_pred, pseudotime, _ = run_sceptic_and_evaluate(
        data=X,
        labels=y_encoded,
        label_list=time_axis,
        method="xgboost",
        model_type="classification",
        cv_strategy="kfold",
    )
    metrics = evaluate_sceptic_results(
        confusion_matrix=cm,
        y_true=y_encoded,
        y_pred=y_pred.astype(int),
        pseudotime=pseudotime,
        true_time=np.array([time_axis[idx] for idx in y_encoded]),
        include_regression=True,
        verbose=False,
    )
    summary = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "spearman_r": metrics["spearman"][0],
    }
    return summary


def main() -> None:
    X, encoded_labels, continuous_labels, ordered_time = _load_scgem()

    classification_metrics = _summarise_classification(X, encoded_labels, ordered_time)
    legacy_reg_metrics = _summarise_regression(
        X,
        continuous_labels,
        ordered_time,
        scale_features=False,
        param_grid=LEGACY_REGRESSION_GRID,
    )
    enhanced_reg_metrics = _summarise_regression(
        X,
        continuous_labels,
        ordered_time,
        scale_features=True,
        param_grid=None,
    )

    print("Classification baseline:", classification_metrics)
    print("Legacy regression:", legacy_reg_metrics)
    print("Enhanced regression:", enhanced_reg_metrics)


if __name__ == "__main__":
    main()
