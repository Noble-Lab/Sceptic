"""
Internal benchmarking script for the embryo beta dataset from TemporalVAE.

Loads the external CSVs, aligns samples, and evaluates classification and
regression pipelines using the current Sceptic implementation.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing

MODULE_DIR = Path(__file__).resolve().parents[2] / "src" / "sceptic"

DATA_ROOT = Path(
    "/Users/gangli/Documents/GitHub/2025_sceptic2/src/20251031-TemporalVAE"
) / "data_fromPsupertime"

LEGACY_REGRESSION_GRID: Dict[str, Any] = {
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.3],
    "n_estimators": [100],
}


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


def _load_temporalvae() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and align the TemporalVAE embryo beta dataset."""
    expr = pd.read_csv(DATA_ROOT / "embryoBeta_X.csv")
    label_df = pd.read_csv(DATA_ROOT / "embryoBeta_Y.csv")

    sample_expr = expr.set_index("Symbol").T
    sample_expr.index.name = "sample"

    label_df = label_df.rename(columns={"Unnamed: 0": "sample"}).set_index("sample")
    aligned_labels = label_df.loc[sample_expr.index]

    X = sample_expr.to_numpy(dtype=float)
    y_continuous = aligned_labels["time"].to_numpy(dtype=float)

    encoder = preprocessing.LabelEncoder()
    y_encoded = encoder.fit_transform(y_continuous)
    ordered_time = np.sort(np.unique(y_continuous))

    return X, y_encoded, y_continuous, ordered_time


def _summarise_regression(
    X: np.ndarray,
    y_cont: np.ndarray,
    time_axis: np.ndarray,
    *,
    scale_features: bool,
    param_grid: Optional[Dict[str, Any]],
    cv_strategy: str,
) -> Dict[str, Any]:
    """Run regression flow and gather metrics."""
    _, _, pseudotime, _ = run_sceptic_and_evaluate(
        data=X,
        labels=y_cont,
        label_list=time_axis,
        parameters=param_grid,
        method="xgboost",
        model_type="regression",
        cv_strategy=cv_strategy,
        scale_features=scale_features,
    )

    metrics = compute_regression_metrics(pseudotime, y_cont)
    correlations = compute_correlation_metrics(pseudotime, y_cont)
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mse": metrics["mse"],
        "spearman_r": correlations["spearman"][0],
        "pearson_r": correlations["pearson"][0],
        "kendall_tau": correlations["kendall"][0],
    }


def _summarise_classification(
    X: np.ndarray,
    y_encoded: np.ndarray,
    time_axis: np.ndarray,
    *,
    cv_strategy: str,
) -> Dict[str, Any]:
    """Run classification flow and compute standard metrics."""
    cm, y_pred, pseudotime, _ = run_sceptic_and_evaluate(
        data=X,
        labels=y_encoded,
        label_list=time_axis,
        method="xgboost",
        model_type="classification",
        cv_strategy=cv_strategy,
    )

    true_times = np.array([time_axis[idx] for idx in y_encoded])
    metrics = evaluate_sceptic_results(
        confusion_matrix=cm,
        y_true=y_encoded,
        y_pred=y_pred.astype(int),
        pseudotime=pseudotime,
        true_time=true_times,
        include_regression=True,
        verbose=False,
    )
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "spearman_r": metrics["spearman"][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Sceptic regression pipelines on TemporalVAE data."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["kfold", "loto"],
        default=["kfold", "loto"],
        help="Cross-validation strategies to evaluate (default: both).",
    )
    args = parser.parse_args()

    X, encoded_labels, continuous_labels, ordered_times = _load_temporalvae()

    for strategy in args.strategies:
        print(f"\n=== {strategy.upper()} ===")
        classification_metrics = _summarise_classification(
            X,
            encoded_labels,
            ordered_times,
            cv_strategy=strategy,
        )
        legacy_regression = _summarise_regression(
            X,
            continuous_labels,
            ordered_times,
            scale_features=False,
            param_grid=LEGACY_REGRESSION_GRID,
            cv_strategy=strategy,
        )
        enhanced_regression = _summarise_regression(
            X,
            continuous_labels,
            ordered_times,
            scale_features=True,
            param_grid=None,
            cv_strategy=strategy,
        )

        print("Classification baseline:", classification_metrics)
        print("Legacy regression:", legacy_regression)
        print("Enhanced regression:", enhanced_regression)


if __name__ == "__main__":
    main()
