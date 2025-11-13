"""Tests for the standalone SCEPTIC training/prediction helpers."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline

from sceptic import predict_sceptic_model, train_sceptic_model


def test_train_and_predict_classification():
    """Training without CV should produce valid class probabilities and pseudotime."""

    X, y = make_blobs(n_samples=60, centers=3, random_state=0, cluster_std=0.60)
    label_list = np.array([0, 4, 8])

    model = train_sceptic_model(
        data=X,
        labels=y,
        label_list=label_list,
        method="svm",
        scale_features=True,
    )

    preds = predict_sceptic_model(model, X[:10])

    assert preds.label_predicted.shape == (10,)
    assert preds.probabilities.shape == (10, len(label_list))
    assert preds.pseudotime.shape == (10,)
    assert preds.probabilities.min() >= 0.0
    assert preds.probabilities.max() <= 1.0


def test_train_and_predict_regression():
    """Regression path should return continuous pseudotime values."""

    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 4))
    weights = np.array([0.2, -0.5, 1.0, 0.7])
    y = X @ weights + rng.normal(scale=0.05, size=80)

    model = train_sceptic_model(
        data=X,
        labels=y,
        method="xgboost",
        model_type="regression",
        parameters={"n_estimators": 20, "max_depth": 2},
        scale_features=True,
    )

    preds = predict_sceptic_model(model, X[:5])

    assert preds.label_predicted is None
    assert preds.probabilities is None
    assert preds.pseudotime.shape == (5,)


def test_dim_reduction_inserts_pca_step():
    X, y = make_blobs(n_samples=100, centers=3, random_state=1, cluster_std=1.2)

    model = train_sceptic_model(
        data=X,
        labels=y,
        method="svm",
        dim_reduction=2,
        scale_features=True,
    )

    assert isinstance(model.estimator, Pipeline)
    assert "pca" in model.estimator.named_steps
