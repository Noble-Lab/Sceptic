import numpy as np
from sklearn.pipeline import Pipeline

import sceptic.sceptic as sceptic_module


def _make_dummy_grid_search(captured, *, return_proba=False):
    """Factory returning a fake GridSearchCV constructor."""

    class _DummyEstimator:
        def __init__(self):
            self._n_classes = 0

        def fit(self, X, y):
            if return_proba:
                self._n_classes = len(np.unique(y))
            return self

        def predict(self, X):
            if return_proba:
                return np.zeros(X.shape[0], dtype=int)
            return np.zeros(X.shape[0], dtype=float)

        def predict_proba(self, X):
            probs = np.full((X.shape[0], self._n_classes), 1.0 / self._n_classes)
            return probs

    def _constructor(estimator, param_grid, cv):
        captured["estimator"] = estimator
        captured["param_grid"] = param_grid
        return _DummyEstimator()

    return _constructor


def test_regression_scales_features_by_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sceptic_module,
        "GridSearchCV",
        _make_dummy_grid_search(captured, return_proba=False),
    )

    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.linspace(0, 19, 20, dtype=float)

    sceptic_module.run_sceptic_and_evaluate(
        data=X,
        labels=y,
        method="xgboost",
        model_type="regression",
        cv_strategy="kfold",
        scale_features=None,
    )

    estimator = captured["estimator"]
    assert isinstance(estimator, Pipeline)
    assert estimator.steps[0][0] == "scaler"
    assert estimator.steps[1][0] == "regressor"
    assert all(key.startswith("regressor__") for key in captured["param_grid"])
    assert captured["param_grid"]["regressor__max_depth"] == [3, 4]


def test_classification_does_not_scale_by_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sceptic_module,
        "GridSearchCV",
        _make_dummy_grid_search(captured, return_proba=True),
    )

    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0, 1] * 10, dtype=int)

    sceptic_module.run_sceptic_and_evaluate(
        data=X,
        labels=y,
        method="svm",
        model_type="classification",
        cv_strategy="kfold",
        scale_features=None,
    )

    estimator = captured["estimator"]
    from sklearn import svm  # local import to avoid polluting namespace

    assert isinstance(estimator, svm.SVC)
    assert captured["param_grid"] == {
        "C": [1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"],
    }


def test_regression_loto_uses_shallow_grid(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sceptic_module,
        "GridSearchCV",
        _make_dummy_grid_search(captured, return_proba=False),
    )

    X = np.arange(54, dtype=float).reshape(18, 3)
    y = np.tile(np.array([0.0, 5.0, 10.0], dtype=float), 6)

    sceptic_module.run_sceptic_and_evaluate(
        data=X,
        labels=y,
        method="xgboost",
        model_type="regression",
        cv_strategy="loto",
        scale_features=None,
    )

    estimator = captured["estimator"]
    assert isinstance(estimator, Pipeline)
    expected = {
        "max_depth": [2, 3],
        "min_child_weight": [3, 5],
        "learning_rate": [0.05],
        "n_estimators": [200],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1.0],
        "reg_alpha": [0.0, 0.5],
    }
    assert captured["param_grid"] == {
        f"regressor__{key}": value for key, value in expected.items()
    }
