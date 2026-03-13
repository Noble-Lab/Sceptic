import numpy as np
import pytest

import sceptic.sceptic as sceptic_module


class _DummyGridSearch:
    def __init__(self, estimator, parameters, cv):
        self.estimator = estimator

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X[:, 0] >= 3).astype(int)

    def predict_proba(self, X):
        predicted = self.predict(X)
        probabilities = np.zeros((X.shape[0], 2), dtype=float)
        probabilities[predicted == 0] = np.array([0.8, 0.2])
        probabilities[predicted == 1] = np.array([0.3, 0.7])
        return probabilities


class _TwoFoldSingleClassTestKFold:
    def __init__(self, n_splits, random_state=None, shuffle=False):
        self.n_splits = n_splits

    def split(self, data):
        yield np.array([2, 3, 4, 5]), np.array([0, 1])
        yield np.array([0, 1, 4, 5]), np.array([2, 3])
        yield np.array([0, 1, 2, 3]), np.array([4, 5])


def test_xgboost_uses_binary_objective_for_two_classes():
    model = sceptic_module._create_xgb_classifier(num_classes=2, use_gpu=False)

    assert model.get_params()["objective"] == "binary:logistic"


def test_two_timepoint_classification_warns_and_computes_pseudotime(monkeypatch):
    monkeypatch.setattr(sceptic_module, "GridSearchCV", _DummyGridSearch)
    monkeypatch.setattr(sceptic_module, "KFold", _TwoFoldSingleClassTestKFold)

    data = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    label_list = np.array([0.0, 10.0], dtype=float)

    with pytest.warns(UserWarning, match="Two-timepoint classification is supported"):
        cm, label_predicted, pseudotime, sceptic_prob = sceptic_module.run_sceptic_and_evaluate(
            data=data,
            labels=labels,
            label_list=label_list,
            parameters={"max_depth": [3]},
            method="xgboost",
            use_gpu=False,
        )

    assert cm.shape == (2, 2)
    assert np.array_equal(label_predicted, np.array([0, 0, 1, 1, 1, 1], dtype=float))
    assert sceptic_prob.shape == (6, 2)
    assert np.allclose(pseudotime[:4], np.array([2.0, 2.0, 7.0, 7.0]))
    assert np.allclose(pseudotime[4:], np.array([7.0, 7.0]))
