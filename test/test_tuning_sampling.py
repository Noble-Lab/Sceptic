import numpy as np
from sklearn.datasets import make_classification

from sceptic.sceptic import (
    _maybe_subsample_training_data,
    run_sceptic_and_evaluate,
)

def test_run_sceptic_with_sampling_classification():
    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_redundant=0,
        random_state=0,
    )
    label_list = np.array(sorted(np.unique(y)))
    params = {"C": [0.1, 1.0], "kernel": ["linear"]}

    cm, pred, pseudo, prob = run_sceptic_and_evaluate(
        data=X,
        labels=y,
        label_list=label_list,
        parameters=params,
        method="svm",
        cv_strategy="kfold",
        model_type="classification",
        eFold=3,
        iFold=2,
        tuning_sample_size=30,
        tuning_random_state=1,
    )

    assert cm.shape == (len(label_list), len(label_list))
    assert pred.shape[0] == y.shape[0]
    assert pseudo.shape[0] == y.shape[0]
    assert prob.shape[0] == y.shape[0]


def test_subsample_helper_stratifies_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.linspace(0, 1, 100)
    X_sub, y_sub = _maybe_subsample_training_data(
        X,
        y,
        max_samples=25,
        random_state=0,
        is_regression=True,
        tuning_label_bins=5,
    )

    assert X_sub.shape[0] == 25
    assert y_sub.shape[0] == 25
    assert y_sub.min() >= y.min() and y_sub.max() <= y.max()
