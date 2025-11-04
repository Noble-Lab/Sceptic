"""
Simple test script for new Sceptic features: LOTO and Regression
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys
sys.path.insert(0, 'src')
from sceptic import run_sceptic_and_evaluate
from sceptic.evaluation import compute_correlation_metrics, compute_regression_metrics

print("=" * 80)
print("Testing New Sceptic Features: LOTO and Regression")
print("=" * 80)

# === Load example dataset ===
print("\nLoading scGEM example data...")
data_concat = np.loadtxt("example_data/scGEM/expression.txt")
y = np.loadtxt("example_data/scGEM/expression_type.txt")

# Convert labels to categorical values
lab = preprocessing.LabelEncoder()
label = lab.fit_transform(y)

time_dictionary = {1.0:8, 2.0:16, 3.0:24, 4.0:30, 0.0:0}
y_mapped = pd.Series(np.unique(label)).map(time_dictionary).to_numpy()
label_list = np.transpose(np.unique(y_mapped))

print(f"Data shape: {data_concat.shape}")
print(f"Time points: {label_list}")

# === Parameters ===
parameters = {
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.3],
    "n_estimators": [100]
}

# === Test 1: Classification + k-fold (existing behavior) ===
print("\n" + "=" * 80)
print("Test 1: Classification + k-fold (existing behavior)")
print("=" * 80)
try:
    cm, label_predicted, pseudotime, sceptic_prob = run_sceptic_and_evaluate(
        data=data_concat,
        labels=label,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False
    )

    print(f"✓ Success!")
    print(f"  Confusion Matrix:\n{cm}")

    corr_metrics = compute_correlation_metrics(label, pseudotime)
    print(f"  Spearman: r={corr_metrics['spearman'][0]:.4f}, p={corr_metrics['spearman'][1]:.4e}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# === Test 2: Classification + LOTO (new feature) ===
print("\n" + "=" * 80)
print("Test 2: Classification + LOTO (new feature)")
print("=" * 80)
try:
    cm, label_predicted, pseudotime, sceptic_prob = run_sceptic_and_evaluate(
        data=data_concat,
        labels=label,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False,
        cv_strategy="loto"
    )

    print(f"✓ Success!")
    print(f"  Confusion Matrix:\n{cm}")

    corr_metrics = compute_correlation_metrics(label, pseudotime)
    print(f"  Spearman: r={corr_metrics['spearman'][0]:.4f}, p={corr_metrics['spearman'][1]:.4e}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# === Test 3: Regression + k-fold (new feature) ===
print("\n" + "=" * 80)
print("Test 3: Regression + k-fold (new feature)")
print("=" * 80)
try:
    cm, label_predicted, pseudotime, sceptic_prob = run_sceptic_and_evaluate(
        data=data_concat,
        labels=label,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False,
        model_type="regression"
    )

    print(f"✓ Success!")
    print(f"  Pseudotime range: [{pseudotime.min():.2f}, {pseudotime.max():.2f}]")
    print(f"  True label range: [{label.min():.2f}, {label.max():.2f}]")

    corr_metrics = compute_correlation_metrics(label, pseudotime)
    reg_metrics = compute_regression_metrics(label, pseudotime)
    print(f"  Spearman: r={corr_metrics['spearman'][0]:.4f}, p={corr_metrics['spearman'][1]:.4e}")
    print(f"  MAE: {reg_metrics['mae']:.4f}, RMSE: {reg_metrics['rmse']:.4f}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# === Test 4: Regression + LOTO (new feature) ===
print("\n" + "=" * 80)
print("Test 4: Regression + LOTO (new feature)")
print("=" * 80)
try:
    cm, label_predicted, pseudotime, sceptic_prob = run_sceptic_and_evaluate(
        data=data_concat,
        labels=label,
        label_list=label_list,
        parameters=parameters,
        method="xgboost",
        use_gpu=False,
        cv_strategy="loto",
        model_type="regression"
    )

    print(f"✓ Success!")
    print(f"  Pseudotime range: [{pseudotime.min():.2f}, {pseudotime.max():.2f}]")

    corr_metrics = compute_correlation_metrics(label, pseudotime)
    reg_metrics = compute_regression_metrics(label, pseudotime)
    print(f"  Spearman: r={corr_metrics['spearman'][0]:.4f}, p={corr_metrics['spearman'][1]:.4e}")
    print(f"  MAE: {reg_metrics['mae']:.4f}, RMSE: {reg_metrics['rmse']:.4f}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
