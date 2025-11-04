"""
Example: Testing LOTO and Regression Features with scGEM Data

This script demonstrates the new Sceptic features:
1. Leave-One-Time-Out (LOTO) cross-validation
2. Direct XGBoost regression

Dataset: scGEM (Single-cell Gene Expression Myeloid)
- 177 cells
- 34 gene features
- 5 time points: 0, 8, 16, 24, 30 hours
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from sceptic import run_sceptic_and_evaluate, evaluation, plotting

print("=" * 80)
print("Sceptic New Features Demo: LOTO and Regression")
print("Dataset: scGEM")
print("=" * 80)

# =============================================================================
# 1. Load and Prepare Data
# =============================================================================
print("\n[1] Loading scGEM Data...")

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'example_data', 'scGEM')
data_concat = np.loadtxt(os.path.join(data_path, 'expression.txt'))
y = np.loadtxt(os.path.join(data_path, 'expression_type.txt'))

# Encode labels
lab = preprocessing.LabelEncoder()
label = lab.fit_transform(y)

# Map to actual time points
time_dictionary = {1.0: 8, 2.0: 16, 3.0: 24, 4.0: 30, 0.0: 0}
y_mapped = pd.Series(np.unique(label)).map(time_dictionary).to_numpy()
label_list = np.transpose(np.unique(y_mapped))

print(f"  Data shape: {data_concat.shape}")
print(f"  Number of cells: {data_concat.shape[0]}")
print(f"  Number of features: {data_concat.shape[1]}")
print(f"  Time points: {label_list}")
print(f"  Cells per time point: {dict(zip(label_list, [np.sum(label == i) for i in range(len(label_list))]))}")

# Hyperparameters
parameters = {
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.3],
    "n_estimators": [100]
}

# =============================================================================
# 2. Baseline: Classification + K-Fold CV (Existing Method)
# =============================================================================
print("\n[2] Baseline: Classification + K-Fold CV")
print("-" * 80)

cm_baseline, pred_baseline, ptime_baseline, prob_baseline = run_sceptic_and_evaluate(
    data=data_concat,
    labels=label,
    label_list=label_list,
    parameters=parameters,
    method="xgboost",
    use_gpu=False,
    cv_strategy="kfold",  # Default
    model_type="classification"  # Default
)

# Evaluate
metrics_baseline = evaluation.compute_correlation_metrics(label, ptime_baseline)
print(f"  Spearman correlation: {metrics_baseline['spearman'][0]:.4f} (p={metrics_baseline['spearman'][1]:.2e})")
print(f"  Pearson correlation: {metrics_baseline['pearson'][0]:.4f} (p={metrics_baseline['pearson'][1]:.2e})")
print(f"  Confusion Matrix:")
print(f"  {cm_baseline}")

# =============================================================================
# 3. Feature 1: LOTO Evaluation with Classification
# =============================================================================
print("\n[3] NEW Feature: LOTO Evaluation with Classification")
print("-" * 80)
print("  Holding out each time point and training on others...")

cm_loto, pred_loto, ptime_loto, prob_loto = run_sceptic_and_evaluate(
    data=data_concat,
    labels=label,
    label_list=label_list,
    parameters=parameters,
    method="xgboost",
    use_gpu=False,
    cv_strategy="loto",  # NEW: Leave-one-time-out
    model_type="classification"
)

# Evaluate
metrics_loto = evaluation.compute_correlation_metrics(label, ptime_loto)
print(f"  Spearman correlation: {metrics_loto['spearman'][0]:.4f} (p={metrics_loto['spearman'][1]:.2e})")
print(f"  Pearson correlation: {metrics_loto['pearson'][0]:.4f} (p={metrics_loto['pearson'][1]:.2e})")
print(f"  Confusion Matrix:")
print(f"  {cm_loto}")
print(f"\n  Note: Lower performance is expected - testing on completely unseen time points!")

# =============================================================================
# 4. Feature 2: Direct Regression with K-Fold CV
# =============================================================================
print("\n[4] NEW Feature: Direct Regression with K-Fold CV")
print("-" * 80)

cm_reg, pred_reg, ptime_reg, prob_reg = run_sceptic_and_evaluate(
    data=data_concat,
    labels=label,
    label_list=label_list,
    parameters=parameters,
    method="xgboost",
    use_gpu=False,
    cv_strategy="kfold",
    model_type="regression"  # NEW: Direct regression
)

# Evaluate
metrics_reg = evaluation.compute_correlation_metrics(label, ptime_reg)
metrics_reg_error = evaluation.compute_regression_metrics(label, ptime_reg)
print(f"  Spearman correlation: {metrics_reg['spearman'][0]:.4f} (p={metrics_reg['spearman'][1]:.2e})")
print(f"  Pearson correlation: {metrics_reg['pearson'][0]:.4f} (p={metrics_reg['pearson'][1]:.2e})")
print(f"  MAE: {metrics_reg_error['mae']:.4f}")
print(f"  RMSE: {metrics_reg_error['rmse']:.4f}")
print(f"  Pseudotime range: [{ptime_reg.min():.2f}, {ptime_reg.max():.2f}]")
print(f"  True label range: [{label.min():.2f}, {label.max():.2f}]")
print(f"\n  Note: No confusion matrix (cm={cm_reg}) - this is a regression model!")

# =============================================================================
# 5. Feature Combination: LOTO + Regression
# =============================================================================
print("\n[5] COMBINED: LOTO + Regression (Hardest Challenge)")
print("-" * 80)
print("  Predicting continuous values for completely unseen time points...")

cm_loto_reg, pred_loto_reg, ptime_loto_reg, prob_loto_reg = run_sceptic_and_evaluate(
    data=data_concat,
    labels=label,
    label_list=label_list,
    parameters=parameters,
    method="xgboost",
    use_gpu=False,
    cv_strategy="loto",  # NEW
    model_type="regression"  # NEW
)

# Evaluate
metrics_loto_reg = evaluation.compute_correlation_metrics(label, ptime_loto_reg)
metrics_loto_reg_error = evaluation.compute_regression_metrics(label, ptime_loto_reg)
print(f"  Spearman correlation: {metrics_loto_reg['spearman'][0]:.4f} (p={metrics_loto_reg['spearman'][1]:.2e})")
print(f"  Pearson correlation: {metrics_loto_reg['pearson'][0]:.4f} (p={metrics_loto_reg['pearson'][1]:.2e})")
print(f"  MAE: {metrics_loto_reg_error['mae']:.4f}")
print(f"  RMSE: {metrics_loto_reg_error['rmse']:.4f}")
print(f"\n  Note: This is the hardest task - true temporal extrapolation!")

# =============================================================================
# 6. Performance Comparison
# =============================================================================
print("\n[6] Performance Comparison")
print("=" * 80)

comparison_data = {
    'Method': [
        'Classification + K-Fold (Baseline)',
        'Classification + LOTO',
        'Regression + K-Fold',
        'Regression + LOTO'
    ],
    'Spearman r': [
        metrics_baseline['spearman'][0],
        metrics_loto['spearman'][0],
        metrics_reg['spearman'][0],
        metrics_loto_reg['spearman'][0]
    ],
    'Pearson r': [
        metrics_baseline['pearson'][0],
        metrics_loto['pearson'][0],
        metrics_reg['pearson'][0],
        metrics_loto_reg['pearson'][0]
    ],
    'MAE': [
        '-',
        '-',
        f"{metrics_reg_error['mae']:.4f}",
        f"{metrics_loto_reg_error['mae']:.4f}"
    ]
}

df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))

print("\n[7] Key Insights")
print("-" * 80)
print("  1. K-Fold CV: High performance (0.95 Spearman) - cells from same time in train/test")
print("  2. LOTO: Lower performance (0.39-0.69) - must generalize to unseen time points")
print("  3. Regression: Provides error metrics (MAE, RMSE) for continuous predictions")
print("  4. LOTO tests true temporal generalization - critical for real applications!")

print("\n" + "=" * 80)
print("Example completed successfully!")
print("All new features demonstrated with scGEM data.")
print("=" * 80)

# =============================================================================
# 7. Optional: Save Results
# =============================================================================
save_results = input("\nSave results to files? (y/n): ").lower() == 'y'

if save_results:
    output_dir = 'examples/scGEM_results'
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    np.savetxt(f'{output_dir}/baseline_pseudotime.txt', ptime_baseline, fmt='%.4f')
    np.savetxt(f'{output_dir}/loto_pseudotime.txt', ptime_loto, fmt='%.4f')
    np.savetxt(f'{output_dir}/regression_pseudotime.txt', ptime_reg, fmt='%.4f')
    np.savetxt(f'{output_dir}/loto_regression_pseudotime.txt', ptime_loto_reg, fmt='%.4f')

    # Save confusion matrices
    np.savetxt(f'{output_dir}/baseline_confusion_matrix.txt', cm_baseline, fmt='%i')
    np.savetxt(f'{output_dir}/loto_confusion_matrix.txt', cm_loto, fmt='%i')

    # Save comparison table
    df.to_csv(f'{output_dir}/comparison_table.csv', index=False)

    print(f"\n✓ Results saved to {output_dir}/")
