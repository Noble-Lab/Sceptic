# Sceptic Examples - Development Features

This directory contains examples demonstrating Sceptic's new development features.

## Available Examples

### 1. `scGEM_loto_regression_example.py` (NEW - Development Branch)

**Demonstrates**: LOTO evaluation and direct regression features

**Features shown**:
- Leave-One-Time-Out (LOTO) cross-validation
- Direct XGBoost regression model
- All 4 combinations: Classification/Regression × K-Fold/LOTO
- Performance comparison and interpretation

**Usage**:
```bash
# Activate environment
source ~/.venvs/sceptic/bin/activate

# Run example
python examples/scGEM_loto_regression_example.py
```

**Expected output**:
- Performance metrics for all 4 methods
- Comparison table showing Spearman r, Pearson r, and MAE
- Key insights about temporal generalization

**Results** (scGEM dataset):
| Method | Spearman r | Notes |
|--------|------------|-------|
| Classification + K-Fold | 0.95 | Baseline (existing) |
| Classification + LOTO | 0.69 | Tests unseen time points |
| Regression + K-Fold | 0.95 | Best overall |
| Regression + LOTO | 0.39 | Hardest - temporal extrapolation |

---

### 2. `basic_usage.ipynb`

**Status**: Existing example (main branch)

**Demonstrates**: Standard Sceptic usage with classification

---

### 3. `custom_evaluation.ipynb`

**Status**: Existing example (main branch)

**Demonstrates**: Custom evaluation metrics and plotting

---

## New Feature APIs

### LOTO Evaluation

```python
from sceptic import run_sceptic_and_evaluate

cm, pred, ptime, prob = run_sceptic_and_evaluate(
    data=data,
    labels=labels,
    label_list=label_list,
    method="xgboost",
    cv_strategy="loto"  # NEW: Leave-one-time-out
)
```

**What it does**: Iteratively holds out each time point as test data, trains on all others.

**Use case**: Test model's ability to generalize to completely unseen temporal states.

### Direct Regression

```python
from sceptic import run_sceptic_and_evaluate

cm, pred, ptime, prob = run_sceptic_and_evaluate(
    data=data,
    labels=labels,
    label_list=label_list,
    method="xgboost",
    model_type="regression"  # NEW: Direct regression
)
```

**What it does**: XGBoost regressor for continuous time prediction (no classification step).

**Use case**: Direct modeling of continuous temporal processes, GPU-accelerated.

**Note**: For regression, `cm` and `prob` will be `None`.

### Combined (LOTO + Regression)

```python
cm, pred, ptime, prob = run_sceptic_and_evaluate(
    data=data,
    labels=labels,
    label_list=label_list,
    method="xgboost",
    cv_strategy="loto",
    model_type="regression"
)
```

**What it does**: Predicts continuous time values for completely unseen time points.

**Use case**: Hardest test - true temporal extrapolation.

---

## Development Branch Info

**Branch**: `dev/loto-and-regression`

**Status**: Development/Testing (not yet merged to main)

**Key changes**:
- Added `cv_strategy` parameter: "kfold" (default) or "loto"
- Added `model_type` parameter: "classification" (default) or "regression"
- Fully backward compatible

**Testing**: All features tested on scGEM dataset with excellent results.

---

## Next Steps

To use these features:

1. **Switch to development branch**:
   ```bash
   git checkout dev/loto-and-regression
   ```

2. **Install/reinstall Sceptic** (if needed):
   ```bash
   pip install -e .
   ```

3. **Run examples**:
   ```bash
   python examples/scGEM_loto_regression_example.py
   ```

4. **For your own data**:
   - Use the scGEM example as a template
   - Replace data loading with your dataset
   - Choose appropriate `cv_strategy` and `model_type`

---

## Questions?

See `DEVELOPMENT_PLAN.md` in the root directory for detailed implementation notes.

See `test/dev/TEST_RESULTS.md` for comprehensive test results and performance analysis.
