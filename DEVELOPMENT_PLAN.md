# Sceptic Model Enhancement Plan: Development Version

**Date**: 2025-11-04
**Version**: Internal Development v0.5.0-dev
**Status**: Planning Phase

---

## Overview

This document outlines the plan to add two major features to the Sceptic pseudotime prediction model as internal development versions before systematic testing and public release.

### Current State
- **Version**: v0.4.0 (public)
- **Model Type**: Hybrid classification-to-continuous model
  - Uses multi-class classification (SVM/XGBoost)
  - Generates continuous pseudotime via probability-weighted averaging
- **Evaluation**: Nested 3-fold cross-validation (external) with 4-fold GridSearchCV (internal)
- **Location**: `/Users/gangli/Documents/GitHub/Sceptic`

### Proposed Features

#### Feature 1: Leave-One-Time-Out (LOTO) Evaluation
**Objective**: Implement temporal hold-out validation where each time point is iteratively held out as test data.

**Current Limitation**: Standard k-fold CV doesn't respect temporal structure - cells from the same time point can appear in both training and test sets, potentially inflating performance metrics.

**Proposed Solution**:
- For each unique time point in the dataset:
  - Hold out ALL cells from that time point as test set
  - Train model on cells from all other time points
  - Predict pseudotime for held-out cells
- Aggregate predictions across all iterations
- Evaluate model's ability to generalize to unseen temporal states

**Example**: For dataset with [0hr, 8hr, 16hr, 24hr, 30hr]
- Iteration 1: Test on 0hr → Train on [8hr, 16hr, 24hr, 30hr]
- Iteration 2: Test on 8hr → Train on [0hr, 16hr, 24hr, 30hr]
- ... and so on

---

#### Feature 2: Direct Regression Model
**Objective**: Implement true regression modeling instead of classification-based approach.

**Current Limitation**:
- Current model treats time as discrete classes
- Relies on probability weighting to generate continuous output
- May not fully capture continuous temporal dynamics
- Requires time labels to be discrete/categorical

**Proposed Solution**:
- Implement direct regression model using **XGBoost Regressor** only
  - GPU-accelerated: `xgb.XGBRegressor` with `tree_method='gpu_hist'`
  - Objective: `reg:squarederror` for continuous time prediction
  - Consistent with existing Sceptic classifier (XGBoost-based)
- Output continuous pseudotime directly without probability intermediates
- Enable modeling of truly continuous temporal processes
- Scalable to large single-cell datasets via GPU support

---

## Technical Implementation

### 1. Code Organization Strategy

**Goal**: Keep development versions separate from public code without breaking existing functionality.

#### Unified API Design (Feature Branch)
```
Sceptic/
├── src/sceptic/
│   ├── sceptic.py                    # MODIFY: Add model_type and cv_strategy parameters
│   ├── evaluation.py                 # Keep existing (already has regression metrics)
│   └── __init__.py                   # No changes needed
├── examples/
│   ├── dev_loto_evaluation.ipynb     # NEW: LOTO demo
│   └── dev_regression_model.ipynb    # NEW: Regression demo
└── test/
    └── dev/                           # NEW: Development tests
        ├── test_loto.py
        ├── test_regression.py
        └── test_integration.py
```

**Git Workflow**:
```bash
# Create development branch
cd /Users/gangli/Documents/GitHub/Sceptic
git checkout -b dev/loto-and-regression
git push -u origin dev/loto-and-regression

# Work on features in this branch
# Do NOT merge to main until systematic testing complete
```

**Key Design Principle**: Modify existing `run_sceptic_and_evaluate()` to support new modes while preserving backward compatibility.

---

### 2. Feature 1: LOTO Evaluation Implementation

#### 2.1 Modify: `src/sceptic/sceptic.py`

**Updated Function Signature**:
```python
def run_sceptic_and_evaluate(data, labels, label_list=None, parameters=None,
                            method="svm", use_gpu=False,
                            cv_strategy="kfold", eFold=3, iFold=4):
    """
    Sceptic pseudotime prediction with flexible CV strategies.

    Parameters
    ----------
    data : np.ndarray
        Cell-by-feature matrix (n_cells, n_features)
    labels : np.ndarray
        True time labels for each cell
    label_list : list, optional
        Ordered unique time points. Auto-inferred if None.
    parameters : dict, optional
        Hyperparameter grid for GridSearchCV
    method : str
        "svm" or "xgboost"
    use_gpu : bool
        Use GPU for XGBoost
    cv_strategy : str, optional (NEW)
        "kfold" (default) or "loto" (leave-one-time-out)
    eFold : int, optional
        Number of external CV folds (used when cv_strategy="kfold")
    iFold : int, optional
        Number of internal GridSearchCV folds

    Returns
    -------
    cm : np.ndarray
        Confusion matrix
    label_predicted : np.ndarray
        Predicted class labels
    pseudotime : np.ndarray
        Continuous pseudotime predictions
    sceptic_prob : np.ndarray
        Class probabilities (n_cells × n_timepoints)
    """
```

**Implementation Steps for LOTO**:
1. Add conditional logic: `if cv_strategy == "loto":`
2. For each time point in `label_list`:
   - Create boolean mask: `test_mask = (labels == time_point)`
   - Split data: `X_train, y_train = data[~test_mask], labels[~test_mask]`
   - Split data: `X_test, y_test = data[test_mask], labels[test_mask]`
   - Create new `label_list_train` excluding held-out time (for classification mode)
   - **Run GridSearchCV on training data** (re-tune hyperparameters per iteration)
   - Predict on test data
   - Store predictions with original indices
3. Aggregate results across all iterations
4. Return results based on model_type:
   - **Classification mode**: Build confusion matrix, return cm, predicted_labels, pseudotime, probabilities
   - **Regression mode**: Return only pseudotime (no confusion matrix, no predicted_labels, no probabilities)

**Key Considerations**:
- **Hyperparameter tuning**: Each LOTO iteration runs GridSearchCV independently
- **Label encoding**: Handle dynamic label lists (excluding held-out time) - only for classification
- **Probability output**: For classification, may have different class counts per iteration
- **Regression output**: For regression, only continuous pseudotime predictions (no classification involved)
- **Backward compatibility**: Default `cv_strategy="kfold"` preserves existing behavior

---

### 3. Feature 2: Direct Regression Implementation

#### 3.1 Modify: `src/sceptic/sceptic.py`

**Updated Function Signature** (continued from above):
```python
def run_sceptic_and_evaluate(data, labels, label_list=None, parameters=None,
                            method="svm", use_gpu=False,
                            cv_strategy="kfold", eFold=3, iFold=4,
                            model_type="classification"):
    """
    Sceptic pseudotime prediction with flexible CV strategies and model types.

    Parameters
    ----------
    ... (previous parameters)
    model_type : str, optional (NEW)
        "classification" (default) - existing behavior with probability weighting
        "regression" - direct XGBoost regression (continuous predictions)

    Returns
    -------
    Same as before for classification mode.
    For regression mode:
        - cm: None (no confusion matrix for regression)
        - label_predicted: None (no discrete predictions)
        - pseudotime: continuous predictions (n_cells,)
        - sceptic_prob: None (no probabilities for regression)
    """
```

**Model**: XGBoost Regressor only (GPU-accelerated) when `model_type="regression"`

**Default Hyperparameter Grid for Regression**:
```python
# When model_type="regression" and method="xgboost"
default_params_regression = {
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.3],
    'n_estimators': [100]
}
```

**Rationale for XGBoost-only**:
- Native GPU support for large single-cell datasets
- Consistent with existing Sceptic classifier implementation
- State-of-the-art performance on tabular data
- Built-in regularization prevents overfitting
- Simple hyperparameter search: 2×2×1 = 4 combinations (same as current defaults)

**Implementation Steps for Regression**:
1. Add conditional logic: `if model_type == "regression":`
2. Validate input data and labels (must be numeric)
3. Setup cross-validation (KFold or LOTO, using cv_strategy parameter)
4. For each fold:
   - Split train/test
   - Initialize XGBRegressor with `objective='reg:squarederror'`
   - If use_gpu: set `tree_method='gpu_hist'`
   - Run GridSearchCV on training data
   - Predict on test data (continuous values directly)
   - Store predictions with original indices
5. Aggregate predictions
6. Return results (pseudotime only, other outputs as None)

**Key Considerations**:
- No probability weighting needed - direct continuous output
- No confusion matrix or discrete labels for regression
- Evaluation uses correlation and error metrics (not classification metrics)
- Backward compatibility: Default `model_type="classification"` preserves existing behavior

#### 3.2 Evaluation Metrics

**Good news**: `src/sceptic/evaluation.py` already has regression metrics!
- Existing functions: `get_correlation_metrics()`, `get_regression_metrics()`
- Metrics available: Pearson, Spearman, Kendall correlations, MSE, MAE, RMSE
- No modifications needed to evaluation.py

---

## Development Workflow

### Phase 1: Environment Setup (Week 1)

**Tasks**:
1. Create development branch
2. Update development environment
3. Document dependencies
4. Setup testing framework

**Git Commands**:
```bash
cd /Users/gangli/Documents/GitHub/Sceptic
git checkout -b dev/loto-and-regression
git push -u origin dev/loto-and-regression

# Optional: Add development dependencies
# Create requirements-dev.txt
```

**Dependencies** (verify versions):
```
# Existing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
scipy>=1.7.0
matplotlib>=3.4.0

# Additional for development
pytest>=7.0.0
jupyter>=1.0.0
```

---

### Phase 2: Feature Implementation (Weeks 2-3)

#### Week 2: LOTO Evaluation
- [ ] Modify `run_sceptic_and_evaluate()` in `sceptic.py` to add `cv_strategy` parameter
- [ ] Implement LOTO logic with per-iteration hyperparameter tuning
- [ ] Handle edge cases (single time point, unbalanced data)
- [ ] Add unit tests in `test/dev/test_loto.py`
- [ ] Create example notebook: `examples/dev_loto_evaluation.ipynb`
- [ ] Test on scGEM example dataset

#### Week 3: Regression Model
- [ ] Extend `run_sceptic_and_evaluate()` in `sceptic.py` to add `model_type` parameter
- [ ] Implement XGBoost regressor with GPU support (`model_type="regression"`)
- [ ] Ensure regression works with both cv_strategy options (kfold and loto)
- [ ] Add unit tests in `test/dev/test_regression.py`
- [ ] Create example notebook: `examples/dev_regression_model.ipynb`
- [ ] Test on scGEM example dataset

---

### Phase 3: Integration Testing (Week 4)

**Test Cases**:
1. **Compatibility**: Ensure existing code still works
2. **LOTO Evaluation**:
   - Test with 3, 5, and 10 time points
   - Test with imbalanced time points (varying cell counts)
   - Compare metrics to standard CV
3. **Regression Model**:
   - Compare XGBoost regressor vs classifier
   - Evaluate on continuous vs discrete time labels
   - Test GPU acceleration performance
4. **Combined**: LOTO + Regression

**Example Test Script**:
```python
# test/dev/test_integration.py
import pytest
from sceptic import run_sceptic_and_evaluate

def test_loto_vs_standard_cv():
    # Load scGEM data
    data, labels, label_list = load_test_data()

    # Standard k-fold CV (existing behavior)
    cm1, pred1, pseudo1, prob1 = run_sceptic_and_evaluate(
        data, labels, label_list, method="xgboost", cv_strategy="kfold"
    )

    # LOTO evaluation (new feature)
    cm2, pred2, pseudo2, prob2 = run_sceptic_and_evaluate(
        data, labels, label_list, method="xgboost", cv_strategy="loto"
    )

    # Compare metrics
    assert pseudo1.shape == pseudo2.shape

def test_regression_vs_classification():
    # Load scGEM data
    data, labels, label_list = load_test_data()

    # Classification mode (existing behavior)
    cm1, pred1, pseudo1, prob1 = run_sceptic_and_evaluate(
        data, labels, label_list, method="xgboost", model_type="classification"
    )

    # Regression mode (new feature)
    cm2, pred2, pseudo2, prob2 = run_sceptic_and_evaluate(
        data, labels, label_list, method="xgboost", model_type="regression"
    )

    # Check outputs
    assert cm2 is None  # No confusion matrix for regression
    assert pseudo2 is not None  # Continuous predictions exist

def test_combined_loto_regression():
    # Test LOTO + Regression together
    data, labels, label_list = load_test_data()

    cm, pred, pseudo, prob = run_sceptic_and_evaluate(
        data, labels, label_list, method="xgboost",
        cv_strategy="loto", model_type="regression"
    )

    # Validate results
    assert pseudo.shape[0] == data.shape[0]
```

---

### Phase 4: Systematic Testing (Weeks 5-6)

**Testing Strategy**:

1. **Unit Tests**: Individual functions
   - `pytest test/dev/test_loto.py`
   - `pytest test/dev/test_regression.py`

2. **Integration Tests**: End-to-end workflows
   - `pytest test/dev/test_integration.py`

3. **Benchmark Tests**: Performance comparison
   - Classification vs Regression on multiple datasets
   - Standard CV vs LOTO evaluation
   - Document results in `test/dev/BENCHMARK_RESULTS.md`

4. **Edge Case Tests**:
   - Very few time points (2-3)
   - Many time points (>10)
   - Imbalanced data
   - High-dimensional features
   - Small sample sizes

5. **Validation**:
   - Reproduce existing results with new code
   - Ensure metrics are consistent
   - Check for overfitting/underfitting

---

### Phase 5: Documentation (Week 7)

**Documentation Tasks**:
- [ ] Write detailed docstrings (Google/NumPy style)
- [ ] Create tutorial notebooks
- [ ] Document API changes
- [ ] Write internal technical report comparing methods
- [ ] Update README (development version notes)

**Internal Report** (`docs/DEV_FEATURE_COMPARISON.md`):
- LOTO vs Standard CV: When to use each
- Regression vs Classification: Performance comparison
- Recommendations for different data types
- Known limitations and future work

---

## Testing & Validation Checklist

### Correctness Checks
- [ ] LOTO correctly holds out all cells from target time point
- [ ] Regression models output continuous values in correct range
- [ ] Predictions align with original cell indices
- [ ] Cross-validation folds are non-overlapping
- [ ] Metrics are computed correctly

### Performance Checks
- [ ] LOTO runs in reasonable time (< 5x standard CV)
- [ ] Regression models converge
- [ ] GPU acceleration works (if enabled)
- [ ] Memory usage is acceptable

### Robustness Checks
- [ ] Handles missing values gracefully
- [ ] Works with different data dimensions
- [ ] Handles edge cases without crashing
- [ ] Clear error messages for invalid inputs

### Reproducibility Checks
- [ ] Results are reproducible with same random seed
- [ ] Different CV strategies give consistent rankings
- [ ] Documentation includes reproducible examples

---

## Decision Points & Questions

### Decisions Made

1. **LOTO Evaluation Scope** ✓
   - Each LOTO iteration will run GridSearchCV independently to find optimal hyperparameters for that specific train/test split
   - Question: How to handle interpolation vs extrapolation scenarios (e.g., test time between training times)?

2. **Regression Model Design** ✓
   - Use XGBoost Regressor only for GPU acceleration on large datasets
   - Question: Should we keep probability outputs for regression, or only continuous predictions?

3. **API Design** ✓
   - Use unified API: `run_sceptic_and_evaluate(..., cv_strategy="loto", model_type="regression")`
   - model_type options: "classification" (default, existing behavior) or "regression" (new)
   - cv_strategy options: "kfold" (default, existing behavior) or "loto" (new)
   - This keeps backwards compatibility: existing code continues to work without changes

4. **Merge Criteria** ✓
   - Acceptance criteria based on existing metrics in evaluation.py:
     - Pearson correlation (r and p-value)
     - Spearman correlation (r and p-value)
     - Kendall correlation (tau and p-value)
     - MSE (Mean Squared Error)
     - MAE (Mean Absolute Error)
   - New features must achieve comparable or better performance on benchmark datasets

### Open Questions

5. **Data Requirements**:
   - For LOTO: Minimum number of cells per time point?
   - For Regression: Should time labels be strictly numeric, or allow encoding?

6. **Public Release Timeline**:
   - When do you want to merge to main/public repo?

---

## File Structure Summary

### New Files to Create
```
Sceptic/
├── examples/
│   ├── dev_loto_evaluation.ipynb    # LOTO tutorial
│   └── dev_regression_model.ipynb   # Regression tutorial
├── test/dev/
│   ├── test_loto.py                 # LOTO unit tests
│   ├── test_regression.py           # Regression unit tests
│   ├── test_integration.py          # Integration tests
│   └── BENCHMARK_RESULTS.md         # Performance benchmarks
├── docs/
│   └── DEV_FEATURE_COMPARISON.md    # Technical comparison report
└── DEVELOPMENT_PLAN.md              # This document
```

### Modified Files
```
Sceptic/
├── src/sceptic/
│   └── sceptic.py                   # MODIFY: Add cv_strategy and model_type parameters
│                                    #         to run_sceptic_and_evaluate()
└── README.md                        # MODIFY: Add development notes

# No changes needed:
# - src/sceptic/__init__.py          # No changes (exports already correct)
# - src/sceptic/evaluation.py        # No changes (regression metrics already exist)
```

---

## Risk Assessment

### Technical Risks

1. **LOTO with Few Time Points**:
   - Risk: Insufficient training data if only 2-3 time points
   - Mitigation: Document minimum requirements, add warnings

2. **Regression Model Overfitting**:
   - Risk: Direct regression may overfit with high-dimensional data
   - Mitigation: Include regularization, cross-validation, feature selection

3. **API Complexity**:
   - Risk: Too many options confuse users
   - Mitigation: Clear documentation, sensible defaults, comprehensive examples

4. **Performance**:
   - Risk: LOTO with many time points could be slow
   - Mitigation: Optimize code, parallelize if needed, provide progress bars

### Process Risks

1. **Scope Creep**:
   - Risk: Adding too many features beyond original plan
   - Mitigation: Stick to plan, document future enhancements separately

2. **Testing Gaps**:
   - Risk: Missing edge cases in testing
   - Mitigation: Comprehensive test checklist, peer review

3. **Documentation Lag**:
   - Risk: Code completed but undocumented
   - Mitigation: Write docs alongside code, include in checklist

---

## Success Criteria

### Feature 1: LOTO Evaluation
- [ ] Successfully runs on scGEM dataset
- [ ] Produces valid confusion matrix and metrics
- [ ] Handles all time points correctly
- [ ] Performance within 5x of standard CV
- [ ] Documented with examples

### Feature 2: Regression Model
- [ ] XGBoost regressor implementation works
- [ ] GPU acceleration functional
- [ ] Produces continuous predictions
- [ ] Comparable or better correlation metrics than classification
- [ ] Handles continuous time labels
- [ ] Documented with examples

### Overall Project
- [ ] No breaking changes to existing code
- [ ] All tests pass
- [ ] Code review completed
- [ ] Documentation complete
- [ ] Benchmark results documented
- [ ] Ready for systematic testing

---

## Timeline

| Week | Dates | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1 | TBD | Environment setup, branch creation | Dev branch, testing framework |
| 2 | TBD | Implement LOTO evaluation | LOTO feature in sceptic.py, unit tests |
| 3 | TBD | Implement regression model | Regression feature in sceptic.py, unit tests |
| 4 | TBD | Integration testing | Integration tests, example notebooks |
| 5-6 | TBD | Systematic testing & benchmarking | Test results, benchmark report |
| 7 | TBD | Documentation & internal review | Full documentation, technical report |
| 8+ | TBD | Revisions & preparation for public release | Polished code, ready to merge |

**Total Estimated Time**: 8+ weeks for complete development and testing

---

## Next Steps

1. ✓ Review this plan and make decisions
2. ✓ XGBoost-only for regression (GPU support)
3. ✓ LOTO re-tunes hyperparameters per iteration
4. ✓ Unified API design
5. ✓ Use existing regression metrics for acceptance criteria
6. **Begin implementation**: Start with Phase 1 (environment setup)

---

## Notes

- This is a living document - update as development progresses
- Track progress in issues/tasks on development branch
- Schedule regular check-ins to review progress
- Consider code review before merging each major component

---

**Contact**: Gang Li
**Repository**: `/Users/gangli/Documents/GitHub/Sceptic`
**Development Branch**: `dev/loto-and-regression` (to be created)
