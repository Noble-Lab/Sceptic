# Sceptic Examples

This directory contains example notebooks demonstrating how to use Sceptic for pseudotime analysis on different types of single-cell data.

## Available Examples

### 1. Basic Usage - Simplified (`basic_usage_simplified.ipynb`) ⭐ **START HERE**
The easiest way to get started with Sceptic:
- **No manual label encoding required!**
- Pass actual biological time values directly (hours, days, etc.)
- Automatic handling of time label mapping
- Clear comparison of old vs new workflow

**Recommended for:** Everyone, especially new users!

### 2. Basic Usage (`basic_usage.ipynb`)
Complete introduction to Sceptic covering:
- Loading example data
- Manual time label mapping (for understanding internals)
- Running Sceptic with SVM and XGBoost
- Interpreting results (confusion matrix, pseudotime, probabilities)
- Basic visualization

**Recommended for:** Users who want to understand the full workflow details.

### 3. Custom Evaluation (`custom_evaluation.ipynb`)
Advanced evaluation and visualization using the new utility modules:
- Using `sceptic.evaluation` for comprehensive metrics (accuracy, correlations, MAE/MSE)
- Using `sceptic.plotting` for publication-quality figures
- Stratified analysis by groups
- Best practices for reporting results

**Recommended for:** Users preparing results for publication or who need detailed performance metrics.

### 4. Two-Timepoint Classification (`two_timepoint_classification.py`)
Short script showing how to run Sceptic on a two-timepoint subset of the bundled
scGEM dataset:
- demonstrates both `svm` and `xgboost`
- uses the existing example data rather than a synthetic toy dataset
- documents that `kfold` is the supported CV mode for two time points
- reminds users that this setting was not benchmarked in the original study

**Recommended for:** Users with exactly two time points who want a minimal
working example.

## Getting Started

### Installation

Ensure you have Sceptic installed with all dependencies:

```bash
pip install sceptic
```

For the examples, you'll also need:

```bash
pip install jupyter matplotlib seaborn scipy
```

### Running the Examples

1. Clone or download this repository
2. Navigate to the examples directory:
   ```bash
   cd examples
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open any of the example notebooks

To run the two-timepoint script directly from the repository root:

```bash
python examples/two_timepoint_classification.py
```

## Data Requirements

The examples use the included `example_data/scGEM/` dataset. For your own data, you'll need:

- **Expression matrix**: Cells × features (genes, peaks, etc.)
- **Time labels**: Categorical or continuous time information for each cell

Supported formats:
- NumPy arrays (`.npy`, `.txt`)
- Pandas DataFrames
- AnnData objects (`.h5ad`) - recommended for single-cell data

## Example Datasets

The repository includes:

- **scGEM**: Single-cell gene expression data from developmental time course
  - 538 cells × features
  - 5 developmental time points

For more diverse examples, see the publications using Sceptic or contact the authors.

## Troubleshooting

**Issue**: Out of memory errors
- **Solution**: Reduce the number of features using dimensionality reduction (PCA) or highly variable gene selection

**Issue**: Poor accuracy
- **Solution**: Try adjusting the parameter grid, using more features, or checking data quality

**Issue**: Cross-validation warnings
- **Solution**: Ensure you have sufficient samples per time point (recommended: >20 cells per time point)

## Additional Resources

- [Sceptic documentation](https://github.com/Noble-Lab/Sceptic)
- [Publication](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03679-3) - Genome Biology 2025
- [Issue tracker](https://github.com/Noble-Lab/Sceptic/issues)

## Contributing

Found a bug or have an example to contribute? Please open an issue or pull request on GitHub!
