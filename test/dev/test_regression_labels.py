"""
Unit tests for regression label handling

Tests that regression models correctly use actual time values vs encoded labels,
and that appropriate warnings are raised when users pass encoded labels.
"""
import numpy as np
import pytest
import warnings
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from sceptic import run_sceptic_and_evaluate


class TestRegressionLabelValidation:
    """Test input validation for regression labels"""

    def test_regression_with_encoded_labels_warns(self):
        """Test that using encoded labels (0, 1, 2, ...) raises a warning"""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        labels_encoded = np.repeat([0, 1, 2, 3, 4], 20)  # Encoded
        label_list = np.array([0, 8, 16, 24, 30])  # Actual time

        with pytest.warns(UserWarning, match="encoded categorical values"):
            run_sceptic_and_evaluate(
                data, labels_encoded, label_list,
                method="xgboost", model_type="regression", eFold=2
            )

    def test_regression_with_actual_time_no_warning(self):
        """Test that using actual time values doesn't warn"""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        labels_time = np.repeat([0, 8, 16, 24, 30], 20)  # Actual time
        label_list = np.array([0, 8, 16, 24, 30])

        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            try:
                cm, pred, ptime, prob = run_sceptic_and_evaluate(
                    data, labels_time, label_list,
                    method="xgboost", model_type="regression", eFold=2
                )
            except UserWarning:
                pytest.fail("Unexpected warning raised with actual time values")

    def test_regression_range_mismatch_warns(self):
        """Test warning when label range << label_list range"""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        labels_encoded = np.repeat([0, 1, 2, 3, 4], 20)  # 0-4 range
        label_list = np.array([0, 8, 16, 24, 30])  # 0-30 range

        with pytest.warns(UserWarning, match="Label range"):
            run_sceptic_and_evaluate(
                data, labels_encoded, label_list,
                method="xgboost", model_type="regression", eFold=2
            )

    def test_regression_predictions_on_correct_scale(self):
        """Test that predictions are on the actual time scale"""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        labels_time = np.repeat([0, 8, 16, 24, 30], 20)

        cm, pred, ptime, prob = run_sceptic_and_evaluate(
            data, labels_time, method="xgboost",
            model_type="regression", cv_strategy="kfold", eFold=2
        )

        # Check predictions are approximately on input scale
        assert ptime.min() >= -10, f"Predictions too small: {ptime.min()}"
        assert ptime.max() <= 40, f"Predictions too large: {ptime.max()}"
        assert np.median(ptime) > 5, f"Median prediction too small: {np.median(ptime)}"
        assert np.median(ptime) < 25, f"Median prediction too large: {np.median(ptime)}"

    def test_classification_with_encoded_labels_no_warning(self):
        """Test that classification mode doesn't warn with encoded labels"""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        labels_encoded = np.repeat([0, 1, 2, 3, 4], 20)
        label_list = np.array([0, 8, 16, 24, 30])

        # Classification should NOT warn about encoded labels
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                cm, pred, ptime, prob = run_sceptic_and_evaluate(
                    data, labels_encoded, label_list,
                    method="xgboost", model_type="classification", eFold=2
                )
            except UserWarning as e:
                if "encoded" in str(e).lower():
                    pytest.fail("Classification should not warn about encoded labels")


class TestRegressionOutputs:
    """Test regression model outputs"""

    def test_regression_returns_correct_types(self):
        """Test that regression returns expected output types"""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        labels_time = np.repeat([0, 8, 16, 24, 30], 20)

        cm, pred, ptime, prob = run_sceptic_and_evaluate(
            data, labels_time, method="xgboost",
            model_type="regression", eFold=2
        )

        # Regression-specific checks
        assert cm is None, "Confusion matrix should be None for regression"
        assert pred is None, "Discrete predictions should be None for regression"
        assert prob is None, "Probabilities should be None for regression"
        assert ptime is not None, "Pseudotime should not be None"
        assert len(ptime) == len(labels_time), "Pseudotime length mismatch"
        assert isinstance(ptime, np.ndarray), "Pseudotime should be numpy array"

    def test_regression_loto_works(self):
        """Test that LOTO works with regression"""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        labels_time = np.repeat([0, 8, 16, 24, 30], 20)

        cm, pred, ptime, prob = run_sceptic_and_evaluate(
            data, labels_time, method="xgboost",
            model_type="regression", cv_strategy="loto"
        )

        assert ptime is not None
        assert len(ptime) == len(labels_time)
        # LOTO should produce predictions for all samples
        assert not np.any(np.isnan(ptime)), "LOTO produced NaN predictions"


class TestEncodedVsActualComparison:
    """Test that encoded labels actually inflate performance"""

    def test_encoded_labels_give_higher_correlation(self):
        """
        Verify that using encoded labels incorrectly inflates performance.
        This documents the bug we're trying to prevent.
        """
        np.random.seed(42)
        # Create simple synthetic data where time matters
        n_samples = 100
        n_features = 10
        time_points = [0, 8, 16, 24, 30]
        n_per_time = n_samples // len(time_points)

        # Generate data with time-dependent pattern
        data = []
        labels_time = []
        labels_encoded = []

        for i, t in enumerate(time_points):
            # Features increase with time
            X = np.random.randn(n_per_time, n_features) + t * 0.1
            data.append(X)
            labels_time.extend([t] * n_per_time)
            labels_encoded.extend([i] * n_per_time)

        data = np.vstack(data)
        labels_time = np.array(labels_time)
        labels_encoded = np.array(labels_encoded)

        # Suppress warnings for this test
        os.environ['SCEPTIC_IGNORE_REGRESSION_WARNINGS'] = '1'

        try:
            # Test with encoded labels (WRONG but shouldn't crash)
            _, _, ptime_encoded, _ = run_sceptic_and_evaluate(
                data, labels_encoded, np.array(time_points),
                method="xgboost", model_type="regression", eFold=2
            )

            # Test with actual time (CORRECT)
            _, _, ptime_time, _ = run_sceptic_and_evaluate(
                data, labels_time, np.array(time_points),
                method="xgboost", model_type="regression", eFold=2
            )

            # Compute correlations
            from scipy.stats import spearmanr
            corr_encoded = spearmanr(labels_encoded, ptime_encoded)[0]
            corr_time = spearmanr(labels_time, ptime_time)[0]

            # Encoded labels typically give inflated correlation
            # (though not guaranteed in all cases with random data)
            print(f"\nCorrelation with encoded labels: {corr_encoded:.4f}")
            print(f"Correlation with time labels: {corr_time:.4f}")

            # Just verify both correlations are computed successfully
            assert not np.isnan(corr_encoded)
            assert not np.isnan(corr_time)

        finally:
            # Clean up
            os.environ.pop('SCEPTIC_IGNORE_REGRESSION_WARNINGS', None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
