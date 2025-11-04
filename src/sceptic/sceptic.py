'''
---------------------
sceptic functions
author: Gang Li
e-mail:gangliuw@uw.edu
MIT LICENSE
---------------------
'''
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import svm
import numpy as np
import sklearn
import xgboost as xgb
import warnings
from packaging import version

eFold=3
iFold=4

def _create_xgb_classifier(num_classes, use_gpu=False):
    """
    Create XGBClassifier with version-appropriate parameters.

    Args:
        num_classes (int): Number of classes for classification.
        use_gpu (bool): Whether to use GPU acceleration.

    Returns:
        xgb.XGBClassifier: Configured XGBoost classifier.
    """
    # Detect XGBoost version
    xgb_version = version.parse(xgb.__version__)

    # Base parameters that work across versions
    base_params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'eval_metric': 'mlogloss'
    }

    # Version-specific GPU parameters
    if xgb_version >= version.parse("3.1.0"):
        # XGBoost 3.1+ uses 'device' parameter
        if use_gpu:
            try:
                base_params['device'] = 'cuda:0'
            except Exception as e:
                warnings.warn(
                    f"GPU requested but failed to configure: {e}. Falling back to CPU.",
                    UserWarning
                )
                base_params['device'] = 'cpu'
        else:
            base_params['device'] = 'cpu'
    elif xgb_version >= version.parse("2.0.0"):
        # XGBoost 2.x uses 'device' parameter (introduced in 2.0)
        base_params['device'] = 'cuda:0' if use_gpu else 'cpu'
    else:
        # XGBoost 1.x uses 'gpu_id' and 'tree_method'
        base_params['tree_method'] = 'gpu_hist' if use_gpu else 'auto'
        base_params['gpu_id'] = 0 if use_gpu else -1

    try:
        return xgb.XGBClassifier(**base_params)
    except Exception as e:
        # If there's still an error, fall back to minimal configuration
        warnings.warn(
            f"Failed to create XGBClassifier with optimal parameters: {e}. "
            f"Using minimal configuration.",
            UserWarning
        )
        return xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes
        )

def _create_xgb_regressor(use_gpu=False):
    """
    Create XGBRegressor with version-appropriate parameters.

    Args:
        use_gpu (bool): Whether to use GPU acceleration.

    Returns:
        xgb.XGBRegressor: Configured XGBoost regressor.
    """
    # Detect XGBoost version
    xgb_version = version.parse(xgb.__version__)

    # Base parameters that work across versions
    base_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # Version-specific GPU parameters
    if xgb_version >= version.parse("3.1.0"):
        # XGBoost 3.1+ uses 'device' parameter
        if use_gpu:
            try:
                base_params['device'] = 'cuda:0'
            except Exception as e:
                warnings.warn(
                    f"GPU requested but failed to configure: {e}. Falling back to CPU.",
                    UserWarning
                )
                base_params['device'] = 'cpu'
        else:
            base_params['device'] = 'cpu'
    elif xgb_version >= version.parse("2.0.0"):
        # XGBoost 2.x uses 'device' parameter (introduced in 2.0)
        base_params['device'] = 'cuda:0' if use_gpu else 'cpu'
    else:
        # XGBoost 1.x uses 'gpu_id' and 'tree_method'
        base_params['tree_method'] = 'gpu_hist' if use_gpu else 'auto'
        base_params['gpu_id'] = 0 if use_gpu else -1

    try:
        return xgb.XGBRegressor(**base_params)
    except Exception as e:
        # If there's still an error, fall back to minimal configuration
        warnings.warn(
            f"Failed to create XGBRegressor with optimal parameters: {e}. "
            f"Using minimal configuration.",
            UserWarning
        )
        return xgb.XGBRegressor(objective='reg:squarederror')

def run_sceptic_and_evaluate(data, labels, label_list=None, parameters=None, method="svm", use_gpu=False,
                             cv_strategy="kfold", model_type="classification", eFold=3, iFold=4):
    """
    Run pseudotime estimation using SVM or XGBoost with flexible CV and model types.

    Args:
        data (np.ndarray): Cell-by-feature matrix (cells × features).
        labels (np.ndarray): Ground-truth time labels for each cell.
            Can be either:
            - Actual time values (e.g., [0, 8, 16, 24, 30, ...])
            - Pre-encoded categorical labels (e.g., [0, 1, 2, 3, 4, ...])
        label_list (np.ndarray, optional): Ordered unique time points for pseudotime calculation.
            If None, will be automatically inferred from unique values in labels.
            Use this to specify actual biological time points when labels are encoded.
            Example: labels=[0,0,1,1,2,2], label_list=[0, 8, 16]
        parameters (dict, optional): Grid search parameters for the classifier/regressor.
            If None, uses default parameters.
        method (str): "svm" or "xgboost". Note: regression mode only supports "xgboost".
        use_gpu (bool): Only applies if method="xgboost".
        cv_strategy (str): Cross-validation strategy. Options:
            - "kfold" (default): Standard k-fold cross-validation
            - "loto": Leave-one-time-out cross-validation (holds out each time point)
        model_type (str): Model type. Options:
            - "classification" (default): Multi-class classification with probability weighting
            - "regression": Direct regression (XGBoost only, outputs continuous values)
        eFold (int): Number of external CV folds (used when cv_strategy="kfold").
        iFold (int): Number of internal GridSearchCV folds.

    Returns:
        tuple: (cm, label_predicted, pseudotime, sceptic_prob)
            For classification mode:
                - cm: Confusion matrix (n_timepoints × n_timepoints)
                - label_predicted: Predicted encoded labels for each cell
                - pseudotime: Continuous pseudotime values for each cell
                - sceptic_prob: Class probabilities for each cell (cells × n_timepoints)
            For regression mode:
                - cm: None
                - label_predicted: None
                - pseudotime: Continuous pseudotime predictions for each cell
                - sceptic_prob: None

    Examples:
        >>> # Example 1: Standard classification with k-fold CV (existing behavior)
        >>> cm, pred, ptime, prob = run_sceptic_and_evaluate(
        ...     data, labels, method="xgboost"
        ... )

        >>> # Example 2: LOTO evaluation with classification
        >>> cm, pred, ptime, prob = run_sceptic_and_evaluate(
        ...     data, labels, method="xgboost", cv_strategy="loto"
        ... )

        >>> # Example 3: Direct regression with k-fold CV
        >>> cm, pred, ptime, prob = run_sceptic_and_evaluate(
        ...     data, labels, method="xgboost", model_type="regression"
        ... )

        >>> # Example 4: LOTO evaluation with regression
        >>> cm, pred, ptime, prob = run_sceptic_and_evaluate(
        ...     data, labels, method="xgboost", cv_strategy="loto", model_type="regression"
        ... )
    """
    from sklearn import preprocessing

    # Validate inputs
    if model_type not in ["classification", "regression"]:
        raise ValueError(f"model_type must be 'classification' or 'regression', got '{model_type}'")
    if cv_strategy not in ["kfold", "loto"]:
        raise ValueError(f"cv_strategy must be 'kfold' or 'loto', got '{cv_strategy}'")
    if model_type == "regression" and method != "xgboost":
        raise ValueError("Regression mode only supports method='xgboost'")

    # Handle labels and label_list based on model type
    unique_labels = np.unique(labels)

    if model_type == "classification":
        # Classification: encode labels to 0, 1, 2, ...
        if label_list is None:
            label_list = unique_labels
            if np.array_equal(unique_labels, np.arange(len(unique_labels))):
                encoded_labels = labels.astype(int)
            else:
                lab = preprocessing.LabelEncoder()
                encoded_labels = lab.fit_transform(labels)
        else:
            if len(unique_labels) != len(label_list):
                raise ValueError(
                    f"Number of unique labels ({len(unique_labels)}) does not match "
                    f"length of label_list ({len(label_list)})"
                )
            lab = preprocessing.LabelEncoder()
            encoded_labels = lab.fit_transform(labels)
    else:
        # Regression: use labels as continuous values
        if label_list is None:
            label_list = unique_labels
        encoded_labels = None  # Not needed for regression

    # Set default parameters if none provided
    if not parameters:
        if model_type == "classification":
            if method == "svm":
                parameters = {
                    "C": [1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale"]
                }
            elif method == "xgboost":
                parameters = {
                    "max_depth": [3, 5],
                    "learning_rate": [0.1, 0.3],
                    "n_estimators": [100],
                    "subsample": [0.8]
                }
        else:  # regression
            parameters = {
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.3],
                "n_estimators": [100]
            }

    # Initialize output arrays
    if model_type == "classification":
        cm = np.zeros((len(label_list), len(label_list)))
        label_predicted = np.zeros(len(labels))
        sceptic_prob = np.zeros((len(labels), len(label_list)))
    else:
        cm = None
        label_predicted = None
        sceptic_prob = None
    pseudotime = np.zeros(len(labels))

    # Determine CV strategy
    if cv_strategy == "kfold":
        # Standard k-fold cross-validation
        kf = KFold(n_splits=eFold, random_state=23, shuffle=True)
        cv_splits = list(kf.split(data))
    else:  # loto
        # Leave-one-time-out: hold out each unique value in labels
        cv_splits = []
        for label_value in unique_labels:
            test_mask = (labels == label_value)
            train_index = np.where(~test_mask)[0]
            test_index = np.where(test_mask)[0]
            if len(test_index) > 0 and len(train_index) > 0:
                cv_splits.append((train_index, test_index))

    # Cross-validation loop
    for i, (train_index, test_index) in enumerate(cv_splits):
        X_train, X_test = data[train_index], data[test_index]

        if model_type == "classification":
            y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

            # For LOTO, need to handle label_list for current fold
            if cv_strategy == "loto":
                # Find which time point is held out
                held_out_label = labels[test_index[0]]
                # Find the index of the held-out label in the original encoding
                held_out_encoded = encoded_labels[test_index[0]]
                # Create label_list_train excluding held-out time
                label_list_train_mask = np.ones(len(label_list), dtype=bool)
                label_list_train_mask[held_out_encoded] = False
                label_list_train = label_list[label_list_train_mask]
                num_classes = len(label_list_train)

                # Re-encode training labels to be 0, 1, 2, ..., num_classes-1
                # This is necessary because encoded_labels might not be consecutive after holdout
                unique_train_labels = np.unique(y_train)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_train_labels)}
                y_train = np.array([label_mapping[label] for label in y_train])
            else:
                num_classes = len(label_list)

            # Initialize classifier
            if method == "xgboost":
                xgb_model = _create_xgb_classifier(
                    num_classes=num_classes,
                    use_gpu=use_gpu
                )
                clf = GridSearchCV(xgb_model, parameters, cv=iFold)
            elif method == "svm":
                svc = svm.SVC(probability=True)
                clf = GridSearchCV(svc, parameters, cv=iFold)
            else:
                raise ValueError(f"Unsupported method '{method}'. Choose 'svm' or 'xgboost'.")

            # Train and predict
            clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)

            # For LOTO, map predictions back to original label space
            if cv_strategy == "loto":
                # Create reverse mapping from re-encoded to original labels
                reverse_mapping = {new_label: old_label for old_label, new_label in label_mapping.items()}
                predicted_original = np.array([reverse_mapping[pred] for pred in predicted])
                label_predicted[test_index] = predicted_original
                cm += sklearn.metrics.confusion_matrix(y_test, predicted_original, labels=np.arange(len(label_list)))
            else:
                label_predicted[test_index] = predicted
                cm += sklearn.metrics.confusion_matrix(y_test, predicted, labels=np.arange(len(label_list)))

            # Get probabilities and compute pseudotime
            try:
                prob = clf.predict_proba(X_test)
                # For LOTO, need to align probabilities with full label_list
                if cv_strategy == "loto":
                    prob_full = np.zeros((len(X_test), len(label_list)))
                    prob_full[:, label_list_train_mask] = prob
                    prob = prob_full
                sceptic_prob[test_index, :] = prob
                pseudotime[test_index] = np.sum(prob * label_list, axis=1)
            except Exception as e:
                print(f"Warning: predict_proba failed on fold {i}: {e}")
                prob = np.zeros((len(X_test), len(label_list)))
                sceptic_prob[test_index, :] = prob
                pseudotime[test_index] = np.sum(prob * label_list, axis=1)

        else:  # regression
            y_train, y_test = labels[train_index], labels[test_index]

            # Initialize regressor
            xgb_model = _create_xgb_regressor(use_gpu=use_gpu)
            clf = GridSearchCV(xgb_model, parameters, cv=iFold)

            # Train and predict
            clf.fit(X_train, y_train)
            predicted_continuous = clf.predict(X_test)
            pseudotime[test_index] = predicted_continuous

    return cm, label_predicted, pseudotime, sceptic_prob


# # Load your data
# data_concat = np.loadtxt('results/CDP_ds200.txt')
# y = np.loadtxt('results/y_ds200.txt')

# # Convert labels to categorical values
# lab = preprocessing.LabelEncoder()
# label = lab.fit_transform(y)

# label_list = np.transpose(np.unique(y))

# # Define parameter search space
# parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10]}

# # Call the function to perform SVM and evaluation
# cm, label_predicted, pseudotime, sceptic_prob = run_sceptic_and_evaluate(data_concat, label, label_list, parameters)

# # Save results
# np.savetxt('label-predicted-sceptic.txt', label_predicted, fmt='%i')
# np.savetxt('cm-sceptic.txt', cm, fmt='%i')
# np.savetxt('pseudotime-sceptic.txt', pseudotime, fmt='%1.4e')
# np.savetxt('sceptic_probability.txt', sceptic_prob, fmt='%1.5e')
