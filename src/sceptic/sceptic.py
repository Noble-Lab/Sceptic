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
    if num_classes == 2:
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
    else:
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
        if num_classes == 2:
            return xgb.XGBClassifier(objective='binary:logistic')
        return xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes
        )

def run_sceptic_and_evaluate(data, labels, label_list=None, parameters=None, method="svm", use_gpu=False):
    """
    Run pseudotime estimation using SVM or XGBoost.

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
        parameters (dict, optional): Grid search parameters for the classifier.
            If None, uses default parameters.
        method (str): "svm" or "xgboost".
        use_gpu (bool): Only applies if method="xgboost".

    Returns:
        tuple: (cm, label_predicted, pseudotime, sceptic_prob)
            - cm: Confusion matrix (n_timepoints × n_timepoints)
            - label_predicted: Predicted encoded labels for each cell
            - pseudotime: Continuous pseudotime values for each cell
            - sceptic_prob: Class probabilities for each cell (cells × n_timepoints)

    Examples:
        >>> # Example 1: Using actual time values directly
        >>> labels = np.array([0, 0, 8, 8, 16, 16, 24, 24])
        >>> cm, pred, ptime, prob = run_sceptic_and_evaluate(
        ...     data, labels, method="xgboost"
        ... )

        >>> # Example 2: Using encoded labels with actual time points
        >>> labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # encoded
        >>> label_list = np.array([0, 8, 16, 24])  # actual hours
        >>> cm, pred, ptime, prob = run_sceptic_and_evaluate(
        ...     data, labels, label_list=label_list, method="xgboost"
        ... )
    """
    from sklearn import preprocessing

    # Handle labels and label_list
    # Check if labels need encoding (contain non-consecutive integers or floats)
    unique_labels = np.unique(labels)

    if len(unique_labels) == 2:
        warnings.warn(
            "Two-timepoint classification is supported, but this setting was "
            "not benchmarked in the original Sceptic study. Interpret results "
            "with additional caution.",
            UserWarning,
            stacklevel=2
        )

    # If label_list is not provided, infer it from labels
    if label_list is None:
        label_list = unique_labels
        # Check if labels are already encoded (consecutive integers starting from 0)
        if np.array_equal(unique_labels, np.arange(len(unique_labels))):
            # Already encoded
            encoded_labels = labels.astype(int)
        else:
            # Need to encode
            lab = preprocessing.LabelEncoder()
            encoded_labels = lab.fit_transform(labels)
    else:
        # label_list is provided - assume labels need encoding to match label_list
        if len(unique_labels) != len(label_list):
            raise ValueError(
                f"Number of unique labels ({len(unique_labels)}) does not match "
                f"length of label_list ({len(label_list)})"
            )
        # Encode labels to 0, 1, 2, ...
        lab = preprocessing.LabelEncoder()
        encoded_labels = lab.fit_transform(labels)
    # Set default parameters if none provided
    if not parameters:
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
        else:
            raise ValueError(f"Unsupported method '{method}' and no parameters provided.")
        
    cm = np.zeros((len(label_list), len(label_list)))
    label_predicted = np.zeros(len(encoded_labels))
    sceptic_prob = np.zeros((len(encoded_labels), len(label_list)))
    pseudotime = np.zeros(len(encoded_labels))
    kf = KFold(n_splits=eFold, random_state=23, shuffle=True)

    for i, (train_valid_index, test_index) in enumerate(kf.split(data)):
        X_train, X_test = data[train_valid_index], data[test_index]
        y_train, y_test = encoded_labels[train_valid_index], encoded_labels[test_index]
        #break
        if method == "xgboost":
            xgb_model = _create_xgb_classifier(
                num_classes=len(label_list),
                use_gpu=use_gpu
            )
            clf = GridSearchCV(xgb_model, parameters, cv=iFold)
        elif method == "svm":
            svc = svm.SVC(probability=True)
            clf = GridSearchCV(svc, parameters, cv=iFold)
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'svm' or 'xgboost'.")

        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        label_predicted[test_index] = predicted
        cm += sklearn.metrics.confusion_matrix(
            y_test,
            predicted,
            labels=np.arange(len(label_list))
        )

        try:
            prob = clf.predict_proba(X_test)
        except Exception as e:
            print(f"Warning: predict_proba failed on fold {i}: {e}")
            prob = np.zeros((len(X_test), len(label_list)))

        sceptic_prob[test_index, :] = prob
        pseudotime[test_index] = np.sum(prob * label_list, axis=1)

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
