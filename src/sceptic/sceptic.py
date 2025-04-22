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

eFold=3
iFold=4
def run_sceptic_and_evaluate(data, labels, label_list, parameters, method="svm", use_gpu=False):
    """
    Run pseudotime estimation using SVM or XGBoost.
    
    Args:
        data (np.ndarray): Cell-by-feature matrix.
        labels (np.ndarray): Ground-truth time labels.
        label_list (List[float or int]): Ordered time labels (e.g., [0, 1, 2]).
        parameters (dict): Grid search parameters for the classifier.
        method (str): "svm" or "xgboost".
        use_gpu (bool): Only applies if method="xgboost".
        
    Returns:
        cm, label_predicted, pseudotime, sceptic_prob
    """
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
    label_predicted = np.zeros(len(labels))
    sceptic_prob = np.zeros((len(labels), len(label_list)))
    pseudotime = np.zeros(len(labels))
    kf = KFold(n_splits=eFold, random_state=23, shuffle=True)

    for i, (train_valid_index, test_index) in enumerate(kf.split(data)):
        X_train, X_test = data[train_valid_index], data[test_index]
        y_train, y_test = labels[train_valid_index], labels[test_index]
        #break
        if method == "xgboost":
            xgb_model = xgb.XGBClassifier(
                tree_method='gpu_hist' if use_gpu else 'auto',
                gpu_id=0 if use_gpu else -1,
                objective='multi:softprob',
                num_class=len(label_list),
                eval_metric='mlogloss'
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
        cm += sklearn.metrics.confusion_matrix(y_test, predicted)

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
