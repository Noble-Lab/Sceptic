'''
---------------------
sceptic functions
author: Gang Li
e-mail:gangliuw@uw.edu
MIT LICENSE
---------------------
'''
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np
import sklearn

def run_sceptic_and_evaluate(data, labels, label_list, parameters):
    cm = np.zeros((len(np.unique(labels)), len(np.unique(labels))))
    label_predicted = np.zeros(len(labels))
    sceptic_prob = np.zeros((len(labels), len(np.unique(labels))))
    pseudotime = np.zeros(len(labels))

    kf = KFold(n_splits=5, random_state=23, shuffle=True)

    for i, (train_valid_index, test_index) in enumerate(kf.split(data)):
        X_train_valid, X_test = data[train_valid_index, :], data[test_index, :]
        y_train_valid, y_test = labels[train_valid_index], labels[test_index]

        svc = svm.SVC(probability=True)
        clf = GridSearchCV(svc, parameters, cv=4)
        clf.fit(X_train_valid, y_train_valid)

        predicted = clf.predict(X_test)
        label_predicted[test_index] = predicted
        cm = cm + (sklearn.metrics.confusion_matrix(y_test, predicted))

        SVM_prob = clf.predict_proba(X_test)
        sceptic_prob[test_index, :] = SVM_prob
        pt = np.sum(np.multiply(SVM_prob, label_list),axis=1)
        pseudotime[test_index] = pt

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
