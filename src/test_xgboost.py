import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
import sys
import os
sceptic_root = os.path.expanduser('~/proj/2021-GL-vcs-integration/src/2025-04-21-new-implementation/Sceptic/src')
sys.path.insert(0, sceptic_root)
print("Path inserted:", sceptic_root)
print("Files in path:", os.listdir(sceptic_root))
from sceptic.sceptic import run_sceptic_and_evaluate
#from sceptic import run_sceptic_and_evaluate

# === Load example dataset ===
data_concat = np.loadtxt("example_data/scGEM/expression.txt")
y =np.loadtxt("example_data/scGEM/expression_type.txt")

# Convert labels to categorical values
lab = preprocessing.LabelEncoder()
label = lab.fit_transform(y)

time_dictionary = {1.0:8, 2.0:16, 3.0:24,  4.0:30, 0.0:0}
y = pd.Series(np.unique(label)).map(time_dictionary).to_numpy()
label_list = np.transpose(np.unique(y))


# === Choose classifier method: 'svm' or 'xgboost' ===
method = "xgboost"  # change to "svm" to use SVM
use_gpu = False     # set to True if running XGBoost on GPU

# === Define parameter grid for GridSearchCV ===
if method == "svm":
    parameters = {
        "C": [1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"]
    }
else:  # xgboost
    parameters = {
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.3],
        "n_estimators": [100],
        "subsample": [0.8]
    }

# # Call the function to perform SVM and evaluation
# cm, label_predicted, pseudotime, sceptic_prob = sceptic.run_sceptic_and_evaluate(data_concat, label, label_list, parameters)

# # For XGBoost with GPU
# cm, label_predicted, pseudotime, sceptic_prob = sceptic.run_sceptic_and_evaluate(data_concat, label, label_list, parameters, method="xgboost", use_gpu=True)

# === Run Sceptic ===
print(f"Running Sceptic with method: {method}")
cm, label_predicted, pseudotime, sceptic_prob = run_sceptic_and_evaluate(
    data=data_concat,
    labels=label,
    label_list=label_list,
    parameters=parameters,
    method=method,
    use_gpu=use_gpu
)

# === Save or print outputs ===
print("Confusion Matrix:\n", cm)
print("Predicted Labels:\n", label_predicted[:10])
print("Pseudotime:\n", pseudotime[:10])
print("Probabilities (first row):\n", sceptic_prob[0])

np.savetxt('test/scGEM/label-predicted-sceptic.txt', label_predicted, fmt='%i')
np.savetxt('test/scGEM/cm-sceptic.txt', cm, fmt='%i')
np.savetxt('test/scGEM/pseudotime-sceptic.txt', pseudotime, fmt='%1.4e')
np.savetxt('test/scGEM/sceptic_probability.txt', sceptic_prob, fmt='%1.5e')
