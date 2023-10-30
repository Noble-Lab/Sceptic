from sceptic import sceptic
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Load your data
data_concat = np.loadtxt("example_data/scGEM/expression.txt")
y =np.loadtxt("example_data/scGEM/expression_type.txt")

# Convert labels to categorical values
lab = preprocessing.LabelEncoder()
label = lab.fit_transform(y)

time_dictionary = {1.0:8, 2.0:16, 3.0:24,  4.0:30, 0.0:0}
y = pd.Series(np.unique(label)).map(time_dictionary).to_numpy()
label_list = np.transpose(np.unique(y))


# Define parameter search space
parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10]}

# Call the function to perform SVM and evaluation
cm, label_predicted, pseudotime, sceptic_prob = sceptic.run_sceptic_and_evaluate(data_concat, label, label_list, parameters)

# Save results
np.savetxt('test/scGEM/label-predicted-sceptic.txt', label_predicted, fmt='%i')
np.savetxt('test/scGEM/cm-sceptic.txt', cm, fmt='%i')
np.savetxt('test/scGEM/pseudotime-sceptic.txt', pseudotime, fmt='%1.4e')
np.savetxt('test/scGEM/sceptic_probability.txt', sceptic_prob, fmt='%1.5e')
