import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add parent directory to sys.path
sys.path.append(parent_dir)

print(parent_dir)


import numpy as np

import deeparguing.aacbr as aacbr

from helper import load_iris, split_data, normalise_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm




# device=None



SEED = 42


# ## Data Set


# X, y = load_iris(labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
# X, y = load_iris(labels=["Iris-setosa", "Iris-versicolor"])
# X, y = load_iris(labels=["Iris-setosa", "Iris-virginica"])
X, y = load_iris(labels=["Iris-versicolor", "Iris-virginica"])

y = np.argmax(y, axis=1)

all_y = np.unique(y)

print(all_y)
# ## Train Model


# ### Split into Training, Validation and Test


train_full, train, val, test = split_data(X, y, SEED)

print(f"Test Size:  {len(test['X'])}")
print(f"Train Size:  {len(train['X'])}")
print(f"Validation Size:  {len(val['X'])}")




# ### Convert to Torch


X_train_full, y_train_full = train_full["X"], train_full["y"] 
X_train, y_train           = train["X"]     , train["y"]      
X_val, y_val               = val["X"]       , val["y"]        
X_test, y_test             = test["X"]      , test["y"]       



# ### Normalize dataset


train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0, ddof=1)


# print(train_mean)
# print(train_std)

X_train = normalise_input(X_train, train_mean, train_std)
X_val = normalise_input(X_val, train_mean, train_std)
X_test = normalise_input(X_test, train_mean, train_std)





def eval_model(model, X, y):
    # Assess the model pre-training
    y_preds = model(X).squeeze()    
    y_actual = y

    results = ( 
        accuracy_score(y_actual, y_preds),
        precision_score(y_actual, y_preds, average='macro', zero_division=0),
        recall_score(y_actual, y_preds, average='macro', zero_division=0),
        f1_score(y_actual, y_preds, average='macro', zero_division=0),
        confusion_matrix(y_actual, y_preds)
    )


    return results

# ### Train Model


DEFAULT_CASE = np.zeros_like(X_train[0])
Y_DEFAULT = all_y[1]

print("DEFAULT", Y_DEFAULT)

no_features = X_train.shape[-1]


def sigmoid(z):
    return 1/(1 + np.exp(-z))

TEMP = 0.1

def comparison_func(weights, attacker, target):
    if attacker.ndim == 1:
        attacker = np.expand_dims(attacker, axis=0)
    if target.ndim == 1:
        target = np.expand_dims(target, axis=0)

    # result = np.matmul(attacker, weights) > np.matmul(target, weights)
    # result = np.matmul(attacker, weights) > np.matmul(target, weights)
    result = sigmoid((np.matmul(attacker, weights) - np.matmul(target, weights))/TEMP) > 0.5

    return result

# all_weights = []
# for w1 in tqdm(range(-5, 6, 2)):
#     for w2 in range(-5, 6, 2):
#         for w3 in range(-5, 6, 2):
#             for w4 in range(-5, 6, 2):
#                 all_weights.append(np.array([w1, w2, w3, w4]))



# bestf1 = 0
# for weights in tqdm(all_weights):

#     model = aacbr.AACBR(X_train, y_train, lambda attacker, target: comparison_func(weights, attacker, target), 
#                         DEFAULT_CASE, [Y_DEFAULT], use_symmetric_attacks=False, build_parallel=True)

#     # print("TRAINING SET")
#     result = eval_model(model, X_train, y_train)
#     if result[3] > bestf1:
#         bestf1 = result[3]
#         bestw = weights

# print("BEST ON TRAINING DATA")
# print(bestf1)
# print(bestw)





    


# bestw = np.array([-5.0, -1, -5, -5])
bestw = np.array([-1.0, -1, -5, -5])


model = aacbr.AACBR(X_train, y_train, lambda attacker, target: comparison_func(bestw, attacker, target), 
                    DEFAULT_CASE, [Y_DEFAULT], use_symmetric_attacks=False, build_parallel=False)

model.show_matrix()
model.show_graph_with_labels()

results = eval_model(model, X_val, y_val)

print("RESULTS ON VALIDATION DATA")
print("Accuracy, precision, recall, f1")
print(results[:-1])
print("confusion matrix:")
print(results[-1])




