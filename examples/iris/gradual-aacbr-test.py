import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add parent directory to sys.path
sys.path.append(parent_dir)

print(parent_dir)


import torch
import torch.optim as optim
import numpy as np

import deeparguing.gradual_aacbr as gradual_aacbr
import deeparguing.semantics.relu_semantics as rs
import deeparguing.semantics.sigmoid_semantics as ss
import deeparguing.base_scores.learned_base_score as lbs
import deeparguing.base_scores.constant_base_score as cbs
import deeparguing.casebase_edge_weights.learned_partial_order as lpo
import deeparguing.irrelevance_edge_weights.regular_irrelevance as ri
import deeparguing.feature_extractor.feature_weighted_extractor as fwe
import deeparguing.feature_extractor.mlp_extractor as mlpe
import deeparguing.casebase_edge_weights.compute_partial_order as cpo
import deeparguing.feature_extractor.scaler as scaler

from deeparguing.train import evaluate_model, static_train_model
from deeparguing.regulariser import sparsity_regulariser, community_preservation_regulariser, connectivity_regulariser, feature_smoothness_regulariser, regularise


from helper import load_iris, split_data, normalise_input

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SEED = 42


X, y = load_iris(labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

all_y = np.unique(y, axis=0)
print(all_y)


train_full, train, val, test = split_data(X, y, SEED)

print(f"Test Size:  {len(test['X'])}")
print(f"Train Size:  {len(train['X'])}")
print(f"Validation Size:  {len(val['X'])}")




# ### Convert to Torch


X_train_full, y_train_full = torch.tensor(train_full["X"], device=device),      torch.tensor(train_full["y"], dtype=torch.float32, device=device)
X_train, y_train           = torch.tensor(train["X"]     , device=device),      torch.tensor(train["y"],      dtype=torch.float32, device=device)
X_val, y_val               = torch.tensor(val["X"]       , device=device),      torch.tensor(val["y"],        dtype=torch.float32, device=device)
X_test, y_test             = torch.tensor(test["X"]      , device=device),      torch.tensor(test["y"],       dtype=torch.float32, device=device)


# ### Normalize dataset


train_mean = X_train.mean(dim=0)
train_std = X_train.std(dim=0)


X_train = normalise_input(X_train, train_mean, train_std)
X_val = normalise_input(X_val, train_mean, train_std)
X_test = normalise_input(X_test, train_mean, train_std)



# ### Train Model


DEFAULT_CASE = X_train.mean(axis=0)

X_DEFAULTS = DEFAULT_CASE.tile(len(all_y), 1)
Y_DEFAULTS = torch.tensor(all_y, device=device).flip([0])


MAX_ITERS = len(X_train)
EPOCHS = 3000
USE_SYMMETRIC_ATTACKS = False
LR = 3e-2
TEMPERATURE = 0.1
USE_BLOCKERS = True

totalf1 = 0
N = 100

for torch_seed in range(0, N):

    # torch_seed = 1

    print("="*40)
    print("Seed:", torch_seed)

    torch.manual_seed(torch_seed) # TRY DIFFERENT INITIAL WEIGHTS 

    no_features = X_train.shape[-1]
    semantics = rs.ReluSemantics(max_iters=MAX_ITERS, epsilon=0)

    pofe = fwe.FeatureWeightedExtractor(no_features)
    bsfe = pofe
    bs_scaler = scaler.Scaler(bsfe.get_output_features(), weight=1.0)
    comp_func = cpo.Subtractor(temperature=TEMPERATURE, activation=torch.sigmoid)

    partial_order = lpo.LearnedPartialOrder([pofe], comparison_func=comp_func)
    irrelevance = ri.RegularIrrelevance(partial_order)
    base_score = lbs.LearnedBaseScore([bsfe, bs_scaler], activation=torch.sigmoid)

    alpha = 5e-4

    regulariser = lambda model: regularise(model, [
        # [sparsity_regulariser, alpha], 
        # [connectivity_regulariser, alpha], 
        # [community_preservation_regulariser, alpha],
        # [feature_smoothness_regulariser, alpha]
        ])



    model = gradual_aacbr.GradualAACBR(semantics, 
                                    base_score,
                                    irrelevance,
                                    partial_order).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)


    POST_PROCESS_FUNC = lambda x: x


    # with torch.no_grad():
    #     accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_DEFAULTS, Y_DEFAULTS, 
    #                                                      X_val, y_val, show_confusion=True, use_blockers=USE_BLOCKERS,  
    #                                                      print_matrix=True, print_compute_graph=False, 
    #                                                      print_graph=False, print_results=False, post_process_func=POST_PROCESS_FUNC)


    losses = static_train_model(model, X_train, y_train, 
                    X_DEFAULTS, Y_DEFAULTS, optimizer, 
                    criterion, EPOCHS, X_new_cases=X_train, y_new_cases=y_train, 
                    use_symmetric_attacks=False, use_blockers=USE_BLOCKERS, 
                    plot_loss_curve=False,
                    disable_tqdm=False, post_process_func=POST_PROCESS_FUNC, regularise_graph=regulariser)

    losses = np.array(losses)


    with torch.no_grad():
        accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_DEFAULTS, Y_DEFAULTS, 
                                                        X_val, y_val, show_confusion=False, use_blockers=USE_BLOCKERS,  
                                                        print_matrix=False, print_compute_graph=False, 
                                                        print_graph=False, print_results=False, post_process_func=POST_PROCESS_FUNC)

        print("VAL DATA RESULTS:")
        print("Accuracy, Precision, Recall, F1")
        print(accuracy, precision, recall, f1)
        print("="*40)
        if f1 > 0.7:
            totalf1 += 1
        print(f'Total with F1 > 0.7: {totalf1}/{torch_seed + 1}: {(totalf1/(torch_seed + 1)) * 100}%')

print("="*40)
print(f'Total with F1 > 0.7: {totalf1}/{N}: {(totalf1/N) * 100}%')
print("="*40)

# BASELINE Normal Initialisation in 10 runs:
#  4/10
# 27/100
# 136/500

# Changing to He Uniform initalisation:
#   5/10
#  24/100

# Changing to He Normal initalisation:
#   1/10

# Changing to LeCun initalisation:
#   4/10

# Changing to Xavier Uniform initalisation:
#   7/10
#  40/100

# Changing to Xavier Normal initalisation:
#  2/10


