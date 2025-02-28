
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
from deeparguing.regulariser import sparsity_regulariser, community_preservation_regulariser, connectivity_regulariser, feature_smoothness_regulariser, regularise, community_prev_reg_attacks, community_prev_reg_supports


from helper import load_mnist, split_data, normalise_input, select_samples_per_class


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


SEED = 42



# ### DATA SET


X, y = load_mnist(device=device, as_vector=True, size=500)
X = X.detach().cpu().numpy()
y = y.detach().cpu().numpy()

all_y = np.unique(y, axis=0)
print(all_y)







train_full, train, val, test = split_data(X, y, SEED)

print(f"Test Size:  {len(test['X'])}")
print(f"Train Size:  {len(train['X'])}")
print(f"Validation Size:  {len(val['X'])}")




train["X"], train["y"] = select_samples_per_class(X, y, num_samples=10, random_state=42)



X_train_full, y_train_full = torch.tensor(train_full["X"], device=device),      torch.tensor(train_full["y"], dtype=torch.float32, device=device)
X_train, y_train           = torch.tensor(train["X"]     , device=device),      torch.tensor(train["y"],      dtype=torch.float32, device=device)
X_val, y_val               = torch.tensor(val["X"]       , device=device),      torch.tensor(val["y"],        dtype=torch.float32, device=device)
X_test, y_test             = torch.tensor(test["X"]      , device=device),      torch.tensor(test["y"],       dtype=torch.float32, device=device)


train_mean = X_train.mean(dim=0)
train_std = X_train.std(dim=0) + 1e-6


X_train = normalise_input(X_train, train_mean, train_std)
X_val = normalise_input(X_val, train_mean, train_std)
X_test = normalise_input(X_test, train_mean, train_std)


# ### TRAIN MODEL


DEFAULT_CASE = X_train.mean(axis=0)

X_DEFAULTS = DEFAULT_CASE.tile(len(all_y), 1)
Y_DEFAULTS = torch.tensor(all_y, device=device).flip([0])


MAX_ITERS = len(X_train)
EPOCHS = 5000
USE_SYMMETRIC_ATTACKS = False
LR = 2e-1
TEMPERATURE = 0.05
USE_BLOCKERS = True
USE_SUPPORTS = True

ALPHA = 0
BETA = 0
GAMMA = 0.005
GAMMA_PRIME = 0.005

N = 100

best_f1 = 0
best_seed = -1

for torch_seed in range(N):

    torch.manual_seed(torch_seed) # TRY DIFFERENT INITIAL WEIGHTS 


    no_features = X_train.shape[-1]
    semantics = rs.ReluSemantics(max_iters=MAX_ITERS, epsilon=0)

    # pofe = fwe.FeatureWeightedExtractor(no_features)
    pofe = mlpe.MLPExtractor(no_features, [32], 1)
    bsfe = pofe
    bs_scaler = scaler.Scaler(bsfe.get_output_features(), weight=1.0)
    comp_func = cpo.Subtractor(temperature=TEMPERATURE, activation=torch.sigmoid)

    partial_order = lpo.LearnedPartialOrder([pofe], comparison_func=comp_func)
    irrelevance = ri.RegularIrrelevance(partial_order)
    base_score = lbs.LearnedBaseScore([bsfe, bs_scaler], activation=torch.sigmoid)

    model = gradual_aacbr.GradualAACBR(semantics, 
                                    base_score,
                                    irrelevance,
                                    partial_order).to(device)


    regulariser = lambda model: regularise(model, [
        [sparsity_regulariser, ALPHA], 
        [connectivity_regulariser, BETA], 
        [community_prev_reg_attacks, GAMMA],
        [community_prev_reg_supports, GAMMA_PRIME],
        # [community_preservation_regulariser, GAMMA],
        # [feature_smoothness_regulariser, alpha]
        ])


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)


    def uni_directional(A):

        A = torch.where(torch.abs(A) > torch.abs(A.T), A, 0)
        return A

    POST_PROCESS_FUNC = uni_directional
    # POST_PROCESS_FUNC = lambda x: x

    print("TRAINING WITH SEED:", torch_seed)


    losses = static_train_model(model, X_train, y_train, 
                    X_DEFAULTS, Y_DEFAULTS, optimizer, 
                    criterion, EPOCHS, X_new_cases=X_train, y_new_cases=y_train, 
                    use_symmetric_attacks=False, use_blockers=USE_BLOCKERS, 
                    plot_loss_curve=True,
                    disable_tqdm=False, post_process_func=POST_PROCESS_FUNC, regularise_graph=regulariser,
                    use_supports = USE_SUPPORTS)


    print("RESULTS ON TRAIN SET POST TRAINING")
    with torch.no_grad():
        accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_DEFAULTS, Y_DEFAULTS, 
                                                        X_train, y_train, show_confusion=False, use_blockers=USE_BLOCKERS,  
                                                        print_matrix = False, print_compute_graph=False, 
                                                        print_graph=False, print_results=True, post_process_func=POST_PROCESS_FUNC, 
                                                        use_supports = USE_SUPPORTS)
        
        if f1 > best_f1:
            best_f1 = f1
            best_seed = torch_seed

print("Best F1:", best_f1)
print("Best Seed:", best_seed)


    # print("RESULTS ON VALIDATION SET POST TRAINING")
    # with torch.no_grad():
    #     accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_DEFAULTS, Y_DEFAULTS, 
    #                                                      X_val, y_val, show_confusion=True, use_blockers=USE_BLOCKERS,  
    #                                                      print_matrix=True, print_compute_graph=False, 
    #                                                      print_graph=False, print_results=True, post_process_func=POST_PROCESS_FUNC, 
    #                                                      use_supports = USE_SUPPORTS)

