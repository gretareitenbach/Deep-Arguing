import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add parent directory to sys.path
sys.path.append(parent_dir)

print(parent_dir)

import wandb
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

initalisation_methods = {
  "normal": lambda x: torch.nn.init.normal_(x),
  "He_uniform": lambda x: torch.nn.init.kaiming_uniform_(x.unsqueeze(1), mode="fan_in", nonlinearity="relu"),
  "He_normal": lambda x: torch.nn.init.kaiming_normal_(x.unsqueeze(1), mode="fan_in", nonlinearity="relu"),
  "LeCun": lambda x: torch.nn.init.kaiming_normal_(x.unsqueeze(1), mode='fan_in', nonlinearity='sigmoid'),
  "Xavier_uniform": lambda x: torch.nn.init.xavier_uniform_(x.unsqueeze(1)),  
  "Xavier_normal": lambda x: torch.nn.init.xavier_normal_(x.unsqueeze(1)),  
}

post_process_funcs = {
    "id": lambda A: A,
    "uni_directional": lambda A: torch.where(torch.abs(A) > torch.abs(A.T), A, 0)
}

# ### Train Model

sweep_config = {
    "method": "grid",
    # "method": "random",
    "name": "best_params_regularised",
    "metric": {"goal": "maximize", "name": "f1_percentage"},
    "parameters": {
        "epochs": {"values": [3000, 4500, 6000]},
        "max_iters": {"values": [len(X_train)]},
        "use_symmetric_attacks": {"values": [False]},
        "lr": {"values": [2e-4, 2e-3, 2e-2]},
        "temperature": {"values": [5e-2]},
        "use_blockers": {"values": [True]},
        "initialisation_method": {"values": ["Xavier_uniform"]},
        "alpha": {"values": [0]},
        "beta": {"values": [0]},
        "gamma": {"values": [5e-2, 5e-4, 5e-6]},
        "post_process_func": {"values": ["uni_directional"]},
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="gradual-aa-cbr-iris")


DEFAULT_CASE = X_train.mean(axis=0)

X_DEFAULTS = DEFAULT_CASE.tile(len(all_y), 1)
Y_DEFAULTS = torch.tensor(all_y, device=device).flip([0])

def main():
    run = wandb.init()
    N = 100

    MAX_ITERS = wandb.config["max_iters"]
    EPOCHS = wandb.config["epochs"]
    USE_SYMMETRIC_ATTACKS = wandb.config["use_symmetric_attacks"]
    LR = wandb.config["lr"]
    TEMPERATURE = wandb.config["temperature"]
    USE_BLOCKERS = wandb.config["use_blockers"] 
    INITIALISATION_METHOD = initalisation_methods[wandb.config["initialisation_method"]]
    ALPHA = wandb.config["alpha"]
    BETA = wandb.config["beta"]
    GAMMA = wandb.config["gamma"]
    POST_PROCESS_FUNC = post_process_funcs[wandb.config["post_process_func"]]

    totalf1 = 0

    for torch_seed in range(0, N):
    # torch_seed = 15


        print("="*40)
        print("Seed:", torch_seed)

        torch.manual_seed(torch_seed) # TRY DIFFERENT INITIAL WEIGHTS 

        no_features = X_train.shape[-1]
        semantics = rs.ReluSemantics(max_iters=MAX_ITERS, epsilon=0)

        pofe = fwe.FeatureWeightedExtractor(no_features, initialisation_method=INITIALISATION_METHOD)
        bsfe = pofe
        bs_scaler = scaler.Scaler(bsfe.get_output_features(), weight=1.0)
        comp_func = cpo.Subtractor(temperature=TEMPERATURE, activation=torch.sigmoid)

        partial_order = lpo.LearnedPartialOrder([pofe], comparison_func=comp_func)
        irrelevance = ri.RegularIrrelevance(partial_order)
        base_score = lbs.LearnedBaseScore([bsfe, bs_scaler], activation=torch.sigmoid)


        regulariser = lambda model: regularise(model, [
            [sparsity_regulariser, ALPHA], 
            [connectivity_regulariser, BETA], 
            [community_preservation_regulariser, GAMMA],
            # [feature_smoothness_regulariser, alpha]
            ])



        model = gradual_aacbr.GradualAACBR(semantics, 
                                        base_score,
                                        irrelevance,
                                        partial_order).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR)


        # POST_PROCESS_FUNC = lambda x: x


        losses = static_train_model(model, X_train, y_train, 
                        X_DEFAULTS, Y_DEFAULTS, optimizer, 
                        criterion, EPOCHS, X_new_cases=X_train, y_new_cases=y_train, 
                        use_symmetric_attacks=USE_SYMMETRIC_ATTACKS, use_blockers=USE_BLOCKERS, 
                        plot_loss_curve=False,
                        disable_tqdm=False, post_process_func=POST_PROCESS_FUNC, 
                        regularise_graph=regulariser)

        losses = np.array(losses)


        with torch.no_grad():
            accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_DEFAULTS, Y_DEFAULTS, 
                                                            X_val, y_val, show_confusion=False, use_blockers=USE_BLOCKERS, use_symmetric_attacks=USE_SYMMETRIC_ATTACKS,  
                                                            print_matrix=False, print_compute_graph=False, 
                                                            print_graph=False, print_results=False, post_process_func=POST_PROCESS_FUNC)

            print("VAL DATA RESULTS:")
            print("Accuracy, Precision, Recall, F1")
            print(accuracy, precision, recall, f1)
            print("="*40)
            if f1 > 0.7:
                totalf1 += 1
            print(f'Total with F1 > 0.7: {totalf1}/{torch_seed + 1}: {(totalf1/(torch_seed + 1)) * 100}%')

            wandb.log({f"val_acc_seed_{torch_seed}": accuracy,
                        f"val_precision_seed_{torch_seed}": precision,
                        f"val_recall_seed_{torch_seed}": recall,
                        f"val_f1_seed_{torch_seed}": f1,
                        f"val_loss_seed_{torch_seed}": losses[-1],
                        })

            wandb.log({f"val_acc": accuracy,
                        f"val_precision": precision,
                        f"val_recall": recall,
                        f"val_f1": f1,
                        f"val_loss": losses[-1],
                        })

    print("="*40)
    print(f'Total with F1 > 0.7: {totalf1}/{N}: {(totalf1/N) * 100}%')
    print("="*40)

    wandb.log({"f1_percentage": totalf1/N})



wandb.agent(sweep_id, function=main)