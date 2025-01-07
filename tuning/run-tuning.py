import wandb
import argparse
import deeparguing.base_scores
import deeparguing.base_scores.learned_base_score
import deeparguing.casebase_edge_weights
import deeparguing.casebase_edge_weights.compute_partial_order
import deeparguing.casebase_edge_weights.learned_partial_order
import deeparguing.feature_extractor
import deeparguing.feature_extractor.feature_weighted_extractor
import deeparguing.feature_extractor.scaler
import deeparguing.feature_extractor.threshold_extractor
import deeparguing.irrelevance_edge_weights
import deeparguing.irrelevance_edge_weights.regular_irrelevance
import deeparguing.semantics
import deeparguing.semantics.relu_semantics
import deeparguing.semantics.sigmoid_semantics
import deeparguing.train
import helper

import deeparguing.train as dtrain
import torch
import torch.optim as optim
from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.train import evaluate_model
import deeparguing
import numpy as np
from deeparguing.regulariser import sparsity_regulariser, community_preservation_regulariser, connectivity_regulariser, feature_smoothness_regulariser, regularise
import random




wandb.login()

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for project configuration.")

    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(helper.loaders.keys()),
        help="Name of the dataset."
    )
    parser.add_argument(
        "--cluster",
        type=bool,
        default=False,
        help="Whether to cluster data points. Default is False"
    )
    parser.add_argument(
        "--training_method",
        type=str,
        choices=["dynamic", "static"],
        required=True,
        help="Training method, choose either 'dynamic' or 'static'."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials to average over. Default is 5"
    )


    args = parser.parse_args()


    return args



# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

def main():



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    run = wandb.init(
        tags=[f'dataset:{args.dataset}', 
              f'cluster:{args.cluster}', 
              f'training_method:{args.training_method}']
    )
    """
        Load data:

        If cluster is true:
            casebase = cluser_data(training_data)
            new_cases = training_data
        else:
            casebase = training_data[:80%]
            new_cases = training_data[:20%]
    

        If training_method is static:
            static_train(casebase, new_cases)
        else:
            repeat:
                if cluster:
                    # shuffle and split cluster data and use new_cases=training_data
                else:
                    # shuffle and split training data into casebase and new_case
                dynamic_train(casebase, new_cases)
    """

    config = wandb.config

    X, y = helper.load_dataset(args.dataset)
    no_features = X.shape[1]
    train_full, train, val, _ = helper.split_data(X, y, seed=42) 

    if args.cluster:
        X_centroids, y_centroids = dtrain.cluster_data(train["X"], train["y"], lambda _: wandb.config.cluster_size)
        casebase = {"X": X_centroids, "y": y_centroids}
        new_cases = {"X": train["X"], "y": train["y"]}
        
    else:
        casebase = {"X": train["X"][:config.casebase_size], "y": train["y"][:config.casebase_size]}
        new_cases = {"X": train["X"][config.casebase_size:], "y": train["y"][config.casebase_size:]}

    all_y = np.unique(y, axis=0)
    no_classes = len(all_y)
    
    casebase = helper.to_torch(casebase, device)
    new_cases = helper.to_torch(new_cases, device)
    train_full = helper.to_torch(train_full, device)
    train = helper.to_torch(train, device)
    val = helper.to_torch(val, device)

    DEFAULT_OPTIONS = {
        "zeros": torch.zeros(no_features, device=device),
        "average": torch.mean(train["X"], axis=0)
    }


    default = {"X":  DEFAULT_OPTIONS[config["default"]].tile(no_classes, 1), 
               "y": torch.tensor(all_y, device=device)}
    

    if config["normalise"]:
        mean = torch.mean(train["X"])
        std = torch.std(train["X"]) + 1e-6
        casebase["X"] = helper.normalise_input(casebase["X"], mean, std) 
        new_cases["X"] = helper.normalise_input(new_cases["X"], mean, std) 
        train_full["X"] = helper.normalise_input(train_full["X"], mean, std) 
        train["X"] = helper.normalise_input(train["X"], mean, std) 
        val["X"] = helper.normalise_input(val["X"], mean, std) 

    TRIALS = args.trials
    accuracies = np.zeros((TRIALS))
    precisions = np.zeros((TRIALS))
    recalls = np.zeros((TRIALS))
    f1s = np.zeros((TRIALS))
    best_pred = None
    best_loss = None
    best_seed = None


    for i in range(TRIALS):

        seed = random.randint(0, 100000)
        torch.manual_seed(seed)

        SEMANTICS = {
            "relu": deeparguing.semantics.relu_semantics.ReluSemantics,
            "sigmoid": deeparguing.semantics.sigmoid_semantics.SigmoidSemantics,
        } 
        semantics_type = config.semantics["type"]
        semantics_params = config.semantics[semantics_type]
        semantics = SEMANTICS[semantics_type](**semantics_params)
    
        run.tags = run.tags + ("semantics:"+semantics_type,)


        FEATURE_EXTRACTORS = {
        "ThresholdFeatureExtractor": deeparguing.feature_extractor.threshold_extractor.ThresholdFeatureExtractor,
        "Scaler": deeparguing.feature_extractor.scaler.Scaler, 
        "FeatureWeightedExtractor": deeparguing.feature_extractor.feature_weighted_extractor.FeatureWeightedExtractor,
        }

        feature_extactor_type = config.feature_extractor["type"]
        feature_extactors_params = config.feature_extractor[feature_extactor_type]

        po_feature_extractor = []
        bs_feature_extractor = []


    
        for feature_extractor_name in feature_extactors_params["extractor_list"]:
            shared = feature_extactors_params[feature_extractor_name]["shared"]
            params = feature_extactors_params[feature_extractor_name].copy()
            params.pop("shared")
            if shared:
                shared_extractor = FEATURE_EXTRACTORS[feature_extractor_name](no_features,**params)
                bs_fe = shared_extractor
                po_fe = shared_extractor
            else:
                bs_fe = FEATURE_EXTRACTORS[feature_extractor_name](no_features,**params)
                po_fe = FEATURE_EXTRACTORS[feature_extractor_name](no_features,**params)

            no_features = bs_fe.get_output_features() 
            po_feature_extractor.append(po_fe)
            bs_feature_extractor.append(bs_fe)
            run.tags = run.tags + ("feature_extractor:"+feature_extractor_name,)

        partial_order_config = config["partial_order"]
        partial_order_type = partial_order_config["type"]
        po_params = partial_order_config[partial_order_type]

        ACTIVATIONS = {
            "identity": lambda x: x,
            "relu": torch.relu,
            "sigmoid": torch.sigmoid,
        }

        # TODO: Not all partial orders may be a learned partial order, 
        # should configure this to be more general     
        comparison_func_name = po_params["comparison_func"]
        comparison_func_params = po_params[comparison_func_name].copy()
        comparison_func_params["activation"] = ACTIVATIONS[comparison_func_params["activation"]]
        comparison_func = deeparguing.casebase_edge_weights.compute_partial_order.Subtractor(**comparison_func_params)

        run.tags = run.tags + ("partial_order:" + partial_order_type,)
        partial_order = deeparguing.casebase_edge_weights.learned_partial_order.LearnedPartialOrder(po_feature_extractor, comparison_func)


        # TODO: Not all base scores may be a learned base score, 
        # should configure this to be more general     
        base_score_config = config["base_score"]
        base_score_type = base_score_config["type"]
        bs_params = base_score_config[base_score_type].copy()
        bs_params["activation"] = ACTIVATIONS[bs_params["activation"]]

        run.tags = run.tags + ("base_score:" + base_score_type,)
        base_score = deeparguing.base_scores.learned_base_score.LearnedBaseScore(bs_feature_extractor, **bs_params)

        # TODO: Not all irrelevance may be regular irrelevance, 
        # should configure this to be more general     
        irrelevance_config = config["irrelevance"]
        irrelevance_type = irrelevance_config["type"]

        irrelevance = deeparguing.irrelevance_edge_weights.regular_irrelevance.RegularIrrelevance(partial_order)


        run.tags = run.tags + ("irrelevance:" + irrelevance_type,)

        model = GradualAACBR(semantics,
                            base_score,
                            irrelevance,
                            partial_order).to(device)
    
    

        criterion = torch.nn.CrossEntropyLoss()

        OPTIMIZERS = {
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
        }


        optimizer_config = config["optimizer"]
        optimizer_type = optimizer_config["type"]
        optimizer_params = optimizer_config.get(optimizer_type, {})
        optimizer = OPTIMIZERS[optimizer_type](model.parameters(), lr=config["lr"], **optimizer_params)

        TRAINING_METHODS = {
            "static": deeparguing.train.static_train_model,
            "dynamic": deeparguing.train.dynamic_train_model,
        }


        train_func = TRAINING_METHODS[args.training_method]

        regulariser = lambda model: regularise(model, [
            [sparsity_regulariser, config["sparsity_weight"]], 
            [connectivity_regulariser, config["connectivity_weight"]], 
            [community_preservation_regulariser, config["community_preservation_weight"]],
            [feature_smoothness_regulariser, config["feature_smoothness_weight"]]
        ])


        losses = train_func(model, casebase["X"], casebase["y"], default["X"], default["y"], optimizer, criterion, config["epochs"], config["use_symmetric_attacks"], 
                        X_new_cases=new_cases["X"], y_new_cases=new_cases["y"], n_splits = config.get("n_splits", None), use_blockers=config["use_blockers"], 
                        plot_loss_curve=False, disable_tqdm=True, random_split_state=config.get("random_split_state", None),
                        regularise_graph = regulariser, tau=config["tau"])


        accuracy, precision, recall, f1, y_true, y_pred = evaluate_model(model, casebase["X"], casebase["y"], default["X"], default["y"], 
                                                        val["X"], val["y"],
                                                        show_confusion=False, print_matrix=False,
                                                        print_compute_graph=False, print_graph=False,
                                                        print_results=False, use_symmetric_attacks=config["use_symmetric_attacks"], 
                                                        use_blockers=config["use_blockers"], tau=config["tau"], return_predictions=True)


        accuracies[i] = accuracy
        precisions[i] = precision
        recalls[i] = recall
        f1s[i] = f1

        if f1s.max() == f1:
            best_pred = (y_true, y_pred)
            best_loss = losses
            best_seed = seed
        


    for loss in best_loss:
        wandb.log({"train_loss": loss})

    avg_accuracy = accuracies.mean()
    avg_precision = precisions.mean()
    avg_recall = recalls.mean()
    avg_f1 = f1s.mean()
    f1_std = accuracies.std()
    wandb.log({"f1": avg_f1, "accuracy": avg_accuracy, "precision": avg_precision, "recall": avg_recall, "f1_std": f1_std})
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=best_pred[0], preds=best_pred[1])})
    wandb.log({"seed": best_seed})

    

    



if __name__ == "__main__":
    main()
