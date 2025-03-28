import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.optim import AdamW
from torchviz import make_dot

from deeparguing import GradualAACBR
from deeparguing.base_scores import LearnedBaseScore
from deeparguing.casebase_edge_weights import LearnedPartialOrder, Subtractor
from deeparguing.clustering import IdentityCluster, kMeansCluster
from deeparguing.feature_extractor import FeatureWeightedExtractor, Scaler
from deeparguing.helper import load_tabular_data, normalize_data, split_data
from deeparguing.irrelevance_edge_weights import RegularIrrelevance
from deeparguing.regulariser import (CommunityPreservationRegulariser,
                                     ConnectivityRegulariser, RegulariserList,
                                     SparsityRegulariser, filter_to_attacks,
                                     filter_to_supports)
from deeparguing.semantics import ReluSemantics, SigmoidSemantics
from deeparguing.train import (DynamicTrainer, ReweightTrainer, SimpleTrainer,
                               Trainer)

# TRAINING_METHODS = {
#     "SimpleTrainer": SimpleTrainer,
#     "DynamicTrainer": DynamicTrainer,
#     "ReweightTrainer": ReweightTrainer,
# }
#
# OPTIMISERS = {
#     "AdamW": optim.AdamW,
#     "SGD": optim.SGD,
# }
#
# SEMANTICS = {
#     "ReluSemantics": ReluSemantics,
#     "SigmoidSemantics": SigmoidSemantics,
# }
#
FUNCTIONS = {
    "sigmoid": torch.sigmoid,
    "filter_to_attacks": filter_to_attacks,
    "filter_to_supports": filter_to_supports,
    "weighted_cross_entropy": lambda weight: torch.nn.CrossEntropyLoss(weight=weight),
    "uni_directional": lambda A: torch.where(
        torch.abs(A) > torch.abs(A.T), A, 0
    ),  # todo move to own file?
    "normalize_data": normalize_data,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def evaluate_model(
    model,
    X_casebase,
    y_casebase,
    X_default,
    y_default,
    X_new_cases,
    y_new_cases,
    print_results=True,
    show_confusion=True,
    print_graph=False,
    print_matrix=False,
    print_compute_graph=False,
):
    """
    Fits and executes the model, then evaluates it on accuracy, precision,
    recall and f1

    Parameters
    ----------
    model : GradualAACBR
        GradualAACBR model to train
    X_casebase : torch.Tensor
        Input casebase argument characterisations as a tensor.
        Shape (N, x1, ..., xn) where N is the number of casebase
        arguments and (x1, ..., xn) is the shape of each argument.
    y_casebase : torch.Tensor
        Input casebase label as a tensor. Shape (N, Y) where N is the
        number of casebase arguments and Y is the number of labels
    X_default : torch.Tensor
        Default arguments characterisations as a tensor.
        Shape (Y, x1, ..., xn) where Y is the number of labels
        and (x1, ..., xn) is the shape of each argument.
    y_default : torch.Tensor
        Input casebase label as a tensor. Shape (Y, Y) where Y is the
        number of labels
    X_new_cases : torch.Tensor
        Input new_cases arguments characterisations as a tensor.
        Shape (M, x1, ..., xn) where M is the number of new cases
        and (x1, ..., xn) is the shape of each argument.
    y_new_cases : torch.Tensor
        Input casebase label as a tensor. Shape (M, Y) where M is the
        number of new cases and Y is the number of labels

    print_results : bool, default true
        When true, the accuracy, precision, recall and f1 is printed to
        console

    show_confusion : bool, default false
        When true, the confusion matrix graph is created

    print_graph : bool, default false
        When true, the model adjacency matrix is visualised as a connected
        graph

    print_matrix : bool, default false
        When true, the model adjacency matrix is visualised as a heatmap

    print_compute_graph : bool, default false
        When true, a pdf with the compute graph is outputted


    Returns
    -------
    results : Tuple
        returns a tuple containing the accuracy, precision, recall
        and f1 score

    """
    model.fit(X_casebase, y_casebase, X_default, y_default)
    final_strengths = model(X_new_cases)

    y_predicted = final_strengths.cpu().detach().numpy()
    y_predicted = np.argmax(y_predicted, axis=1)
    y_new_cases_orig = np.argmax(y_new_cases.cpu().detach().numpy(), axis=1)

    results = (
        accuracy_score(y_new_cases_orig, y_predicted),
        precision_score(
            y_new_cases_orig, y_predicted, average="macro", zero_division=0.0
        ),
        recall_score(y_new_cases_orig, y_predicted, average="macro", zero_division=0.0),
        f1_score(y_new_cases_orig, y_predicted, average="macro", zero_division=0.0),
    )

    if print_results:
        print("Accuracy, Precision, Recall, F1")
        print(results)

    if show_confusion:
        cm = confusion_matrix(y_new_cases_orig, y_predicted)
        print(cm)

    if print_graph:
        model.show_graph(
        )

    if print_matrix:
        model.show_matrix(
        )

    if print_compute_graph:
        single_value = final_strengths.sum()
        make_dot(single_value, params=dict(model.named_parameters())).render(
            "gradual_aacbr", format="pdf"
        )

    return results


def load_data_dict(entry, config, ref_stack):
    params = {
        k: parse_entry(v, config, ref_stack) for k, v in entry.get("params", {}).items()
    }
    if entry["sub_type"] == "tabular":
        X, y = load_tabular_data(device=device, **params)
    else:
        raise ValueError(f"Unknown data subtype: {entry['sub_type']}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, seed=42)
    data_dict["labels"] = torch.unique(y, dim=0)
    data_dict["no_features"] = X.shape[-1]
    data_dict["X_train"], data_dict["y_train"] = X_train, y_train
    data_dict["X_val"], data_dict["y_val"] = X_val, y_val
    data_dict["X_test"], data_dict["y_test"] = X_test, y_test
    data_dict["X_train_mean"] = X_train.mean(dim=0)
    data_dict["X_train_std"] = X_train.std(dim=0) + 1e-8


instances = {}
data_dict = {}


def check_entry_for_value(entry):
    if "value" in entry:
        return entry["value"]
    else:
        raise ValueError(f"Entry: {entry} has no value.")


def parse_entry(entry, config, ref_stack):
    if "type" in entry:
        entry_type = entry["type"]
    else:
        raise ValueError(f"Entry has no type: {entry}")

    entry_sub_type = entry["sub_type"] if "sub_type" in entry else None

    if entry_type == "data":
        load_data_dict(entry, config, ref_stack)
        return True

    if entry_type == "value":
        return check_entry_for_value(entry)

    if entry_type == "dict":
        children = check_entry_for_value(entry)
        result = {}
        for key, value in children.items():
            result[key] = parse_entry(value, config, ref_stack)
        return result

    if entry_type == "list":
        children = check_entry_for_value(entry)
        result = []
        for child in children:
            result.append(parse_entry(child, config, ref_stack))
        return result

    if entry_type == "class":
        class_name = entry["class_name"]
        params = {
            k: parse_entry(v, config, ref_stack)
            for k, v in entry.get("params", {}).items()
        }
        if entry_sub_type == "optim":
            model_name = entry["model"]
            if model_name not in instances:
                instances[model_name] = parse_entry(
                    config[model_name], config, ref_stack
                )
            params["params"] = instances[model_name].parameters()

        return globals()[class_name](**params)

    if entry_type == "ref":
        ref_name = check_entry_for_value(entry)
        if ref_name in ref_stack:
            raise ValueError(
                f"Cannot parse self or mutually recursive references: {ref_name}. References being parsed: {ref_stack}"
            )
        if ref_name not in instances:
            ref_stack.append(ref_name)
            instances[ref_name] = parse_entry(config[ref_name], config, ref_stack)
            ref_stack.remove(ref_name)
        return instances[ref_name]

    if entry_type == "data_ref":
        ref_name = check_entry_for_value(entry)
        assert ref_name in data_dict
        return data_dict[ref_name]

    if entry_type == "function":
        func_name = check_entry_for_value(entry)
        if func_name in FUNCTIONS:
            return FUNCTIONS[func_name]
        else:
            raise ValueError(f"Function {func_name} not found.")

    raise ValueError(
        f"Input is not formatted correctly, parsing failed.\nType: {entry_type}\nEntry: {entry}."
    )


def parse_model_config(config):
    for key, value in config.items():
        if key not in instances:
            instances[key] = parse_entry(value, model_config, ref_stack=[])


if __name__ == "__main__":

    # todo: change to make it accept arguments for the data file and model

    with open("examples/iris/iris.yaml", "r") as file:
        data_config = yaml.safe_load(file)

    with open("examples/iris/model.yaml", "r") as file:
        model_config = yaml.safe_load(file)

    torch.manual_seed(60)

    parse_model_config(data_config)

    data_pre_process = instances["data_pre_process"]
    data_dict["X_train"] = data_pre_process["normalize_func"](
        data_dict["X_train"], data_dict["X_train_mean"], data_dict["X_train_std"]
    )
    data_dict["X_val"] = data_pre_process["normalize_func"](
        data_dict["X_val"], data_dict["X_train_mean"], data_dict["X_train_std"]
    )

    parse_model_config(model_config)

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]

    model: GradualAACBR = instances["model"].to(device)

    trainer: Trainer = instances["trainer"]
    train_settings = instances["train_settings"]

    no_features = data_dict["no_features"]
    labels = data_dict["labels"]

    X_casebase, y_casebase = instances["build_casebase"](X_train, y_train)

    X_defaults = X_train.mean(dim=0).tile(len(labels), 1)
    y_defaults = labels.flip([0])

    trainer.train(
        model,
        X_casebase,
        y_casebase,
        X_defaults,
        y_defaults,
        **train_settings,
        disable_tqdm=False,
    )

    evaluate_model(model, X_casebase, y_casebase, X_defaults, y_defaults, X_val, y_val)
