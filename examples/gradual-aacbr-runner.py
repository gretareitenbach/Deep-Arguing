import argparse
import ast
import sys

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.optim import AdamW
from torchviz import make_dot
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.base_scores import *
from deeparguing.casebase_edge_weights import *
from deeparguing.clustering import *
from deeparguing.feature_extractor import *
from deeparguing.helper import *
from deeparguing.irrelevance_edge_weights import *
from deeparguing.regulariser import *
from deeparguing.semantics import *
from deeparguing.train import *

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
    "cross_entropy": lambda: torch.nn.CrossEntropyLoss(),
    "uni_directional": lambda A: torch.where(
        torch.abs(A) > torch.abs(A.T), A, 0
    ),  # todo move to own file?
    "normalize_data": normalize_data,
    "no_normalize": lambda a, b, c: a,
}


PLOT_NONE = 0
PLOT_BEFORE = 1
PLOT_AFTER = 2
PLOT_BOTH = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def evaluate_model(
    model,
    X_casebase,
    y_casebase,
    X_default,
    y_default,
    X_new_cases,
    y_new_cases,
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

    accuracy = accuracy_score(y_new_cases_orig, y_predicted)
    precision = precision_score(
        y_new_cases_orig, y_predicted, average="macro", zero_division=0.0
    )
    recall = recall_score(
        y_new_cases_orig, y_predicted, average="macro", zero_division=0.0
    )
    f1 = f1_score(y_new_cases_orig, y_predicted, average="macro", zero_division=0.0)
    cm = confusion_matrix(y_new_cases_orig, y_predicted)

    assert type(accuracy) == float
    assert type(precision) == float
    assert type(recall) == float
    assert type(f1) == float

    if print_compute_graph:
        single_value = final_strengths.sum()
        make_dot(single_value, params=dict(model.named_parameters())).render(
            "gradual_aacbr", format="pdf"
        )

    return accuracy, precision, recall, f1, cm


def print_results(accuracy, precision, recall, f1, cm, title, labels):
    results_title = "-" * 30 + f" RESULTS ON THE {title} SET " + "-" * 30
    print(results_title)
    print(
        f"Accuracy: {round(accuracy, 4)}; Precision: {round(precision, 4)}; Recall: {round(recall, 4)}; F1: {round(f1, 4)}"
    )
    print("-" * len(results_title))
    df = pd.DataFrame(
        cm,
        index=[f"Actual {torch.argmax(l).item()}" for l in labels],
        columns=[f"Predicted {torch.argmax(l).item()}" for l in labels],
    )

    print(df)


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
    data_dict["y_train"] = y_train
    data_dict["y_val"] = y_val
    data_dict["y_test"] = y_test

    X_train_mean = X_train.mean(dim=0)
    X_train_std = X_train.std(dim=0) + 1e-8

    data_dict["X_train_mean"] = X_train_mean
    data_dict["X_train_std"] = X_train_std

    if "data_pre_process" not in instances.keys():
        instances["data_pre_process"] = parse_entry(
            config["data_pre_process"], config, ref_stack
        )

    assert type(instances["data_pre_process"]) == dict

    data_pre_process = instances["data_pre_process"]
    data_dict["X_train"] = data_pre_process["normalize_func"](
        X_train, X_train_mean, X_train_std
    )
    data_dict["X_val"] = data_pre_process["normalize_func"](
        X_val, X_train_mean, X_train_std
    )

    data_dict["X_test"] = data_pre_process["normalize_func"](
        X_test, X_train_mean, X_train_std
    )

    print("Test Size:", len(X_test))
    print("Train Size:", len(X_train))
    print("Validation Size:", len(X_val))


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
        if not data_dict:
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
        if not data_dict:
            parse_entry(config["data"], config, ref_stack)
        ref_name = check_entry_for_value(entry)
        assert data_dict
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


def parse_seed(value):
    try:
        # Try parsing as a literal (e.g. list or tuple)
        parsed = ast.literal_eval(value)
        if isinstance(parsed, int):
            return [parsed]
        elif isinstance(parsed, list):
            if all(isinstance(x, int) for x in parsed):
                return parsed
            else:
                raise ValueError("List must contain only integers.")
        elif isinstance(parsed, tuple) and len(parsed) == 2:
            start, end = parsed
            if isinstance(start, int) and isinstance(end, int):
                if start <= end:
                    return list(range(start, end))
                else:
                    raise ValueError("Range start must be <= end.")
            else:
                raise ValueError("Range values must be integers.")
        else:
            raise ValueError("Unsupported format for --seed.")
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid --seed value: {value}. Error: {e}")


def read_config_files(config_file_paths):
    model_config = {}
    for config_file_path in config_file_paths:
        with open(config_file_path, "r") as file:
            new_config = yaml.safe_load(file)
            overlapping_keys = set(model_config.keys()) & set(new_config.keys())

            if overlapping_keys:
                print(
                    f"Error: Duplicate key(s) found in '{config_file_path}': {', '.join(overlapping_keys)}",
                    file=sys.stderr,
                )
                sys.exit(1)

            model_config.update(new_config)

    return model_config


def parse_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        nargs="+",
        type=str,
        help="Paths to model config files (space seperated)",
    )

    parser.add_argument(
        "--seed",
        type=parse_seed,
        default=[0],
        help="Torch seed(s). Examples: single value: 42, list of seeds: [1,2,3], or range of seeds: (0,10)",
    )

    parser.add_argument(
        "--disable_tqdm", "-dt", action="store_true", help="Disable tqdm"
    )

    parser.add_argument(
        "--run_test", "-rt", action="store_true", help="Run on the test set"
    )

    parser.add_argument("--plot_loss", "-pl", action="store_true", help="Plot the loss")

    parser.add_argument("--plot_gradients", "-pgr", action="store_true", help="Plot the gradients")

    parser.add_argument(
        "--plot_matrix",
        "-pm",
        type=int,
        choices=[PLOT_NONE, PLOT_BEFORE, PLOT_AFTER, PLOT_BOTH],
        default=0,
        help=(
            "Plot the casebase as a matrix heatmap\n"
            "  0 = plot nothing\n"
            "  1 = plot before training\n"
            "  2 = plot after training\n"
            "  3 = plot both before and after training"
        ),
    )
    parser.add_argument(
        "--plot_graph",
        "-pg",
        type=int,
        choices=[PLOT_NONE, PLOT_BEFORE, PLOT_AFTER, PLOT_BOTH],
        default=0,
        help=(
            "Plot the casebase as a graph\n"
            "  0 = plot nothing\n"
            "  1 = plot before training\n"
            "  2 = plot after training\n"
            "  3 = plot both before and after training"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_command_line()

    plot_matrix_before = args.plot_matrix in [PLOT_BEFORE, PLOT_BOTH]
    plot_matrix_after = args.plot_matrix in [PLOT_AFTER, PLOT_BOTH]
    plot_graph_before = args.plot_graph in [PLOT_BEFORE, PLOT_BOTH]
    plot_graph_after = args.plot_graph in [PLOT_AFTER, PLOT_BOTH]

    total = 0

    model_config = read_config_files(args.config)

    for seed_idx, seed in enumerate(args.seed):
        print("=" * 100)
        print(f"Running With Torch Seed: {seed}")
        torch.manual_seed(seed)
        instances = {}
        data_dict = {}

        parse_model_config(model_config)

        assert data_dict
        assert instances

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

        if plot_matrix_before:
            model.show_matrix()
        if plot_graph_before:
            model.show_graph()

        trainer.train(
            model,
            X_casebase,
            y_casebase,
            X_defaults,
            y_defaults,
            **train_settings,
            disable_tqdm=args.disable_tqdm,
        )

        acc, prec, rec, f1, cm = evaluate_model(
            model, X_casebase, y_casebase, X_defaults, y_defaults, X_val, y_val
        )
        print_results(acc, prec, rec, f1, cm, "VALIDATION", labels)

        if args.run_test:
            X_test = data_dict["X_test"]
            y_test = data_dict["y_test"]

            acc, prec, rec, f1, cm = evaluate_model(
                model, X_casebase, y_casebase, X_defaults, y_defaults, X_test, y_test
            )
            print_results(acc, prec, rec, f1, cm, "TEST", labels)

        if args.plot_loss:
            trainer.plot_loss_curve()
        if args.plot_gradients:
            trainer.plot_grads()
        if plot_matrix_after:
            model.show_matrix()
        if plot_graph_after:
            model.show_graph()

        print("=" * 100)
        if f1 > 0.7:
            total += 1
        if len(args.seed) > 1:
            print("PERCENT F1", total / (seed_idx + 1) * 100)
