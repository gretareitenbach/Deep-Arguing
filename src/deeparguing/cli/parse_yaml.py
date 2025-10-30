import logging
import sys
from typing import Any, Callable, Dict

import torch
import yaml
from optuna import Trial
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *

from deeparguing import GradualAACBR
from deeparguing.base_scores import *
from deeparguing.casebase_edge_weights import *
from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.clustering import *
from deeparguing.feature_extractor import *
from deeparguing.helper import *
from deeparguing.irrelevance_edge_weights import *
from deeparguing.regulariser import *
from deeparguing.semantics import *
from deeparguing.train import *

FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sigmoid": torch.sigmoid,
    "filter_to_attacks": filter_to_attacks,
    "filter_to_supports": filter_to_supports,
    "uni_directional": lambda A: torch.where(
        torch.abs(A) > torch.abs(A.T), A, 0
    ),  # todo move to own file?
    "identity": lambda A: A,
    "normalize_data": normalize_data,
    "no_normalize": lambda a, b, c: a,
    "use_train": lambda X_train, y_train, X_casebase, y_casebase: (X_train, y_train),
    "use_casebase": lambda X_train, y_train, X_casebase, y_casebase: (
        X_casebase,
        y_casebase,
    ),
}

instances: Dict[str, Any] = {}
data_dict: Dict[str, Any] = {}


def read_config_files(config_file_paths: list[str]) -> Dict[str, str]:
    model_config: Dict[str, str] = {}
    for config_file_path in config_file_paths:
        with open(config_file_path, "r") as file:
            new_config = yaml.safe_load(file)
            overlapping_keys = set(model_config.keys()) & set(new_config.keys())

            if overlapping_keys:
                logging.critical(
                    f"Error: Duplicate key(s) found in '{config_file_path}': {', '.join(overlapping_keys)}"
                )
                sys.exit(1)

            model_config.update(new_config)

    return model_config


def load_data_dict(
    entry: Dict[str, Any],
    config: Dict[str, Any],
    ref_stack: list[str],
    trial: Trial | None,
    device: str = "cpu",
):

    params: Dict[str, Any] = {
        k: parse_entry(v, config, ref_stack, trial)
        for k, v in entry.get("params", {}).items()
    }
    if entry["sub_type"] == "tabular":
        X, y = load_tabular_data(device=device, **params)
    elif entry["sub_type"] == "torch_imaging":
        X, y = load_torch_images(device=device, **params)
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
            config["data_pre_process"], config, ref_stack, trial
        )

    assert type(instances["data_pre_process"]) == dict

    data_pre_process: Dict[str, Any] = instances["data_pre_process"]
    data_dict["X_train"] = data_pre_process["pre_process_func"](
        X_train, X_train_mean, X_train_std
    )
    data_dict["X_val"] = data_pre_process["pre_process_func"](
        X_val, X_train_mean, X_train_std
    )

    data_dict["X_test"] = data_pre_process["pre_process_func"](
        X_test, X_train_mean, X_train_std
    )

    logging.debug(f"Test Size: {len(X_test)}")
    logging.debug(f"Train Size: {len(X_train)}")
    logging.debug(f"Validation Size, {len(X_val)}")
    return data_dict


def check_entry_for_value(entry: Dict[str, Any]):
    if "value" in entry:
        return entry["value"]
    else:
        raise ValueError(f"Entry: {entry} has no value.")


def parse_entry(
    entry: Dict[str, Any],
    config: Dict[str, Any],
    ref_stack: list[str],
    trial: Trial | None,
) -> Any:
    if "type" in entry:
        entry_type = entry["type"]
    else:
        raise ValueError(f"Entry has no type: {entry}")

    entry_sub_type = entry["sub_type"] if "sub_type" in entry else None

    if entry_type == "value":
        return check_entry_for_value(entry)

    if entry_type == "dict":
        children = check_entry_for_value(entry)
        result: Dict[str, Any] = {}
        for key, value in children.items():
            if key not in instances:
                child_entry = parse_entry(value, config, ref_stack, trial)
                instances[key] = child_entry
            else:
                child_entry = instances[key]
            result[key] = child_entry
        return result

    if entry_type == "list":
        children = check_entry_for_value(entry)
        result: list[Any] = []
        for child in children:
            result.append(parse_entry(child, config, ref_stack, trial))
        return result

    if entry_type == "class":
        class_name = entry["class_name"]
        params: Dict[str, Any] = {
            k: parse_entry(v, config, ref_stack, trial)
            for k, v in entry.get("params", {}).items()
        }
        if entry_sub_type == "optim":
            model_name = entry["model"]
            if model_name not in instances:
                instances[model_name] = parse_entry(
                    config[model_name], config, ref_stack, trial
                )
            params["params"] = instances[model_name].parameters()

        if entry_sub_type == "scheduler":
            optimizer = entry["optimizer"]
            if optimizer not in instances:
                instances[optimizer] = parse_entry(
                    config[optimizer], config, ref_stack, trial
                )
            params["optimizer"] = instances[optimizer]

        return globals()[class_name](**params)

    if entry_type == "ref":
        ref_name = check_entry_for_value(entry)
        if ref_name in ref_stack:
            raise ValueError(
                f"Cannot parse self or mutually recursive references: {ref_name}. References being parsed: {ref_stack}"
            )
        if ref_name not in instances:
            ref_stack.append(ref_name)
            instances[ref_name] = parse_entry(
                config[ref_name], config, ref_stack, trial
            )
            ref_stack.remove(ref_name)
        return instances[ref_name]

    if entry_type == "data_ref":
        if not data_dict:
            raise ValueError(f"Entry {entry} not found in the data specification.")
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

    if entry_type == "tune":
        return parse_tune_values(entry, config, ref_stack, trial)

    raise ValueError(
        f"Input is not formatted correctly, parsing failed.\nType: {entry_type}\nEntry: {entry}."
    )


def parse_tune_values(
    entry: Dict[str, Any],
    config: Dict[str, Any],
    ref_stack: list[Any],
    trial: Trial | None,
) -> Any:
    if trial is None:
        raise ValueError(
            "Attempt to parse tuning variables. Hyperparameter tuning is not turned on. Use -ht in the command line to ensure you are running with hyperparameter tuning on."
        )
    params: Dict[str, Any] = {
        k: parse_entry(v, config, ref_stack, trial)
        for k, v in entry.get("params", {}).items()
    }
    if "tune_type" not in entry:
        raise ValueError(f"Tune Type is not specified in entry: {entry}")
    elif entry["tune_type"] == "int":
        func = trial.suggest_int
    elif entry["tune_type"] == "float":
        func = trial.suggest_float
    elif entry["tune_type"] == "categorical":
        func = trial.suggest_categorical
    else:
        raise ValueError(
            f"Unsupported tune type {entry['tune_type']} for entry {entry}."
        )
    value: Any = func(**params)

    logging.debug(f"Hyperparameter: ({params['name']}: {value})")
    ExperimentLogger.current().log_params({params["name"]: value})

    return value


def parse_model_config(
    model_config: Dict[str, Any], trial: Trial | None = None, device: str = "cpu"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    global data_dict
    global instances

    instances = {}
    data_dict = {}

    if "data" not in model_config:
        raise ValueError("Cannot find data key in provided config files.")

    data_dict = load_data_dict(
        model_config["data"], model_config, ref_stack=[], trial=trial, device=device
    )

    for key, value in model_config.items():
        if key in instances or key == "data":
            continue
        instances[key] = parse_entry(value, model_config, ref_stack=[], trial=trial)

    return data_dict, instances
