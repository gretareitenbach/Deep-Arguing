import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import Tensor

from deeparguing import GradualAACBR
from deeparguing.base_scores import *
from deeparguing.casebase_edge_weights import *
from deeparguing.clustering import *
from deeparguing.evals.compute_graph import make_dot
from deeparguing.feature_extractor import *
from deeparguing.md_log import write_markdown_log
from deeparguing.helper import *
from deeparguing.irrelevance_edge_weights import *
from deeparguing.criterion.regularisers import *
from deeparguing.semantics import *
from deeparguing.train import *


def evaluate_model(
    model: GradualAACBR,
    X_casebase: Tensor,
    y_casebase: Tensor,
    X_default: Tensor,
    y_default: Tensor,
    X_new_cases: Tensor,
    y_new_cases: Tensor,
    batch_size: int | None = None,
    print_compute_graph: bool = False,
) -> Tuple[float, float, float, float, NDArray[...]]:
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
    model.eval()
    model.fit(X_casebase, y_casebase, X_default, y_default)

    n_samples = X_new_cases.shape[0]
    batch_size = batch_size if batch_size is not None else n_samples

    for i in range(0, len(X_new_cases), batch_size):
        X_batch = X_new_cases[i : i + batch_size]
        final_strengths = model(X_batch)
        batch_predictions = final_strengths.cpu().detach().numpy()
        if i == 0:
            y_predicted = batch_predictions
        else:
            y_predicted = np.concatenate((y_predicted, batch_predictions))

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
    cm = confusion_matrix(y_new_cases_orig, y_predicted, labels=np.arange(y_new_cases.shape[1]))

    # assert type(accuracy) == float
    # assert type(precision) == float
    # assert type(recall) == float
    # assert type(f1) == float
    accuracy = float(accuracy)
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)

    if print_compute_graph:
        single_value = final_strengths.sum()
        make_dot(single_value, params=dict(model.named_parameters())).render(
            "gradual_aacbr", format="pdf"
        )

    return accuracy, precision, recall, f1, cm


def print_results(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    cm: NDArray[Any],
    title: str,
    labels: list[str],
    log_path: str | None = None,
):
    results_title = "-" * 30 + f" RESULTS ON THE {title} SET " + "-" * 30
    logging.info(results_title)
    logging.info(
        f"Accuracy: {round(accuracy, 4)}; Precision: {round(precision, 4)}; Recall: {round(recall, 4)}; F1: {round(f1, 4)}"
    )
    logging.info("-" * len(results_title))
    df = pd.DataFrame(
        cm,
        index=[f"Actual {torch.argmax(l).item()}" for l in labels],
        columns=[f"Pred {torch.argmax(l).item()}" for l in labels],
    )

    logging.info(f"Confusion Matrix: \n{df}")

    if log_path is not None:
        write_markdown_log(
            [
                f"--- {title} RESULTS ---",
                f"Accuracy: {round(accuracy, 4)}; Precision: {round(precision, 4)}; "
                f"Recall: {round(recall, 4)}; F1: {round(f1, 4)}",
                f"Confusion matrix:\n```\n{df.to_string()}\n```",
            ],
            log_path,
        )
