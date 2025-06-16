import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import Tensor
from torchviz import make_dot

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


def evaluate_model(
    model: GradualAACBR,
    X_casebase: Tensor,
    y_casebase: Tensor,
    X_default: Tensor,
    y_default: Tensor,
    X_new_cases: Tensor,
    y_new_cases: Tensor,
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

def print_results(accuracy: float, precision: float, recall: float, f1: float, cm: NDArray[Any], title: str, labels: list[str]):
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

