from typing import Callable

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

type FilterFunc = Callable[[Tensor], Tensor]

filter_to_attacks: FilterFunc = lambda A: torch.where(A < 0, A, 0)
filter_to_supports: FilterFunc = lambda A: torch.where(A > 0, A, 0)


def threshold_adjacency(
    A: Tensor,
    threshold: float = 0.3,
    mode: str = "absolute",
) -> Tensor:
    """
    Apply thresholding to adjacency matrix to produce a cleaner DAG.

    This is the final step in NOTEARS: after training, small edge weights
    are set to zero to obtain a clean graph structure.

    Parameters
    ----------
    A : Tensor
        Adjacency matrix of shape (n, n) or (n, n, d) for batched.
    threshold : float
        Threshold value. Default 0.3 (common in NOTEARS literature).
    mode : str
        Thresholding mode:
        - "absolute": Zero out edges where |A[i,j]| < threshold
        - "relative": Zero out edges where |A[i,j]| < threshold * max(|A|)

    Returns
    -------
    Tensor
        Thresholded adjacency matrix with same shape as input.

    Examples
    --------
    >>> A = model.A.detach().clone()
    >>> A_clean = threshold_adjacency(A, threshold=0.3)
    >>> model.A.data = A_clean  # Update model in-place
    """
    if mode == "absolute":
        mask = torch.abs(A) >= threshold
    elif mode == "relative":
        max_val = torch.abs(A).max()
        mask = torch.abs(A) >= threshold * max_val
    else:
        raise ValueError(
            f"Unknown thresholding mode: {mode}. Use 'absolute' or 'relative'."
        )

    return A * mask


def apply_threshold_to_model(
    model: GradualAACBR,
    threshold: float = 0.3,
    mode: str = "absolute",
) -> None:
    """
    Apply thresholding to a trained model's adjacency matrix in-place.

    This is a convenience function for the final step in NOTEARS:
    after training converges, apply thresholding to obtain a clean DAG.

    Parameters
    ----------
    model : GradualAACBR
        Trained model with adjacency matrix.
    threshold : float
        Threshold value. Default 0.3 (common in NOTEARS literature).
    mode : str
        Thresholding mode: "absolute" or "relative".

    Raises
    ------
    Exception
        If model has not been fit (A is None).

    Examples
    --------
    >>> trainer.train(model, ...)
    >>> apply_threshold_to_model(model, threshold=0.3)
    """
    if model.A is None:
        raise Exception("Model has not been fit. Call model.fit() first.")

    with torch.no_grad():
        model.A.data = threshold_adjacency(model.A.data, threshold, mode)
