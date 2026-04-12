from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.types import Number
from tqdm import tqdm

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.criterion import CriterionType


def generate_random_directions(param_vector: Tensor) -> Tuple[Tensor, Tensor]:
    """Generate two random normalized direction vectors of same size as param_vector."""
    d1: Tensor = torch.randn_like(param_vector)
    d2: Tensor = torch.randn_like(param_vector)
    d1 /= torch.norm(d1)
    d2 /= torch.norm(d2)
    return d1, d2


def compute_loss(
    model: GradualAACBR,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    X_train: Tensor,
    y_train: Tensor,
    X_casebase: Tensor,
    y_casebase: Tensor,
    X_defaults: Tensor,
    y_defaults: Tensor,
    regulariser: CriterionType | None,
) -> Number:
    """Compute loss on the full training data."""
    model.eval()
    with torch.no_grad():
        model.fit(X_casebase, y_casebase, X_defaults, y_defaults)
        outputs = model(X_train)
        y_target = torch.argmax(y_train, dim=1)
        loss = loss_fn(outputs, y_target) + (regulariser(model, outputs, y_target) if regulariser else 0)
    return loss.item()


def compute_loss_grid(
    theta_center: Tensor,
    model: GradualAACBR,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    X_train: Tensor,
    y_train: Tensor,
    X_casebase: Tensor,
    y_casebase: Tensor,
    X_defaults: Tensor,
    y_defaults: Tensor,
    d1: Tensor,
    d2: Tensor,
    regulariser: CriterionType | None = None,
    range_alpha: float = 20.0,
    range_beta: float = 20.0,
    steps: int = 30,
) -> Tuple[
    NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
]:
    """
    Compute the loss surface over a grid of alpha/beta steps around a central weight vector.
    """
    alphas = np.linspace(-range_alpha, range_alpha, steps)
    betas = np.linspace(-range_beta, range_beta, steps)
    losses = np.zeros((steps, steps))

    pbar = tqdm(range(steps), dynamic_ncols=True)
    for i in pbar:
        alpha = alphas[i]
        pbar.set_description(f"Generating perturbation: step: {i}/{len(alphas)}")
        for j, beta in enumerate(betas):
            delta = alpha * d1 + beta * d2
            perturbed_theta = theta_center + delta
            vector_to_parameters(perturbed_theta, model.parameters())
            loss = compute_loss(
                model,
                loss_fn,
                X_train,
                y_train,
                X_casebase,
                y_casebase,
                X_defaults,
                y_defaults,
                regulariser,
            )
            losses[i, j] = loss

    return alphas, betas, losses


def visualize_loss_landscape_3d(
    model: GradualAACBR,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    X_train: Tensor,
    y_train: Tensor,
    X_casebase: Tensor,
    y_casebase: Tensor,
    X_defaults: Tensor,
    y_defaults: Tensor,
    regulariser: CriterionType | None = None,
    range_alpha: float = 20.0,
    range_beta: float = 20.0,
    steps: int = 30,
) -> Tuple[Tensor, Tensor, Tensor, NDArray[np.floating[Any]]]:
    """
    Plot a 3D loss landscape by perturbing the model weights along two random directions.
    """
    # original_model = deepcopy(model)
    theta_star = parameters_to_vector(model.parameters()).detach()

    # Generate two random directions in parameter space
    d1, d2 = generate_random_directions(theta_star)

    alphas, betas, losses = compute_loss_grid(
        theta_star,
        model,
        loss_fn,
        X_train,
        y_train,
        X_casebase,
        y_casebase,
        X_defaults,
        y_defaults,
        d1=d1,
        d2=d2,
        steps=steps,
        range_alpha=range_alpha,
        range_beta=range_beta,
        regulariser=regulariser,
    )

    # Restore the original model parameters
    vector_to_parameters(theta_star, model.parameters())

    # Plotting the loss landscape
    Alpha, Beta = np.meshgrid(alphas, betas)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Alpha, Beta, losses.T, cmap="viridis")
    ax.set_title("3D Loss Landscape")
    ax.set_xlabel("Alpha direction")
    ax.set_ylabel("Beta direction")
    ax.set_zlabel("Loss")
    plt.show()

    theta_pre = theta_star.clone().detach()
    losses = losses.copy()
    return theta_pre, d1, d2, losses


def visualize_overlayed_loss_landscapes(
    model: GradualAACBR,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    X_train: Tensor,
    y_train: Tensor,
    X_casebase: Tensor,
    y_casebase: Tensor,
    X_defaults: Tensor,
    y_defaults: Tensor,
    theta_pre: Tensor,
    regulariser: CriterionType | None = None,
    range_alpha: float = 20.0,
    range_beta: float = 20.0,
    steps: int = 30,
):
    """
    Overlay pre- and post-training loss landscapes on the same 3D plot,
    showing their relative positions in parameter space.
    """

    theta_post = parameters_to_vector(model.parameters()).detach()
    d1 = theta_post - theta_pre
    d1 /= torch.norm(d1)
    v = torch.randn_like(d1)
    projection = torch.dot(v, d1) * d1
    v_orth = v - projection
    d2 = v_orth / v_orth.norm()

    alphas_pre, betas_pre, losses_pre = compute_loss_grid(
        theta_pre,
        model,
        loss_fn,
        X_train,
        y_train,
        X_casebase,
        y_casebase,
        X_defaults,
        y_defaults,
        d1=d1,
        d2=d2,
        steps=steps,
        range_alpha=range_alpha,
        range_beta=range_beta,
        regulariser=regulariser,
    )

    # Compute relative (alpha, beta) offset of post-training θ in the (d1, d2) frame
    delta = theta_post - theta_pre
    alpha_offset = torch.dot(delta, d1).item()
    beta_offset = torch.dot(delta, d2).item()

    # Compute post-training loss surface, centered at theta_post
    alphas_post, betas_post, losses_post = compute_loss_grid(
        theta_post,
        model,
        loss_fn,
        X_train,
        y_train,
        X_casebase,
        y_casebase,
        X_defaults,
        y_defaults,
        d1=d1,
        d2=d2,
        steps=steps,
        range_alpha=range_alpha,
        range_beta=range_beta,
        regulariser=regulariser,
    )

    vector_to_parameters(theta_post, model.parameters())

    # Plot both surfaces in the same frame
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    alphas = np.linspace(-range_alpha, range_alpha, steps)
    betas = np.linspace(-range_beta, range_beta, steps)
    Alpha, Beta = np.meshgrid(alphas, betas)

    Alpha_post = Alpha + alpha_offset
    Beta_post = Beta + beta_offset

    ax.plot_surface(
        Alpha, Beta, losses_pre.T, cmap="plasma", alpha=0.6, label="Pre-training"
    )
    ax.plot_surface(
        Alpha_post,
        Beta_post,
        losses_post.T,
        cmap="plasma",
        alpha=0.6,
        label="Post-training",
    )

    # Mark pre-training and post-training positions
    ax.scatter(
        0,
        0,
        losses_pre[steps // 2, steps // 2],
        color="blue",
        # s=60,
        label="Pre-training θ",
    )
    ax.scatter(
        alpha_offset,
        beta_offset,
        losses_post[steps // 2, steps // 2],
        color="red",
        # s=60,
        label="Post-training θ",
    )

    ax.set_title("Overlayed 3D Loss Landscapes (Pre vs Post Training)")
    ax.set_xlabel("Alpha direction")
    ax.set_ylabel("Beta direction")
    ax.set_zlabel("Loss")
    ax.legend()

    plt.show()
