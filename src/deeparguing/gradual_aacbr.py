from typing import Callable, Tuple, override

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
from torch import Tensor

from deeparguing.base_scores.compute_base_scores import BaseScoreType
from deeparguing.casebase_edge_weights.compute_partial_order import \
    PartialOrderType
from deeparguing.irrelevance_edge_weights.compute_irrelevance import \
    IrrelevanceType
from deeparguing.semantics.gradual_semantics import GradualSemantics
from deeparguing.t_norm import (GodelTNorm, LukasiewiczTNorm, ProductTNorm,
                                TNorm)


class GradualAACBR(torch.nn.Module):

    def __init__(
        self,
        gradual_semantics: GradualSemantics,
        compute_base_score: BaseScoreType,
        irrelevance_edge_weights: IrrelevanceType,
        casebase_edge_weights: PartialOrderType,
        use_symmetric_attacks: bool = True,
        defaults_not_attack: bool = True,
        use_blockers: bool = True,
        use_supports: bool = False,
        post_process_func: Callable[[Tensor], Tensor] = lambda x: x,
        dimensions: int = 1,
        rescale_edges: bool = False,
        t_norm_str: str = "ProductTNorm",
        rand_weight: float = 0.0,
    ):
        """
         Gradual AACBR Model

         Parameters
         ----------
         gradual_semantics : GradualSemantics
             A gradual_semantics to evaluate the edge-weighted QBAF
         compute_base_score : ComputeBaseScores
             The function that computes the base score from the arguments.
             This is the intrinsic strength of each argument.
         irreleance_edge_weights : ComputeIrrelevance
             The function that computes the degree of irrelevance
             between the new case and casebase arguments.
         casebase_edge_weights : ComputePartialOrder
             The function that computes the soft ordering between two
             arguments in the casebase.
         use_symmetric_attack : bool, default True
             When true, symmetric attacks between cases of the same
             characterisation are included
         defaults_not_attack : bool, default True
             When true, the default arguments will not be able to attack any
             case in the casebase
         use_blockers : bool, default True
             When true, the model will optimise attacks of minimal between cases
             of minimal difference
         use_supports : bool, default False
             When true, the model will also consider supports between cases
             with the same outcome
        post_process_func: Callable
             Optionally apply a post process function to the adjacency matrix
             before computing the gradual semantics
        """
        super().__init__()

        self.gradual_semantics = gradual_semantics
        self.compute_base_scores = compute_base_score
        self.casebase_edge_weights = casebase_edge_weights
        self.irrelevance_edge_weights = irrelevance_edge_weights
        self.use_symmetric_attacks = use_symmetric_attacks
        self.defaults_not_attack = defaults_not_attack
        self.use_blockers = use_blockers
        self.use_supports = use_supports
        self.post_process_func = post_process_func
        self.dimensions = dimensions
        self.W = torch.nn.Parameter(torch.ones(dimensions, dtype=torch.float32))
        self.A = None
        self.rescale_edges = rescale_edges
        t_norm_map = {
            "ProductTNorm": ProductTNorm(),
            "GodelTNorm": GodelTNorm(),
            "LukasiewiczTNorm": LukasiewiczTNorm(),
        }
        self.t_norm: TNorm = t_norm_map[t_norm_str]
        self.rand_weight = rand_weight

    @property
    def device(self):
        return next(self.parameters(), torch.tensor(0)).device

    def fit(
        self,
        X_train: Tensor,
        y_train: Tensor,
        X_default: Tensor,
        y_default: Tensor,
        batch_size: int | None = None,
    ):
        """
        Builds the Edge-Weighted Quantitative Bipolar Argumentation
        Framework for the casebase.

        Parameters
        ----------
        X_train : Tensor
            Input casebase argument characterisations as a tensor.
            Shape (N, x1, ..., xn) where N is the number of casebase
            arguments and (x1, ..., xn) is the shape of each argument.
        y_train : Tensor
            Input casebase label as a tensor. Shape (N, Y) where N is the
            number of casebase arguments and Y is the number of labels
        X_default : Tensor
            Default arguments characterisations as a tensor.
            Shape (Y, x1, ..., xn) where Y is the number of labels
            and (x1, ..., xn) is the shape of each argument.
        y_default : Tensor
            Input casebase label as a tensor. Shape (Y, Y) where Y is the
            number of labels

        Notes
        -----
        The fit algorithm uses torch vector and batching operations to
        improve efficiency. See fit_slow for a direct implementation of the
        definition.

        Fit must be called before calling any subsequent function that uses
        the adjacency matrix self.A

        """

        if len(X_train) != len(y_train):
            raise (
                Exception(
                    f"Length of X_train must match length of y_train. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
                )
            )

        if len(X_default) != len(y_default):
            raise (
                Exception(
                    f"Length of X_default must match length of y_default. X_default shape: {X_default.shape}, y_default shape: {y_default.shape}"
                )
            )

        device = X_train.device

        X_train, y_train, default_indexes, indexes, attackers_default_mask = (
            self.__prepare_default(X_train, y_train, X_default, y_default)
        )
        X_attackers, X_targets, y_attackers, y_targets = self.__prepare_casebase(
            X_train, y_train
        )

        train_size = len(X_train)
        edge_weights_strict = self.__casebase_edge_weights_strict(
            X_attackers, X_targets, train_size
        )

        no_heads = edge_weights_strict.shape[-1]

        if self.defaults_not_attack:
            edge_weights_strict = torch.where(
                attackers_default_mask.unsqueeze(-1),
                torch.zeros_like(edge_weights_strict, device=device),
                edge_weights_strict,
            )

        attacks, differing_labels = self.__potential_attacks(
            edge_weights_strict, y_attackers, y_targets, train_size
        )

        blocked_attacks = self.__minimal_attacks(
            y_train, edge_weights_strict, attacks, indexes, batch_size
        )

        if self.use_supports:
            supports, _ = self.__potential_supports(
                edge_weights_strict, y_attackers, y_targets, train_size
            )
            blocked_supports = self.__minimal_supports(edge_weights_strict, batch_size)
            self.B = torch.mul(supports, blocked_supports)
        else:
            self.B = torch.zeros_like(edge_weights_strict)

        symmetric_attacks = self.__symmetric_attacks(
            X_attackers,
            X_targets,
            attackers_default_mask,
            train_size,
            differing_labels,
            no_heads,
        )

        self.A = torch.mul(attacks, blocked_attacks) + symmetric_attacks
        if self.rescale_edges:
            # TODO: This assumes there is the same number of cases per class in the casebase
            # which might not always be the case
            no_classes = y_train.shape[1]
            no_per_class = len(y_train) / no_classes
            self.A = (1 / ((no_classes - 1) * no_per_class)) * self.A
            self.B = (1 / (no_per_class)) * self.B

            
        if self.training and self.rand_weight != 0:
            noise = self.rand_weight * (torch.rand_like(self.A) * 0.25 + 0.5)
            self.A = self.A + noise
            self.B = self.B + noise

        self.A = -self.A + self.B
        self.X_train = X_train
        self.y_train = y_train
        self.default_indexes = default_indexes

    def __casebase_edge_weights_strict(
        self, attacker: Tensor, target: Tensor, train_size: int
    ) -> Tensor:
        edge_weights = self.casebase_edge_weights(attacker, target)

        # Normalize shape: handle various edge weight output formats
        # Expected final shape: (train_size, train_size, d)
        if edge_weights.ndim == 1:
            # Scalar per pair flattened: (n*n,) -> (n, n, 1)
            if edge_weights.shape[0] == train_size * train_size:
                edge_weights = edge_weights.reshape((train_size, train_size, 1))
            # One scalar per argument (n,) -> broadcast not supported, error
            else:
                raise Exception(
                    f"edge_weights has shape {edge_weights.shape}, expected "
                    f"({train_size * train_size},) or ({train_size}, {train_size}) "
                    f"or ({train_size}, {train_size}, d)"
                )
        elif edge_weights.ndim == 2:
            # Shape (n, n) -> (n, n, 1)
            edge_weights = edge_weights.unsqueeze(-1)
        # else: already 3D (n, n, d)

        edge_weights = edge_weights.reshape((train_size, train_size, -1))
        return self.t_norm.and_op(
            edge_weights, self.t_norm.not_op(torch.transpose(edge_weights, 0, 1))
        )

    def __casebase_edge_weights_equal(
        self, attacker: Tensor, target: Tensor, train_size: int
    ):
        edge_weights = self.casebase_edge_weights(attacker, target)

        # Normalize shape: handle various edge weight output formats
        # Expected final shape: (train_size, train_size, d)
        if edge_weights.ndim == 1:
            # Scalar per pair flattened: (n*n,) -> (n, n, 1)
            if edge_weights.shape[0] == train_size * train_size:
                edge_weights = edge_weights.reshape((train_size, train_size, 1))
            else:
                raise Exception(
                    f"edge_weights has shape {edge_weights.shape}, expected "
                    f"({train_size * train_size},) or ({train_size}, {train_size}) "
                    f"or ({train_size}, {train_size}, d)"
                )
        elif edge_weights.ndim == 2:
            # Shape (n, n) -> (n, n, 1)
            edge_weights = edge_weights.unsqueeze(-1)
        # else: already 3D (n, n, d)

        edge_weights = edge_weights.reshape((train_size, train_size, -1))

        return self.t_norm.and_op(edge_weights, torch.transpose(edge_weights, 0, 1))

    def __prepare_default(
        self, X_train: Tensor, y_train: Tensor, X_default: Tensor, y_default: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        X_train, y_train, default_indexes = self._add_default_cases(
            X_train, y_train, X_default, y_default
        )
        train_size = len(X_train)
        device = X_train.device

        indexes = torch.arange(train_size, device=X_train.device)
        idx_attackers = indexes.unsqueeze(1).expand(-1, len(indexes)).reshape(-1)
        attackers_default_mask = (
            torch.isin(idx_attackers, default_indexes)
            .reshape((train_size, train_size))
            .to(device)
        )
        return X_train, y_train, default_indexes, indexes, attackers_default_mask

    def __prepare_casebase(
        self, X_train: Tensor, y_train: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        X_attackers = X_train
        X_targets = X_train

        y_attackers = y_train.unsqueeze(1).expand(-1, len(y_train), -1)
        y_targets = y_train.unsqueeze(0).expand(len(y_train), -1, -1)

        return X_attackers, X_targets, y_attackers, y_targets

    def __potential_supports(
        self,
        edge_weights_strict: Tensor,
        y_attackers: Tensor,
        y_targets: Tensor,
        train_size: int,
    ) -> Tuple[Tensor, Tensor]:

        same_labels = torch.all(y_attackers == y_targets, dim=-1)
        same_labels = torch.reshape(same_labels, (train_size, train_size))

        supports = torch.where(same_labels.unsqueeze(-1), edge_weights_strict, 0)
        # Prevent argument from supporting itself
        mask_self = 1 - torch.diag(
            torch.ones((len(edge_weights_strict)), device=supports.device)
        ).unsqueeze(-1)
        supports = torch.mul(supports, mask_self)

        return supports, same_labels

    def __potential_attacks(
        self,
        edge_weights_strict: Tensor,
        y_attackers: Tensor,
        y_targets: Tensor,
        train_size: int,
    ) -> Tuple[Tensor, Tensor]:

        differing_labels = torch.any(y_attackers != y_targets, dim=-1)
        differing_labels = torch.reshape(
            differing_labels, (train_size, train_size)
        ).unsqueeze(-1)

        attacks = torch.where(differing_labels, edge_weights_strict, 0)

        return attacks, differing_labels

    def __symmetric_attacks(
        self,
        X_attackers: Tensor,
        X_targets: Tensor,
        attackers_default_mask: Tensor,
        train_size: int,
        differing_labels: Tensor,
        no_heads: int,
    ) -> Tensor:

        if self.use_symmetric_attacks:
            symmetric_attacks = self.__casebase_edge_weights_equal(
                X_attackers, X_targets, train_size
            )
            symmetric_attacks = torch.where(differing_labels, symmetric_attacks, 0)
            if self.defaults_not_attack:
                symmetric_attacks = torch.where(
                    attackers_default_mask.unsqueeze(-1), 0, symmetric_attacks
                )

        else:
            symmetric_attacks = torch.zeros(
                (train_size, train_size, no_heads), device=X_attackers.device
            )
        return symmetric_attacks

    def _compute_blocked_product_batched(
        self, A: Tensor, B: Tensor, batch_size: int
    ) -> Tensor:
        """
        Compute elementwise:
            result[i, j, d] = ∏ₖ (1 - A[i, k, d] * B[j, k, d])

        Parameters:
        A: Tensor of shape (n_rows, n, d)
        B: Tensor of shape (m, n, d)
        batch_size: Number of k indices to process at once.

        Returns:
        result: Tensor of shape (n_rows, m, d)
        """
        n_rows, n, d = A.shape
        m = B.shape[0]
        result = torch.ones(n_rows, m, d, device=A.device, dtype=A.dtype)
        for k in range(0, n, batch_size):
            # Process a chunk of the k-dimension.
            A_chunk = A[:, k : k + batch_size, :]  # Shape: (n_rows, batch_size, d)
            B_chunk = B[:, k : k + batch_size, :]  # Shape: (m, batch_size, d)
            # Expand dimensions to broadcast:
            # A_chunk -> (n_rows, 1, batch_size, d)
            # B_chunk -> (1, m, batch_size, d)
            # Then compute the product over the chunk dimension.
            term = self._compute_blocked_product(A_chunk, B_chunk)
            result = self.t_norm.and_op(result, term)
        return result

    def _compute_blocked_product(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Compute blocked product for 3D tensors.

        Parameters:
        A: Tensor of shape (n, n, d)
        B: Tensor of shape (n, n, d)

        Returns:
        result: Tensor of shape (n, n, d)
        """
        A = A.unsqueeze(1)  # Shape (n, 1, n, d)
        B = B.unsqueeze(0)  # Shape (1, n, n, d)
        # eps = 1e-10
        term = self.t_norm.and_op(A, B)
        term = self.t_norm.not_op(term)
        # term = 1 - (A * B)

        # P = exp( sum( log(x) ) ) - sum over k dimension (dim=2)
        # result = torch.exp(torch.sum(torch.log(term + eps), dim=2))
        result = self.t_norm.aggregate(term, dim=2)
        return result

    def __minimal_supports(
        self, edge_weights_strict: Tensor, batch_size: int | None
    ) -> Tensor:
        """
        Compute minimal supports.

        When use_blockers is True, compute:
        blocked_supports[i,j] = ∏ₖ (1 - edge_weights_strict[i, k] * edge_weights_strict[j, k])
        Otherwise, return a tensor of ones.
        """
        if self.use_blockers:
            A = edge_weights_strict
            B = torch.transpose(edge_weights_strict, 0, 1)
            if batch_size is not None:
                blocked_supports = self._compute_blocked_product_batched(
                    A, B, batch_size
                )
            else:
                blocked_supports = self._compute_blocked_product(A, B)
        else:
            blocked_supports = torch.ones_like(edge_weights_strict)
        return blocked_supports

    def __minimal_attacks(
        self,
        y_train: Tensor,
        edge_weights_strict: Tensor,
        attacks: Tensor,
        indexes: Tensor,
        batch_size: int | None,
    ) -> Tensor:
        """
        Compute minimal attacks.

        When use_blockers is True, first determine which pairs share the same label,
        then compute:
        blocked_attacks[i,j] = ∏ₖ (1 - A[i, k] * B[j, k])
        where
        A = torch.where(same_labels, edge_weights_strict, 0)
        B = attacks.T
        Otherwise, return a tensor of ones.
        """
        if self.use_blockers:
            i_indices = indexes.unsqueeze(1)
            j_indices = indexes.unsqueeze(0)
            same_labels = torch.all(
                y_train[i_indices] == y_train[j_indices], dim=-1
            ).unsqueeze(-1)

            A = torch.where(same_labels, edge_weights_strict, 0)
            B = torch.transpose(attacks, 0, 1)
            if batch_size is not None:
                blocked_attacks = self._compute_blocked_product_batched(
                    A, B, batch_size
                )
            else:
                blocked_attacks = self._compute_blocked_product(A, B)
        else:
            blocked_attacks = torch.ones_like(edge_weights_strict)
        return blocked_attacks

    def _add_default_cases(
        self, X_train: Tensor, y_train: Tensor, X_default: Tensor, y_default: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        default_index_start = len(X_train)
        X_train = torch.cat((X_train, X_default), dim=0)
        y_train = torch.cat((y_train, y_default), dim=0)
        default_index_end = len(X_train)
        default_indexes = torch.arange(
            default_index_start, default_index_end, device=X_train.device
        )

        return X_train, y_train, default_indexes

    @override
    def forward(self, new_cases: Tensor, return_all_strengths: bool = False) -> Tensor:
        """
        Computes the final strenghts of the EW-QAF for each new_case input

        Parameters
        ----------
        new_cases : Tensor
            Input newcase argument characterisations as a tensor.
            Shape (N, x1, ..., xn) where N is the number of new cases
            arguments and (x1, ..., xn) is the shape of each argument.


        Returns
        -------
        final_strengths : Tensor
            The final strenghts of the arguments.
            if return_all_strenghts then shape is (N, C), where N is the
            number of new cases and C is the number of cases in the
            casebase. Otherwise shape is (N, D) where D is the number of
            default cases.


        Notes
        -----
        This function will raise an exception if the fit function has not
        been previously called.

        """

        if self.A is None:
            raise (Exception("Ensure the model has been fit first."))
        batch_size = new_cases.shape[0]

        base_scores = self.compute_base_scores(self.X_train)  # (n)
        base_scores = self.__batch_base_scores(base_scores, batch_size)  # B x n x d
        base_scores = self.__new_case_influence(self.X_train, base_scores, new_cases)

        A = self.post_process_func(self.A)
        assert A.shape == self.A.shape
        strengths = self.gradual_semantics(A, base_scores)

        # TODO: Check if this is necessary:
        # final_strengths = strengths[-1].squeeze()
        final_strengths = strengths.squeeze()
        if final_strengths.dim() == 1:
            final_strengths = final_strengths.unsqueeze(0)

        if return_all_strengths:
            return final_strengths

        # Only apply linear combination when d > 1
        if self.dimensions > 1:
            final_strengths = torch.matmul(
                final_strengths, self.W
            )  # (B, n, d) -> (B, n)

        else:
            final_strengths = final_strengths.squeeze(-1)  # (B, n, 1) -> (B, n)

        default_strengths = final_strengths[:, self.default_indexes]
        return default_strengths

    def __batch_base_scores(self, base_scores: Tensor, batch_size: int) -> Tensor:
        base_scores = torch.tile(
            base_scores.unsqueeze(dim=0), (batch_size, 1, 1)
        )  # B x n x d
        return base_scores

    def __new_case_influence(
        self, X_train: Tensor, base_scores: Tensor, new_cases: Tensor
    ) -> Tensor:
        new_cases_base_scores = self.compute_base_scores(new_cases).unsqueeze(
            -1
        )  # (B x d)

        new_cases_attacks_adjacency = self.irrelevance_edge_weights(
            new_cases, X_train
        )  # B x n x d
        new_cases_attacks_adjacency = -new_cases_attacks_adjacency

        new_cases_attacks_adjacency = new_cases_attacks_adjacency.unsqueeze(
            1
        )  # B x 1 x n x d

        # We compute the aggregations *only* for the attacks by the new cases.
        # As new cases are unattacked, this can be computed in a single pass of
        # aggregation/influence function
        aggregations = self.gradual_semantics.aggregation_func(
            new_cases_attacks_adjacency, new_cases_base_scores
        )
        strengths = self.gradual_semantics.influence_func(base_scores, aggregations)

        return strengths

    def show_graph(
        self, logger=lambda x: x, prevent_show=False, positions=None, threshold=0.0
    ):
        """
        Outputs a networkx graph of the casebase

        Notes
        -----
        This function will raise an exception if the fit function has not
        been previously called.

        """

        if self.A is None:
            raise (Exception("Ensure the model has been fit first."))

        A = self.post_process_func(self.A).detach().cpu().numpy()
        y_train = self.y_train.cpu().detach().numpy()
        if self.y_train.shape[-1] > 1:
            y_train = np.argmax(y_train, axis=1)
        else:
            y_train = y_train.squeeze()
        default_indexes = self.default_indexes.cpu().detach().numpy()

        A = np.where(np.abs(A) > threshold, A, 0)

        gr = nx.from_numpy_array(A, create_using=nx.DiGraph)
        if not positions:
            pos = nx.nx_agraph.graphviz_layout(
                gr, prog="dot", args="-Gsplines=true -Gnodesep=2"
            )
        else:
            default_pos = (
                lambda idx: (idx - default_indexes[0] + 1)
                * (1 / (len(default_indexes) + 2))
                * len(gr.nodes)
            )
            positions.update({idx: (default_pos(idx), -1) for idx in default_indexes})
            pos = positions

        unique_labels = np.unique(y_train)
        colormap = plt.get_cmap("rainbow", len(unique_labels))
        label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
        node_colors = [label_to_color[y_train[node]] for node in list(gr.nodes)]

        assert all(
            [default_index in list(gr.nodes) for default_index in default_indexes]
        )

        labels = {x: x for x in list(gr.nodes)}
        for i, default_index in enumerate(default_indexes):
            labels.update({default_index: "Default"})

        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(1, 1, 1)
        for i, label in enumerate(unique_labels):
            ax.plot([0], [0], color=colormap(i), label=f"Class: {label}")

        # For positive weights (0 to 1): white to green.
        top = cm.get_cmap("Greens", 128)(np.linspace(0, 1, 128))
        # For negative weights (–1 to 0): red to white.
        bottom = cm.get_cmap("Reds_r", 128)(np.linspace(0, 1, 128))
        white = np.array([[1, 1, 1, 1]])  # pure white

        newcolors = np.vstack((bottom, white, top))
        ew_colormap = ListedColormap(newcolors, name="WhiteRedGreen")

        _, weights = zip(*nx.get_edge_attributes(gr, "weight").items())
        norm = Normalize(vmin=-1, vmax=1)

        edge_colors = [ew_colormap(norm(w)) for w in weights]

        nx.draw(
            gr,
            pos,
            labels=labels,
            arrowstyle="-|>",
            arrows=True,
            node_color=node_colors,
            arrowsize=20,
            node_size=100,
            font_size=5,
            width=0.4,
            edge_color=edge_colors,
            edge_cmap=ew_colormap,
        )

        plt.legend()
        logger(f)
        if not prevent_show:
            plt.show()

    def show_matrix(self, logger=lambda x: x, prevent_show=False):
        """

        Outputs an image of the adjacency matrix of the casebase

        Notes
        -----
        This function will raise an exception if the fit function has not
        been previously called.

        """

        if self.A is None:
            raise (Exception("Ensure the model has been fit first."))

        f = plt.figure(figsize=(10, 10))
        A = self.post_process_func(self.A).detach().cpu().numpy()
        plt.imshow(A, cmap="cividis", interpolation="nearest")
        ax = plt.gca()
        n = len(self.X_train)
        # Major ticks
        ax.set_xticks(np.arange(0, n, 1))
        ax.set_yticks(np.arange(0, n, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(0, n, 1))
        ax.set_yticklabels(np.arange(0, n, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.2)
        plt.colorbar(label="Value")
        plt.xlabel("Nodes")
        plt.ylabel("Nodes")
        plt.title("Edges")
        logger(f)
        if not prevent_show:
            plt.show()
