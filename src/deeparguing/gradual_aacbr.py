import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
from deeparguing.semantics.gradual_semantics import GradualSemantics
from deeparguing.irrelevance_edge_weights.compute_irrelevance import IrrelevanceType
from deeparguing.casebase_edge_weights.compute_partial_order import PartialOrderType
from deeparguing.base_scores.compute_base_scores import BaseScoreType 


class GradualAACBR(torch.nn.Module):

    def __init__(self, 
                 gradual_semantics: GradualSemantics, 
                 compute_base_score: BaseScoreType, 
                 irrelevance_edge_weights: IrrelevanceType, 
                 casebase_edge_weights: PartialOrderType,
                 use_symmetric_attacks = True, 
                 defaults_not_attack = True, 
                 use_blockers = True, 
                 use_supports=False, 
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
        self.A = None
    


    
    def fit(self, 
            X_train: torch.Tensor, 
            y_train: torch.Tensor, 
            X_default: torch.Tensor, 
            y_default: torch.Tensor, 
            batch_size=None):

        """
            Builds the Edge-Weighted Quantitative Bipolar Argumentation 
            Framework for the casebase. 

            Parameters
            ----------
            X_train : torch.Tensor
                Input casebase argument characterisations as a tensor. 
                Shape (N, x1, ..., xn) where N is the number of casebase 
                arguments and (x1, ..., xn) is the shape of each argument.
            y_train : torch.Tensor
                Input casebase label as a tensor. Shape (N, Y) where N is the 
                number of casebase arguments and Y is the number of labels
            X_default : torch.Tensor
                Default arguments characterisations as a tensor.
                Shape (Y, x1, ..., xn) where Y is the number of labels 
                and (x1, ..., xn) is the shape of each argument.
            y_default : torch.Tensor
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


        if (X_train is None or y_train is None or len(X_train) != len(y_train)):
            raise(Exception(f"Length of X_train must match length of y_train. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"))

        if (X_default is None or y_default is None or len(X_default) != len(y_default)):
            raise(Exception(f"Length of X_default must match length of y_default. X_default shape: {X_default.shape}, y_default shape: {y_default.shape}"))

        self.A = None
        
        X_train, y_train, default_indexes, indexes, attackers_default_mask = self.__prepare_default(X_train, y_train, X_default, y_default)
        X_attackers, X_targets, y_attackers, y_targets = self.__prepare_casebase(X_train, y_train)

        train_size = len(X_train)
        edge_weights_strict = self.__casebase_edge_weights_strict(X_attackers, X_targets, train_size)

        if self.defaults_not_attack:
            edge_weights_strict = torch.where(attackers_default_mask, 0, edge_weights_strict)

        attacks, differing_labels = self.__potential_attacks(edge_weights_strict, y_attackers, y_targets, train_size)

        blocked_attacks = self.__minimal_attacks(y_train, edge_weights_strict, attacks, indexes, batch_size)

        if self.use_supports:
            supports, _ = self.__potential_supports(edge_weights_strict, y_attackers, y_targets, train_size)
            blocked_supports = self.__minimal_supports(edge_weights_strict, batch_size)
            self.B = (torch.mul(supports, blocked_supports))
        else:
            self.B = torch.zeros_like(edge_weights_strict)

        symmetric_attacks = self.__symmetric_attacks(X_attackers, X_targets, attackers_default_mask, train_size, differing_labels)

        self.A = -(torch.mul(attacks, blocked_attacks) + symmetric_attacks) 
        self.A = self.A + self.B
        self.X_train = X_train
        self.y_train = y_train
        self.default_indexes = default_indexes

    def __casebase_edge_weights_strict(self, attacker, target, train_size):
        edge_weights = self.casebase_edge_weights(attacker, target).reshape((train_size, train_size))
        return edge_weights * (1 - (edge_weights.T))

    def __casebase_edge_weights_equal(self, attacker, target, train_size):
        edge_weights = self.casebase_edge_weights(attacker, target).reshape((train_size, train_size))
        return edge_weights * edge_weights.T

    def __prepare_default(self, X_train, y_train, X_default, y_default):
        X_train, y_train, default_indexes = self._add_default_cases(X_train, y_train, X_default, y_default)
        train_size = len(X_train)
        device = X_train.device

        indexes = torch.arange(train_size)
        idx_attackers = indexes.unsqueeze(1).expand(-1, len(indexes)).reshape(-1) 
        attackers_default_mask = torch.isin(idx_attackers, default_indexes).reshape((train_size, train_size)).to(device)
        return X_train, y_train, default_indexes, indexes, attackers_default_mask

    def __prepare_casebase(self, X_train, y_train):

        X_attackers = X_train.unsqueeze(1).expand(-1, len(X_train), -1)
        X_targets = X_train.unsqueeze(0).expand(len(X_train), -1, -1)

        y_attackers = y_train.unsqueeze(1).expand(-1, len(y_train), -1)
        y_targets = y_train.unsqueeze(0).expand(len(y_train), -1, -1)

        return X_attackers, X_targets, y_attackers, y_targets


    def __potential_supports(self, edge_weights_strict,  y_attackers, y_targets,  train_size):

        same_labels = torch.any(y_attackers == y_targets, dim=-1)
        same_labels = torch.reshape(same_labels, (train_size, train_size))

        supports = torch.where(same_labels, edge_weights_strict, 0)
        # Prevent argument from supporting itself
        mask_self = 1 - torch.diag(torch.ones((len(edge_weights_strict)), device=supports.device))
        supports = torch.mul(supports, mask_self)

        return supports, same_labels

    def __potential_attacks(self, edge_weights_strict,  y_attackers, y_targets,  train_size):

        differing_labels = torch.any(y_attackers != y_targets, dim=-1)
        differing_labels = torch.reshape(differing_labels, (train_size, train_size))

        attacks = torch.where(differing_labels, edge_weights_strict, 0)

        return attacks, differing_labels
    


    def __symmetric_attacks(self, X_attackers,  X_targets, 
                               attackers_default_mask, train_size, differing_labels):

        if self.use_symmetric_attacks: 
            symmetric_attacks = self.__casebase_edge_weights_equal(X_attackers, X_targets, train_size)
            symmetric_attacks = torch.where(differing_labels, symmetric_attacks, 0)
            if self.defaults_not_attack:
                symmetric_attacks = torch.where(attackers_default_mask, 0, symmetric_attacks)
            
        else:
            symmetric_attacks = torch.zeros((train_size, train_size)).to(X_attackers.device)
        
        return symmetric_attacks 


    

    def _compute_blocked_product_batched(self, A, B, batch_size):
        """
        Compute elementwise:
            result[i, j] = ∏ₖ (1 - A[i, k] * B[j, k])
    
        Parameters:
        A: Tensor of shape (n_rows, n)
        B: Tensor of shape (m, n)
        batch_size: Number of k indices to process at once.
    
        Returns:
        result: Tensor of shape (n_rows, m)
        """
        n_rows, n = A.shape
        m = B.shape[0]
        result = torch.ones(n_rows, m, device=A.device, dtype=A.dtype)
        for k in range(0, n, batch_size):
            # Process a chunk of the k-dimension.
            A_chunk = A[:, k:k+batch_size]    # Shape: (n_rows, batch_size)
            B_chunk = B[:, k:k+batch_size]      # Shape: (m, batch_size)
            # Expand dimensions to broadcast:
            # A_chunk -> (n_rows, 1, batch_size)
            # B_chunk -> (1, m, batch_size)
            # Then compute the product over the chunk dimension.
            term = torch.prod(1 - A_chunk.unsqueeze(1) * B_chunk.unsqueeze(0), dim=-1)
            result = result * term
        return result

    def _compute_blocked_product(self, A, B):
        A = A.unsqueeze(1) # Shape (n, 1, n)
        B = B.unsqueeze(0)  # Shape (1, n, n)
        result = torch.prod(1 - (A * B), dim=2)
        return result


    def __minimal_supports(self, edge_weights_strict, batch_size):
        """
        Compute minimal supports.
    
        When use_blockers is True, compute:
        blocked_supports[i,j] = ∏ₖ (1 - edge_weights_strict[i, k] * edge_weights_strict[j, k])
        Otherwise, return a tensor of ones.
        """
        if self.use_blockers:
            A = edge_weights_strict             
            B = edge_weights_strict.T           
            if batch_size is not None:
                blocked_supports = self._compute_blocked_product_batched(A, B, batch_size)
            else:
                blocked_supports = self._compute_blocked_product(A, B)
        else:
            blocked_supports = torch.ones_like(edge_weights_strict)
        return blocked_supports


    def __minimal_attacks(self, y_train, edge_weights_strict, attacks, indexes, batch_size):
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
            same_labels = torch.all(y_train[i_indices] == y_train[j_indices], dim=-1)
        
            A = torch.where(same_labels, edge_weights_strict, 0)
            B = attacks.T
            if batch_size is not None:
                blocked_attacks = self._compute_blocked_product_batched(A, B, batch_size)
            else:
                blocked_attacks = self._compute_blocked_product(A, B)
        else:
            blocked_attacks = torch.ones_like(edge_weights_strict)
        return blocked_attacks



    def _add_default_cases(self, X_train, y_train, X_default, y_default):
        default_index_start = len(X_train)
        X_train = torch.cat((X_train, X_default), dim=0)
        y_train = torch.cat((y_train, y_default), dim=0)
        default_index_end = len(X_train)
        default_indexes = torch.arange(default_index_start, default_index_end)

        return X_train, y_train, default_indexes

    def forward(self, new_cases: torch.Tensor, post_process_func = lambda x: x, return_all_strengths = False):
        """
            Computes the final strenghts of the EW-QAF for each new_case input

            Parameters
            ----------
            new_cases : torch.Tensor
                Input newcase argument characterisations as a tensor. 
                Shape (N, x1, ..., xn) where N is the number of new cases 
                arguments and (x1, ..., xn) is the shape of each argument.

           post_process_func: Callable
                Optionally apply a post process function to the adjacency matrix
                before computing the gradual semantics

            Returns
            -------
            final_strengths : torch.Tensor
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
            raise(Exception("Ensure the model has been fit first."))
        batch_size = new_cases.shape[0]

        base_scores = self.compute_base_scores(self.X_train)  # (n)
        base_scores = self.__batch_base_scores(base_scores, batch_size)
        base_scores = self.__new_case_influence(self.X_train, base_scores, new_cases)
        A = post_process_func(self.A)
        assert(A.shape == self.A.shape)
        strengths = self.gradual_semantics(A, base_scores)

        # TODO: Check if this is necessary:
        # final_strengths = strengths[-1].squeeze()
        final_strengths = strengths.squeeze()
        if final_strengths.dim() == 1:
            final_strengths = final_strengths.unsqueeze(0)

        if return_all_strengths:
            return final_strengths

        return final_strengths[:, self.default_indexes]


    def __batch_base_scores(self, base_scores: torch.Tensor, batch_size: int) -> torch.Tensor:
        base_scores = torch.tile(base_scores.unsqueeze(
            dim=0), (batch_size, 1))  # (B x n)
        base_scores = base_scores.unsqueeze(2)  # (B x n x 1)
        return base_scores
    

    def __new_case_influence(self, X_train, base_scores, new_cases):
        new_cases_base_scores = self.compute_base_scores(
            new_cases).unsqueeze(-1).unsqueeze(-1)  # (B x 1 x 1)
        
        new_cases_attacks_adjacency = self.irrelevance_edge_weights(new_cases, X_train) # B x n
        new_cases_attacks_adjacency = -new_cases_attacks_adjacency

        # B x 1 x n
        new_cases_attacks_adjacency = new_cases_attacks_adjacency.unsqueeze(-2)

        # We compute the aggregations *only* for the attacks by the new cases.
        # As new cases are unattacked, this can be computed in a single pass of 
        # aggregation/influence function
        # b x n x 1
        aggregations = self.gradual_semantics.aggregation_func(
            new_cases_attacks_adjacency, new_cases_base_scores)
        strengths = self.gradual_semantics.influence_func(base_scores, aggregations)

        return strengths
    
    
    def show_graph(self, post_process_func = lambda x: x, logger = lambda x: x, 
                               prevent_show = False, positions = None, threshold = 0.):
        """
            Outputs a networkx graph of the casebase

            Notes
            -----
            This function will raise an exception if the fit function has not
            been previously called.

        """

        if self.A is None:
            raise(Exception("Ensure the model has been fit first."))

        A = post_process_func(self.A).detach().cpu().numpy()
        y_train = self.y_train.cpu().detach().numpy()
        if self.y_train.shape[-1] > 1:
            y_train = np.argmax(y_train, axis=1)
        else:
            y_train = y_train.squeeze()
        default_indexes = self.default_indexes.cpu().detach().numpy()
        
        A = np.where(np.abs(A) > threshold, A , 0)

        gr = nx.from_numpy_array(A, create_using=nx.DiGraph)
        if not positions:
            pos = nx.nx_agraph.graphviz_layout(gr, prog='dot',
                                        args='-Gsplines=true -Gnodesep=2')
        else:
            default_pos = lambda idx: (idx-default_indexes[0] + 1) * (1/(len(default_indexes) + 2)) * len(gr.nodes)
            positions.update({idx: (default_pos(idx), -1) for idx in default_indexes})
            pos = positions

        unique_labels = np.unique(y_train)
        colormap = plt.get_cmap('rainbow', len(unique_labels))
        label_to_color = {label: colormap(i)
                          for i, label in enumerate(unique_labels)}
        node_colors = [label_to_color[y_train[node]] for node in list(gr.nodes)]

        assert (all([default_index in list(gr.nodes)
                for default_index in default_indexes]))

        labels = {x: x for x in list(gr.nodes)}
        for i, default_index in enumerate(default_indexes):
            labels.update({default_index: "Default"})


        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(1, 1, 1)
        for i, label in enumerate(unique_labels):
            ax.plot([0], [0], color=colormap(i), label=f'Class: {label}')


        # For positive weights (0 to 1): white to green.
        top = cm.get_cmap('Greens', 128)(np.linspace(0, 1, 128))
        # For negative weights (–1 to 0): red to white.
        bottom = cm.get_cmap('Reds_r', 128)(np.linspace(0, 1, 128))
        white = np.array([[1, 1, 1, 1]])  # pure white

        newcolors = np.vstack((bottom, white, top))
        ew_colormap = ListedColormap(newcolors, name='WhiteRedGreen')

        _,weights = zip(*nx.get_edge_attributes(gr,'weight').items())
        norm = Normalize(vmin=-1, vmax=1)

        edge_colors = [ew_colormap(norm(w)) for w in weights]

        nx.draw(gr, pos, labels=labels,
                arrowstyle='-|>', arrows=True, node_color=node_colors,  arrowsize=20,
                node_size=100, font_size=5, width=0.4, edge_color=edge_colors, edge_cmap=ew_colormap)

        plt.legend()
        logger(f)
        if not prevent_show:
            plt.show()

    def show_matrix(self, post_process_func = lambda x: x, logger = lambda x: x, prevent_show = False):
        """

            Outputs an image of the adjacency matrix of the casebase

            Notes
            -----
            This function will raise an exception if the fit function has not
            been previously called.

        """

        if self.A is None:
            raise(Exception("Ensure the model has been fit first."))

        f = plt.figure(figsize=(10, 10))
        A = post_process_func(self.A).detach().cpu().numpy()
        plt.imshow(A, cmap='cividis', interpolation='nearest')
        ax = plt.gca()
        n = len(self.X_train)
        # Major ticks
        ax.set_xticks(np.arange(0, n, 1))
        ax.set_yticks(np.arange(0, n, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(0, n, 1))
        ax.set_yticklabels(np.arange(0, n, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)

        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.2)
        plt.colorbar(label='Value')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        plt.title('Edges')
        logger(f)
        if not prevent_show:
            plt.show()
    
    

    
