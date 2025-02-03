import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from deeparguing.semantics.gradual_semantics import GradualSemantics
from deeparguing.irrelevance_edge_weights.compute_irrelevance import ComputeIrrelevance
from deeparguing.casebase_edge_weights.compute_partial_order import ComputePartialOrder
from deeparguing.base_scores.compute_base_scores import ComputeBaseScores 


class GradualAACBR(torch.nn.Module):

    def __init__(self, gradual_semantics: GradualSemantics, compute_base_score: ComputeBaseScores, 
                       irrelevance_edge_weights: ComputeIrrelevance, casebase_edge_weights: ComputePartialOrder):
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
        """
        super().__init__() 

        self.gradual_semantics = gradual_semantics
        self.compute_base_scores = compute_base_score
        self.casebase_edge_weights = casebase_edge_weights
        self.irrelevance_edge_weights = irrelevance_edge_weights
        self.A = None
    
    def __casebase_edge_weights_strict(self, attacker, target, train_size):
        edge_weights = self.casebase_edge_weights(attacker, target).reshape((train_size, train_size))
        return edge_weights * (1 - (edge_weights.T))

    def __casebase_edge_weights_equal(self, attacker, target, train_size):
        edge_weights = self.casebase_edge_weights(attacker, target).reshape((train_size, train_size))
        return edge_weights * edge_weights.T


    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, X_default: torch.Tensor, y_default: torch.Tensor, 
            use_symmetric_attacks = True, defaults_not_attack = True, use_blockers = True, approximate_blockers = False, tau=None):

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
            use_symmetric_attack : bool, default True
                When true, symmetric attacks between cases of the same 
                characterisation are included
            defaults_not_attack : bool, default True
                When true, the default arguments will not be able to attack any
                case in the casebase
            use_blockers : bool, default True
                When true, the model will optimise attacks of minimal between cases
                of minimal difference
            

            Notes
            -----
            The fit algorithm uses torch vector and batching operations to
            improve efficiency. See fit_slow for a direct implementation of the
            definition.

            Fit must be called before calling any subsequent function that uses
            the adjacency matrix self.A

        """

        device = X_train.device

        if (X_train is None or y_train is None or len(X_train) != len(y_train)):
            raise(Exception(f"Length of X_train must match length of y_train. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"))

        if (X_default is None or y_default is None or len(X_default) != len(y_default)):
            raise(Exception(f"Length of X_default must match length of y_default. X_default shape: {X_default.shape}, y_default shape: {y_default.shape}"))

        self.A = None
        X_train, y_train, default_indexes = self.__add_default_cases(X_train, y_train, X_default, y_default)

        train_size = len(X_train)
        indexes = torch.arange(train_size)
        
        index_prod = torch.cartesian_prod(indexes, indexes)

        idx_attackers = index_prod[:, 0]
        idx_targets = index_prod[:, 1]

        attackers_default_mask = torch.isin(idx_attackers, default_indexes).reshape((train_size, train_size)).to(device)

        X_attackers, y_attackers = X_train[idx_attackers], y_train[idx_attackers]
        X_targets, y_targets = X_train[idx_targets], y_train[idx_targets]

        edge_weights_strict = self.__casebase_edge_weights_strict(X_attackers, X_targets, train_size)

        if defaults_not_attack:
            edge_weights_strict = torch.where(attackers_default_mask, 0, edge_weights_strict)

        attacks, differing_labels = self.__potential_attacks(edge_weights_strict, y_attackers, y_targets, train_size)

        if approximate_blockers and use_blockers:
            blocked_attacks = torch.ones_like(attacks)
            attacks = self.masked_softmax(attacks, attacks != 0, dim=0)
        else:
            blocked_attacks = self.__minimal_attacks(use_blockers, y_train, edge_weights_strict, attacks, indexes)


        symmetric_attacks = self.__symmetric_attacks(use_symmetric_attacks, X_attackers,  X_targets, 
                               defaults_not_attack, attackers_default_mask, train_size, differing_labels)

        self.A = (torch.mul(attacks, blocked_attacks) + symmetric_attacks) 
        if tau:
            self.A = self.masked_softmax(self.A/tau, self.A != 0, dim=0)
        self.A = -self.A
        self.X_train = X_train
        self.y_train = y_train
        self.default_indexes = default_indexes

    def masked_softmax(self, vec, mask, dim=0):
        # Masked softmax from https://discuss.pytorch.org/t/apply-mask-softmax/14212/12#:~:text=Jun%202018-,I,-had%20to%20implement
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros=(masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps/masked_sums


    def __symmetric_attacks(self, use_symmetric_attacks, X_attackers,  X_targets, 
                               defaults_not_attack, attackers_default_mask, train_size, differing_labels):

        if use_symmetric_attacks: 
            symmetric_attacks = self.__casebase_edge_weights_equal(X_attackers, X_targets, train_size)
            symmetric_attacks = torch.where(differing_labels, symmetric_attacks, 0)
            if defaults_not_attack:
                symmetric_attacks = torch.where(attackers_default_mask, 0, symmetric_attacks)
            
        else:
            symmetric_attacks = torch.zeros((train_size, train_size)).to(X_attackers.device)
        
        return symmetric_attacks 

    def __potential_attacks(self, edge_weights_strict,  y_attackers, y_targets,  train_size):



        if len(y_attackers.shape) == 2:
            differing_labels = torch.any(y_attackers != y_targets, dim=-1)
        else:
            differing_labels = y_attackers != y_targets
        
        differing_labels = torch.reshape(differing_labels, (train_size, train_size))

        attacks = torch.where(differing_labels, edge_weights_strict, 0)

        return attacks, differing_labels

    
    def __minimal_attacks(self, use_blockers, y_train, edge_weights_strict, attacks, indexes):

        if use_blockers:
            i_indices = indexes.unsqueeze(1)  # Column vector (rows x 1)
            j_indices = indexes.unsqueeze(0)  # Row vector (1 x cols)
            if len(y_train.shape) == 2:
                same_labels = torch.all(y_train[i_indices] == y_train[j_indices], dim=-1)
            else:
                same_labels = (y_train[i_indices] == y_train[j_indices])
            
            order_with_same_labels = torch.where(same_labels, edge_weights_strict, 0).unsqueeze(1) # Shape (n, 1, n)
            attacks_expanded = attacks.T.unsqueeze(0)  # Shape (1, n, n)
            blocked_attacks = torch.prod(1 - (order_with_same_labels * attacks_expanded), dim=2)
        else:
            blocked_attacks = torch.ones_like(edge_weights_strict)
        
        return blocked_attacks


    def __add_default_cases(self, X_train, y_train, X_default, y_default):
        default_index_start = len(X_train)
        X_train = torch.cat((X_train, X_default), dim=0)
        y_train = torch.cat((y_train, y_default), dim=0)
        default_index_end = len(X_train)
        default_indexes = torch.arange(default_index_start, default_index_end)

        return X_train, y_train, default_indexes


    def __batch_base_scores(self, base_scores: torch.Tensor, batch_size: int) -> torch.Tensor:
        base_scores = torch.tile(base_scores.unsqueeze(
            dim=0), (batch_size, 1))  # (B x n)
        base_scores = base_scores.unsqueeze(2)  # (B x n x 1)
        return base_scores
    

    def __new_case_influence(self, X_train, base_scores, new_cases):
        new_cases_base_scores = self.compute_base_scores(
            new_cases).unsqueeze(-1).unsqueeze(-1)  # (B x 1)

        new_cases_attacks_adjacency = self.irrelevance_edge_weights(new_cases, X_train) # B x n
        new_cases_attacks_adjacency = -new_cases_attacks_adjacency

        # B x 1 x n
        new_cases_attacks_adjacency = new_cases_attacks_adjacency.unsqueeze(-2)

        # We compute the aggregations *only* for the attacks by the new cases.
        # As new cases are unattacked, this can be computed in a single pass of 
        # aggregation/influence function
        aggregations = self.gradual_semantics.aggregation_func(
            new_cases_attacks_adjacency, new_cases_base_scores)
        strengths = self.gradual_semantics.influence_func(base_scores, aggregations)

        return strengths

    def forward(self, new_cases: torch.Tensor, return_all_strenghts = False, post_process_func = lambda x: x):
        """
            Computes the final strenghts of the EW-QAF for each new_case input

            Parameters
            ----------
            new_cases : torch.Tensor
                Input newcase argument characterisations as a tensor. 
                Shape (N, x1, ..., xn) where N is the number of new cases 
                arguments and (x1, ..., xn) is the shape of each argument.

            return_all_strenghts : bool, default False
                When false, the final strenghts of only the default cases are
                returned.
                When true, the final strenght for all casebase 
                arguments will be returned.
                

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
        final_strengths = strengths[-1].squeeze()
        if final_strengths.dim() == 1:
            final_strengths = final_strengths.unsqueeze(0)

        if return_all_strenghts:
            return final_strengths
        else:
            return final_strengths[:, self.default_indexes]
    
    def show_graph_with_labels(self, post_process_func = lambda x: x, logger = lambda x: x, prevent_show = False):
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
        y_train = np.argmax(self.y_train.cpu().detach().numpy(), axis=1)
        default_indexes = self.default_indexes.cpu().detach().numpy()


        gr = nx.from_numpy_array(A, create_using=nx.DiGraph)
        pos = nx.nx_agraph.graphviz_layout(gr, prog='dot',
                                           args='-Gsplines=true -Gnodesep=2')

        unique_labels = np.unique(y_train)
        colormap = plt.get_cmap('gist_rainbow', len(unique_labels))
        label_to_color = {label: colormap(i)
                          for i, label in enumerate(unique_labels)}
        node_colors = [label_to_color[y_train[node]] for node in list(gr.nodes)]

        assert (all([default_index in list(gr.nodes)
                for default_index in default_indexes]))

        labels = {x: x for x in list(gr.nodes)}
        for i, default_index in enumerate(default_indexes):
            labels.update({default_index: "Default"})
            # pos.update({default_index: (0 + i * 100, 0)})

        # for k, v in pos.items():
        #     pos.update({k: (v[0] * random.randint(-100, 100), v[1])})

        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(1, 1, 1)
        for i, label in enumerate(unique_labels):
            ax.plot([0], [0], color=colormap(i), label=f'Class: {label}')

        nx.draw(gr, pos, labels=labels,
                arrowstyle='-|>', arrows=True, node_color=node_colors,
                node_size=100, font_size=5, width=0.4)
                
        labels = nx.get_edge_attributes(gr, 'weight')
        rounded_labels = {edge: round(weight, 5) for edge, weight in labels.items()}

        # Draw the edge labels with rounded weights
        nx.draw_networkx_edge_labels(gr, pos, edge_labels=rounded_labels, font_size=5)

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
    
    def plot_base_score_parameters(self):
        self.compute_base_scores.plot_parameters()

    def plot_casebase_edge_weights_parameters(self):
        self.casebase_edge_weights.plot_parameters()

    def plot_irrelevance_edge_weights_parameters(self):
        self.irrelevance_edge_weights.plot_parameters()
    
    ############################################################################
    # The following contains an implementation of fit that makes limited use of 
    #  broadcasting, vectorisation or matrix operations and is used for sanity
    # checking results    
    ############################################################################
    def __casebase_edge_weights_strict_old(self, attacker, target):
        #This is ineffecient because of repeated calls to casebase_edge_weights
        return self.casebase_edge_weights(attacker, target) * (1 - self.casebase_edge_weights(target, attacker))

    def __casebase_edge_weights_equal_old(self, attacker, target):
        #This is ineffecient because of repeated calls to casebase_edge_weights
        return self.casebase_edge_weights(attacker, target) * self.casebase_edge_weights(target, attacker)

    def slow_fit(self, X_train: torch.Tensor, y_train: torch.Tensor, X_default: torch.Tensor, y_default: torch.Tensor, 
            use_symmetric_attacks = True, defaults_not_attack = True, use_blockers = True):

        if (X_train is None or y_train is None or len(X_train) != len(y_train)):
            raise(Exception(f"Length of X_train must match length of y_train. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"))

        if (X_default is None or y_default is None or len(X_default) != len(y_default)):
            raise(Exception(f"Length of X_default must match length of y_default. X_default shape: {X_default.shape}, y_default shape: {y_default.shape}"))

        self.A = None
        X_train, y_train, default_indexes = self.__add_default_cases(X_train, y_train, X_default, y_default)

        train_size = len(X_train)
        indexes = torch.arange(train_size)
        
        index_prod = torch.cartesian_prod(indexes, indexes)

        idx_attackers = index_prod[:, 0]
        idx_targets = index_prod[:, 1]


        X_attackers = X_train[idx_attackers]
        X_targets = X_train[idx_targets]
        
        edge_weights_strict = self.__casebase_edge_weights_strict_old(X_attackers, X_targets).reshape((train_size, train_size))

        if use_symmetric_attacks:
            edge_weights_equal = self.__casebase_edge_weights_equal_old(X_attackers, X_targets).reshape((train_size, train_size))
        else:
            edge_weights_equal = torch.zeros_like(edge_weights_strict)

    
        result = torch.zeros_like(edge_weights_strict)

        for attacker_index in range(train_size):

            if defaults_not_attack and attacker_index in default_indexes:
                continue

            for target_index in range(train_size):

                if torch.all(y_train[attacker_index] == y_train[target_index], dim=-1):
                    continue

                blocker_value = 1
                if use_blockers:

                    for blocker_index in range(train_size):

                        if torch.all(y_train[attacker_index] != y_train[blocker_index], dim=-1) or (defaults_not_attack and blocker_index in default_indexes):
                            continue

                        blocker_value = blocker_value * (1 - (edge_weights_strict[attacker_index, blocker_index] * edge_weights_strict[blocker_index, target_index]))


                result[attacker_index, target_index] = edge_weights_strict[attacker_index, target_index] * blocker_value + edge_weights_equal[attacker_index, target_index]        


            


        self.A = -result
        self.X_train = X_train
        self.y_train = y_train
        self.default_indexes = default_indexes


    