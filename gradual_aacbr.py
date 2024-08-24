import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from semantics.gradual_semantics import GradualSemantics
from irrelevance_edge_weights.compute_irrelevance import ComputeIrrelevance
from casebase_edge_weights.compute_partial_order import ComputePartialOrder
from base_scores.compute_base_scores import ComputeBaseScores 
from typing import Callable


class GradualAACBR(torch.nn.Module):

    def __init__(self, gradual_semantics: GradualSemantics, compute_base_score: ComputeBaseScores, 
                       irrelevance_edge_weights: ComputeIrrelevance, casebase_edge_weights: ComputePartialOrder):
        super(GradualAACBR, self).__init__()

        self.gradual_semantics = gradual_semantics
        self.compute_base_scores = compute_base_score
        self.casebase_edge_weights = casebase_edge_weights
        self.irrelevance_edge_weights = irrelevance_edge_weights
        self.A = None
    
    def casebase_edge_weights_strict(self, attacker, target):
        return self.casebase_edge_weights(attacker, target) * (1 - self.casebase_edge_weights(target, attacker))

    def casebase_edge_weights_equal(self, attacker, target):
        return self.casebase_edge_weights(attacker, target) * self.casebase_edge_weights(target, attacker)
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, X_default: torch.Tensor, y_default: torch.Tensor, 
            use_symmetric_attacks = True, defaults_not_attack = True):
        

        if (X_train is None or y_train is None or len(X_train) != len(y_train)):
            raise(Exception("Length of X_train must match length of y_train."))

        if (X_default is None or y_default is None or len(X_default) != len(y_default)):
            raise(Exception("Length of X_default must match length of y_default"))

        self.A = None
        X_train, y_train, default_indexes = self.__add_default_cases(X_train, y_train, X_default, y_default)

        train_size = len(X_train)
        indexes = torch.arange(train_size)
        index_prod = torch.cartesian_prod(indexes, indexes, indexes)

        X_attackers, y_attackers = X_train[index_prod[:, 0]], y_train[index_prod[:, 0]]
        X_targets, y_targets     = X_train[index_prod[:, 1]], y_train[index_prod[:, 1]]
        X_blockers, y_blockers   = X_train[index_prod[:, 2]], y_train[index_prod[:, 2]]

        attackers_default_mask = torch.isin(index_prod[:, 0], default_indexes)
        blockers_default_mask  = torch.isin(index_prod[:, 2], default_indexes)

        attack_targets = self.__potential_attackers(X_attackers, X_targets, y_attackers, 
                                                    y_targets, defaults_not_attack, attackers_default_mask) 
        
        blocked_attacks = self.__minimal_attacks(X_attackers, X_blockers, X_targets, y_blockers, 
                                                 y_attackers, defaults_not_attack, blockers_default_mask)

        if use_symmetric_attacks: 
            symmetric_attacks = self.__symmetric_attacks(X_attackers, X_targets, y_attackers, 
                                                        y_targets, defaults_not_attack, attackers_default_mask)
        else:
            symmetric_attacks = torch.zeros_like(attack_targets)

        symmetric_attacks = symmetric_attacks.reshape((train_size, train_size, train_size)).prod(axis=-1)

        # Combine potential attacks and blocked attacks
        attacks = torch.mul(attack_targets, blocked_attacks)

        # Convert to adjacency matrix
        all_attacks = attacks.reshape((train_size, train_size, train_size))
        A = all_attacks.prod(axis=-1) + symmetric_attacks
        self.A = (-A) 
        self.X_train = X_train
        self.y_train = y_train
        self.default_indexes = default_indexes

    def fit_no_blockers(self, X_train: torch.Tensor, y_train: torch.Tensor, X_default: torch.Tensor, y_default: torch.Tensor, 
            use_symmetric_attacks = True, defaults_not_attack = True):
        

        if (X_train is None or y_train is None or len(X_train) != len(y_train)):
            raise(Exception("Length of X_train must match length of y_train."))

        if (X_default is None or y_default is None or len(X_default) != len(y_default)):
            raise(Exception("Length of X_default must match length of y_default"))

        self.A = None
        X_train, y_train, default_indexes = self.__add_default_cases(X_train, y_train, X_default, y_default)

        train_size = len(X_train)
        indexes = torch.arange(train_size)
        index_prod = torch.cartesian_prod(indexes, indexes)

        X_attackers, y_attackers = X_train[index_prod[:, 0]], y_train[index_prod[:, 0]]
        X_targets, y_targets     = X_train[index_prod[:, 1]], y_train[index_prod[:, 1]]

        attackers_default_mask = torch.isin(index_prod[:, 0], default_indexes)

        attack_targets = self.__potential_attackers(X_attackers, X_targets, y_attackers, 
                                                    y_targets, defaults_not_attack, attackers_default_mask) 
        
        if use_symmetric_attacks: 
            symmetric_attacks = self.__symmetric_attacks(X_attackers, X_targets, y_attackers, 
                                                        y_targets, defaults_not_attack, attackers_default_mask)
        else:
            symmetric_attacks = torch.zeros_like(attack_targets)

        symmetric_attacks = symmetric_attacks.reshape((train_size, train_size))

        # Combine potential attacks and blocked attacks

        # Convert to adjacency matrix
        A = attack_targets.reshape((train_size, train_size)) + symmetric_attacks
        self.A = (-A) 
        self.X_train = X_train
        self.y_train = y_train
        self.default_indexes = default_indexes


    def __symmetric_attacks(self, X_attackers, X_targets, y_attackers, 
                              y_targets, defaults_not_attack, attackers_default_mask):

        attack_targets = self.casebase_edge_weights_equal(X_attackers, X_targets)

        assert(attack_targets.ndim == 1)
        if len(y_attackers.shape) == 2:
            differing_labels = torch.any(y_attackers != y_targets, dim=-1)
        else:
            differing_labels = y_attackers != y_targets
            print(differing_labels)

        attack_targets = torch.where(differing_labels, attack_targets, 0)

        if defaults_not_attack:
            attack_targets = torch.where(attackers_default_mask, 0, attack_targets)
        
        return attack_targets

    def __potential_attackers(self, X_attackers, X_targets, y_attackers, 
                              y_targets, defaults_not_attack, attackers_default_mask):

        attack_targets = self.casebase_edge_weights_strict(X_attackers, X_targets) 

        assert(attack_targets.ndim == 1)
        if len(y_attackers.shape) == 2:
            differing_labels = torch.any(y_attackers != y_targets, dim=-1)
        else:
            differing_labels = y_attackers != y_targets
            print(differing_labels)

        attack_targets = torch.where(differing_labels, attack_targets, 0)

        if defaults_not_attack:
            attack_targets = torch.where(attackers_default_mask, 0, attack_targets)
        
        return attack_targets
    
    def __minimal_attacks(self, X_attackers, X_blockers, X_targets, y_blockers, 
                          y_attackers, defaults_not_attack, blockers_default_mask):
        blocked_attacks = torch.mul(self.casebase_edge_weights_strict(X_attackers, X_blockers),
                                    self.casebase_edge_weights_strict(X_blockers, X_targets))

        if len(y_attackers.shape) == 2:
            same_labels = torch.any(y_blockers == y_attackers, dim=-1)
        else:
            same_labels = y_blockers == y_attackers

        blocked_attacks = torch.where(same_labels, blocked_attacks, 0)

        if defaults_not_attack:
            blocked_attacks = torch.where(blockers_default_mask, 0, blocked_attacks)
        
        return 1 - blocked_attacks


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

        # B x 1 x n
        new_cases_attacks_adjacency = new_cases_attacks_adjacency.unsqueeze(-2)

        # We compute the aggregations *only* for the attacks by the new cases.
        # As new cases are unattacked, this can be computed in a single pass of 
        # aggregation/influence function
        aggregations = self.gradual_semantics.aggregation_func(
            new_cases_attacks_adjacency, new_cases_base_scores)
        strengths = self.gradual_semantics.influence_func(base_scores, aggregations)

        return strengths

    def forward(self, new_cases: torch.Tensor, return_all_strenghts = False):

        if self.A is None:
            raise(Exception("Ensure the model has been fit first."))

        batch_size = new_cases.shape[0]

        base_scores = self.compute_base_scores(self.X_train)  # (n)
        base_scores = self.__batch_base_scores(base_scores, batch_size)
        base_scores = self.__new_case_influence(self.X_train, base_scores, new_cases)
        strengths = self.gradual_semantics(self.A, base_scores)

        # TODO: Check if this is necessary:
        final_strengths = strengths[-1].squeeze()
        if final_strengths.dim() == 1:
            final_strengths = final_strengths.unsqueeze(0)

        if return_all_strenghts:
            return final_strengths
        else:
            return final_strengths[:, self.default_indexes]

    def show_graph_with_labels(self):
        A = self.A.detach().cpu().numpy()
        y_train = np.argmax(self.y_train.cpu().detach().numpy(), axis=1)
        default_indexes = self.default_indexes.cpu().detach().numpy()

        # rows, cols = np.where(A != 0)
        # edges = zip(rows.tolist(), cols.tolist())

        # gr = nx.DiGraph()
        # gr.add_edges_from(edges)

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
        plt.show()

    def show_matrix(self):
        plt.figure(figsize=(10, 10))
        A = self.A.detach().cpu().numpy()
        plt.imshow(A, cmap='cividis', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        plt.title('Edges')
        plt.show()
    
    def plot_base_score_parameters(self):
        self.compute_base_scores.plot_parameters()

    def plot_casebase_edge_weights_parameters(self):
        self.casebase_edge_weights.plot_parameters()

    def plot_irrelevance_edge_weights_parameters(self):
        self.irrelevance_edge_weights.plot_parameters()