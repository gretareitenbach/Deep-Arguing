import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class LearnedPartialOrder(torch.nn.Module):
    def __init__(self, no_features):
        super(LearnedPartialOrder, self).__init__()
        self.W = torch.nn.Parameter(torch.Tensor(no_features))
        torch.nn.init.normal_(self.W)
    
    def forward(self, a, b, old=False):
        if b.ndim == 1:
            b = b.unsqueeze(0)

        # if old:
        #     return torch.matmul(a, self.W) > torch.matmul(b, self.W)

        a_score = torch.matmul(a, self.W)
        b_score = torch.matmul(b, self.W)

        return torch.sigmoid((a_score - b_score) * 100)
    
    def plot_parameters(self):
        weights = self.W.detach().numpy()
        plt.figure(figsize=(20, 5))
        plt.bar(range(len(weights)), weights)
        for i, value in enumerate(weights):
            plt.text(i, value + (0.1 * (-1 if value <= 0 else 1)), str(round(value, 3)), ha='center', fontsize=6)
        plt.xlabel('Features')
        plt.ylabel('Weights')
        plt.title('Feature Attribution Weights')
        plt.show()
    

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class AACBRTorch(torch.nn.Module):

    def __init__(self, X_train, y_train, comparison_func, default_case, default_outcomes,
                 use_symmetric_attacks=True, casebase_indices=None) -> None:

        super(AACBRTorch, self).__init__()
        assert (len(X_train) == len(y_train))

        self.comparison_func = comparison_func
        self.X_train = X_train
        self.y_train = y_train
        self.default_case = default_case
        self.default_outcomes = default_outcomes
        self.use_symmetric_attacks = torch.tensor(use_symmetric_attacks)
        self.casebase_indices = casebase_indices

        self.add_default_cases(default_case, default_outcomes)
        self.build_af_parallel()
    
    def set_casebase_indices(self, casebase_indices):
        self.casebase_indices = casebase_indices

    def add_default_cases(self, default_case, default_outcomes):

        pre = len(self.X_train)
        self.X_train = torch.cat(
            (self.X_train, default_case.repeat(len(default_outcomes), 1)), dim=0)
        self.y_train = torch.cat((self.y_train, default_outcomes), dim=0)

        post = len(self.X_train)
        self.default_indexes = torch.tensor(list(range(pre, post)))
        # self.outcome_map = {outcome: self.default_indexes[i] for i, outcome in enumerate(self.default_outcomes)}

    def get_indices(self):
        if self.casebase_indices is None:
            return torch.arange(len(self.X_train))
        else:
            return torch.cat((self.casebase_indices, self.default_indexes))

    def build_af_parallel(self):

        # TODO: Could find a way to make use of Topological ordering to speed this up further
        train_size = len(self.X_train)
        all_indices = torch.arange(train_size)
        indexes = torch.cartesian_prod(
            all_indices, all_indices,  all_indices
        )

        attackers, attackers_labels = self.X_train[indexes[:, 0]
                                                   ], self.y_train[indexes[:, 0]]
        targets, targets_labels = self.X_train[indexes[:, 1]
                                               ], self.y_train[indexes[:, 1]]
        blockers, blockers_labels = self.X_train[indexes[:, 2]
                                                 ], self.y_train[indexes[:, 2]]

        attackers_mask = torch.isin(indexes[:, 0], self.get_indices())
        targets_mask = torch.isin(indexes[:, 1], self.get_indices())
        blockers_mask = torch.logical_not(
            torch.isin(indexes[:, 2], self.get_indices()))
        # index_mask = True


        round_ste = RoundSTE.apply

        # Check potential attackers:

        # TODO: NOT SURE ROUND IS NECESSARY?
        # attack_argets = round_ste(self.comparison_func(attackers, targets))

        attack_targets = self.comparison_func(attackers, targets)
        differing_labels = torch.abs(attackers_labels - targets_labels)
        differing_labels = torch.clamp(differing_labels, 0, 1)
        attack_targets = torch.mul(differing_labels, attack_targets)

        

        # Check blocked attackers:

        # TODO: NOT SURE ROUND IS NECESSARY?
        # blocked_attacks = torch.mul(round_ste(self.comparison_func(attackers, blockers)), round_ste(self.comparison_func(blockers, targets)))

        blocked_attacks = torch.mul(self.comparison_func(attackers, blockers), self.comparison_func(blockers, targets))
        differing_labels = torch.abs(blockers_labels - attackers_labels)
        differing_labels = 1 - torch.clamp(differing_labels, 0, 1)
        blocked_attacks = torch.mul(differing_labels, blocked_attacks)
        
        # Handle Symmetric attacks
        # TODO: NEED TO HANDLE SYMMETRIC ATTACKS

            # g = torch.where(symmetric_attacks, 1, 0)
            # print(g)

            # if self.use_symmetric_attacks:
            #     h = torch.all(attackers == targets, axis=1)
            #     h = torch.where(h, 1, 0)
            #     print(h)

            #     g = torch.abs(attackers_labels - targets_labels)
            #     g = torch.clamp(g, 0, 1)
            #     g = torch.mul(g, h)
            # else:
            #     g = torch.ones_like(a)


        # Combine potential attacks and blocked attacks 
        attacks = torch.mul(attack_targets, (1 - blocked_attacks))
        attacks = torch.where(blockers_mask, 1, attacks)
        attacks = torch.where(attackers_mask, attacks, 0)
        attacks = torch.where(targets_mask, attacks, 0)


        # Resolve to adjacency matrix
        all_attacks = attacks.reshape((train_size, train_size, train_size))
        A = all_attacks.prod(axis=-1)
        # TODO: NEED TO ENSURE DEFAULTS DO NOT ATTACK ANYTHING USING A TORCH.WHERE
            # attacks[self.default_indexes, :] = False
            # self.A = torch.where(attacks, -1, 0)
        self.A = -A



    def is_concise(self, attacker, target, attacker_outcome):
        return not any([
            self.y_train[k] == attacker_outcome and
            self.comparison_func(attacker, self.X_train[k])[0] and self.comparison_func(self.X_train[k], target)[0] for k in self.get_indices()])

    def show_graph_with_labels(self):
        A = self.A.detach().cpu().numpy()
        rows, cols = np.where(A != 0)
        edges = zip(rows.tolist(), cols.tolist())

        gr = nx.DiGraph()
        gr.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(gr, prog='dot',
                                           args='-Gsplines=true -Gnodesep=2')

        unique_labels = np.unique(self.y_train)
        colormap = plt.get_cmap('gist_rainbow', len(unique_labels))
        label_to_color = {label: colormap(i)
                          for i, label in enumerate(unique_labels)}
        node_colors = [label_to_color[self.y_train[node]]
                       for node in list(gr.nodes)]

        assert (all([default_index in list(gr.nodes)
                for default_index in self.default_indexes]))

        labels = {x: x for x in list(gr.nodes)}
        for i, default_index in enumerate(self.default_indexes):
            labels.update({default_index: "Default"})
            # pos.update({default_index: (0 + i * 100, 0)})

        # for k, v in pos.items():
        #     pos.update({k: (v[0] * random.randint(-100, 100), v[1])})

        f = plt.figure(figsize=(20, 20))
        ax = f.add_subplot(1, 1, 1)
        for i, label in enumerate(unique_labels):
            ax.plot([0], [0], color=colormap(i), label=f'Class: {label}')

        nx.draw(gr, pos, labels=labels,
                arrowstyle='-|>', arrows=True, node_color=node_colors,
                node_size=100, font_size=5, width=0.4)

        plt.legend()
        plt.show()

    def show_matrix(self):
        plt.figure(figsize=(10, 10))
        A = self.A.detach().cpu().numpy()
        plt.imshow(A, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        plt.title('Edges')
        plt.show()

    def get_new_case_attacks_mask(self, new_cases):

        #TODO: Investigate if gradient computation is going wrong here:

        if new_cases.ndim == 1:
            new_cases = new_cases.unsqueeze(0)

        result = torch.logical_not(self.comparison_func(
            new_cases.unsqueeze(1), self.X_train))
        

        # Default should not be attacked by new case
        result[:, self.default_indexes] = False

        # Cases that are not in the casebase indices are not attacked
        mask = torch.ones(len(self.X_train), dtype=bool)
        mask[self.get_indices()] = False
        result[:, mask] = False

        return result

    def compute_grounded(self, new_cases_attacks):

        if new_cases_attacks.ndim == 1:
            new_cases_attacks = new_cases_attacks.unsqueeze(0)

        A = self.A.clone()

        batch_size = new_cases_attacks.shape[0]

        # Batch A - to support multiple new_cases at once
        A = torch.tile(A.unsqueeze(0), (batch_size, 1, 1))

        n = A.shape[-1]

        # Filter cases in A that are attacked by new_cases
        new_cases_attacks_mask = new_cases_attacks.unsqueeze(-1).repeat(1, 1, n)
        A = torch.where(new_cases_attacks_mask, 0, A)

        # Find unattacked nodes (i.e columns with all 0s)
        # For each node, x, that they attack, set all attacks originating from x to 0
        # Repeat until no more changes
        """
        a -> b -> c

        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]

        first iteration:
            unattacked = [true, false, false] as column 0 is all 0s
            Mask = [[true], [false], [false]] -> dim change
            only_unattacked = [[0, 1, 0],  [0, 0, 0],  [0, 0, 0]] -> only the first row is unattacked so give that row only
            attacked =  [false, true, false] as the first row is non-zero in column 1
            A  = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
        """
        while True:
            unattacked = torch.logical_and(torch.all(A == 0, axis=1), torch.logical_not(
                new_cases_attacks))  # Shape: B x n

            mask = unattacked.unsqueeze(2)

            # Get the rows of the adjacency matrix that are unattacekd and zero the attacked rows
            only_unattacked = torch.where(mask, A, 0)

            # Get the mask of the rows that are attacked by unattacked rows
            attacked = torch.any(only_unattacked != 0, axis=1)  # Shape: B x n

            if torch.all(A[attacked] == 0):
                break
            
            # Filter out attacked nodes
            attacked_mask = attacked.unsqueeze(-1).repeat(1, 1, n)
            A = torch.where(attacked_mask, 0, A)


        new_cases_attacks_ints = torch.where(new_cases_attacks, 1, 0)
        A = (1 + A)
        result = A.prod(axis=1)
        result = torch.mul((1-new_cases_attacks_ints), result)

        return result
        
        

    def forward(self, new_cases):
        
        new_cases_attacks = self.get_new_case_attacks_mask(new_cases)
        grounded = self.compute_grounded(new_cases_attacks)
        return grounded[:, self.default_indexes]

