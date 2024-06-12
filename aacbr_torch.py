import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class AACBRTorch(torch.nn.Module):

    def __init__(self, X_train, y_train, comparison_func, default_case, default_outcomes,
                 use_symmetric_attacks=True, build_parallel=False, casebase_indices=None) -> None:

        super(AACBRTorch, self).__init__()
        assert (len(X_train) == len(y_train))
        self.comparison_func = comparison_func
        self.X_train = X_train
        self.y_train = y_train
        self.default_case = default_case
        self.default_outcomes = default_outcomes
        self.use_symmetric_attacks = use_symmetric_attacks
        self.casebase_indices = casebase_indices

        self.add_default_cases(default_case, default_outcomes)
        if build_parallel:
            self.build_af_parallel()
        else:
            self.build_af()

    def add_default_cases(self, default_case, default_outcomes):

        pre = len(self.X_train)
        self.X_train = torch.cat(
            (self.X_train, default_case.repeat(len(default_outcomes), 1)), dim=0)
        self.y_train = torch.cat((self.y_train, default_outcomes), dim=0)

        post = len(self.X_train)
        self.default_indexes = list(range(pre, post))
        # self.outcome_map = {outcome: self.default_indexes[i] for i, outcome in enumerate(self.default_outcomes)}

    def get_indices(self):
        if self.casebase_indices is None:
            return list(range(len(self.X_train)))
        else:
            return torch.cat((self.casebase_indices, self.default_indexes))

    def build_af(self):
        # Builds an AF with a matrix representation with a size of len(X_train)
        # If casebase_indices is  specified, only cases at these indices will be used to build the AF
        # which can result in a matrix that is larger than necessary but has a fixed size
        # and nodes in a fixed location if the casebase changes
        # TODO: Use more efficient implementation that sorts topologically as in AA-CBR Library

        A = torch.zeros((len(self.X_train), len(self.X_train)),
                        dtype=torch.float32)

        for i in self.get_indices():
            attacker = self.X_train[i]
            if i in self.default_indexes:
                continue
            for j in self.get_indices():
                target = self.X_train[j]
                if self.y_train[i] == self.y_train[j]:
                    continue

                if self.comparison_func(attacker, target)[0] and self.is_concise(attacker, target, self.y_train[i]):
                    A[i, j] = -1
                elif self.use_symmetric_attacks and all(attacker == target):
                    A[i, j] = -1

        self.A = A

    def build_af_parallel(self):

        # TODO: Could find a way to make use of Topological ordering to speed this up further
        train_size = len(self.X_train)
        all_indices = torch.arange(train_size)
        indexes = torch.cartesian_prod(
            [all_indices, all_indices,  all_indices]
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

        potential_attacks = torch.logical_and(
            attackers_labels != targets_labels, self.comparison_func(attackers, targets))
        is_blocked = torch.logical_and(blockers_labels == attackers_labels, torch.logical_and(
            self.comparison_func(attackers, blockers), self.comparison_func(blockers, targets)))
        symmetric_attacks = torch.logical_and(self.use_symmetric_attacks, torch.logical_and(
            attackers_labels != targets_labels, torch.all(attackers == targets, axis=1)))
        attacks = torch.logical_or(torch.logical_and(
            potential_attacks, torch.logical_not(is_blocked)), symmetric_attacks)

        attacks[blockers_mask] = True
        attacks = torch.logical_and(
            attacks, torch.logical_and(attackers_mask, targets_mask))

        attacks = attacks.reshape((train_size, train_size, train_size))
        attacks = torch.all(attacks, axis=-1)
        attacks[self.default_indexes, :] = False
        self.A = torch.where(attacks, -1, 0)

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

        A[new_cases_attacks, :] = 0

        # Find unattacked nodes (i.e columns with all 0s)
        # For each node, x, that they attack, set all attacks originating from x to 0
        # Repeat until no more changes
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

            A[attacked, :] = 0

        final_unattacked = torch.logical_and(
            torch.all(A[:,] == 0, axis=1), torch.logical_not(new_cases_attacks))

        return final_unattacked

    def forward(self, new_cases):
        new_cases_attacks = self.get_new_case_attacks_mask(new_cases)
        grounded = self.compute_grounded(new_cases_attacks)
        predicted = torch.where(
            grounded[:, self.default_indexes], 1, 0)
        return predicted

    # def __call__(self, new_cases):
    #     return self.forward(new_cases)
