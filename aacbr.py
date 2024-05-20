import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


class AACBR:

    def __init__(self, X_train, y_train, comparison_func, default_case, default_outcomes, use_symmetric_attacks = True, build_parallel = False) -> None:
        self.comparison_func = comparison_func
        self.X_train = X_train
        self.y_train = y_train
        self.default_case = default_case
        self.default_outcomes = default_outcomes
        self.use_symmetric_attacks = use_symmetric_attacks
        self.add_default_cases(default_case, default_outcomes)
        if build_parallel:
            self.build_af_parallel()
        else:
            self.build_af()

    def add_default_cases(self, default_case, default_outcomes):

        pre = len(self.X_train)
        self.X_train = np.append(self.X_train, [default_case]*len(default_outcomes), axis=0)
        self.y_train = np.append(self.y_train, default_outcomes)
        post = len(self.X_train)
        self.default_indexes = list(range(pre, post)) 
        # self.outcome_map = {outcome: self.default_indexes[i] for i, outcome in enumerate(self.default_outcomes)}

    def build_af(self):
        # TODO: Use more efficient implementation that sorts topologically as in AA-CBR Library

        A = np.zeros((len(self.X_train), len(self.X_train)), dtype=np.float32)

        for i, attacker in enumerate(self.X_train):
            if i in self.default_indexes:
                continue
            for j, target in enumerate(self.X_train):
                if self.y_train[i] == self.y_train[j]:
                    continue
                # if np.all(target == 0):
                #     print(attacker)
                # if self.comparison_func(attacker, target)[0]:
                #     print("AHH")
                if self.comparison_func(attacker, target)[0] and self.is_concise(attacker, target, self.y_train[i]):
                    A[i, j] = -1
                elif self.use_symmetric_attacks and all(attacker == target):
                    A[i, j] = -1

        self.A = A

    def cartesian_product_simple_transpose(self, arrays):
        # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[i, ...] = a
        return arr.reshape(la, -1).T
    
    def build_af_parallel(self):

        #TODO: Could find a way to make use of Topological ordering to speed this up further
        train_size = len(self.X_train)
        indexes = self.cartesian_product_simple_transpose(
            [np.arange(train_size),
             np.arange(train_size),
             np.arange(train_size)
            ]
        )

        attackers, attackers_labels = self.X_train[indexes[:, 0]], self.y_train[indexes[:, 0]]
        targets, targets_labels     = self.X_train[indexes[:, 1]], self.y_train[indexes[:, 1]]
        blockers, blockers_labels   = self.X_train[indexes[:, 2]], self.y_train[indexes[:, 2]]

        potential_attacks = np.logical_and(attackers_labels != targets_labels, self.comparison_func(attackers, targets))
        is_blocked = np.logical_and(blockers_labels == attackers_labels, np.logical_and(self.comparison_func(attackers, blockers), self.comparison_func(blockers, targets)))
        symmetric_attacks = np.logical_and(self.use_symmetric_attacks, np.logical_and(attackers_labels != targets_labels, np.all(attackers == targets, axis=1)))
        attacks = np.logical_or(np.logical_and(potential_attacks, np.logical_not(is_blocked)), symmetric_attacks)


        attacks = attacks.reshape((train_size, train_size, train_size))
        attacks = np.all(attacks, axis=-1)
        attacks[self.default_indexes, :] = False
        self.A = np.where(attacks, -1, 0)



    def is_concise(self, attacker, target, attacker_outcome):
        return not any([
            self.y_train[k] == attacker_outcome and
            self.comparison_func(attacker, blocker)[
                0] and self.comparison_func(blocker, target)[0]
            for k, blocker in enumerate(self.X_train)])

    def show_graph_with_labels(self):
        rows, cols = np.where(self.A != 0)
        edges = zip(rows.tolist(), cols.tolist())

        gr = nx.DiGraph()
        gr.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(gr, prog='dot',
                                           args='-Gsplines=true -Gnodesep=2')

        unique_labels = np.unique(self.y_train)
        colormap = plt.get_cmap('gist_rainbow', len(unique_labels))
        label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
        node_colors = [label_to_color[self.y_train[node]] for node in list(gr.nodes)]

        assert(all([default_index in list(gr.nodes) for default_index in self.default_indexes]))


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
                node_size=100, font_size = 5, width=0.4)

        plt.legend()
        plt.show()

    def show_matrix(self):
        plt.figure(figsize=(10, 10))

        B = self.A.copy()
        plt.imshow(B, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        plt.title('Edges')
        plt.show()

    def get_new_case_attacks_mask(self, new_cases):
        if new_cases.ndim == 1:
            new_cases = np.expand_dims(new_cases, axis=0)

        result = np.logical_not(self.comparison_func(
            new_cases[:, np.newaxis, :], self.X_train))
        
        # Default should not be attacked by new case
        result[:, self.default_indexes] = False

        return result

    def compute_grounded(self, new_cases_attacks):

        if new_cases_attacks.ndim == 1:
            new_cases_attacks = np.expand_dims(new_cases_attacks, axis=0)

        A = self.A.copy()

        batch_size = new_cases_attacks.shape[0]

        # Batch A - to support multiple new_cases at once
        A = np.tile(A[np.newaxis, :, :], (batch_size, 1, 1))

        A[new_cases_attacks, :] = 0

        # Find unattacked nodes (i.e columns with all 0s)
        # For each node, x, that they attack, set all attacks originating from x to 0
        # Repeat until no more changes
        while True:
            unattacked = np.logical_and(np.all(A == 0, axis=1), np.logical_not(
                new_cases_attacks))  # Shape: B x n

            mask = unattacked[:, :, np.newaxis]

            # Get the rows of the adjacency matrix that are unattacekd and zero the attacked rows
            only_unattacked = np.where(mask, A, 0)

            # Get the mask of the rows that are attacked by unattacked rows
            attacked = np.any(only_unattacked != 0, axis=1)  # Shape: B x n

            if np.all(A[attacked] == 0):
                break

            A[attacked, :] = 0

        final_unattacked = np.logical_and(
            np.all(A[:,] == 0, axis=1), np.logical_not(new_cases_attacks))

        return final_unattacked

    def forward(self, new_cases):
        new_cases_attacks = self.get_new_case_attacks_mask(new_cases)
        grounded = self.compute_grounded(new_cases_attacks)
        predicted = np.where(
            grounded[:, self.default_indexes], 1, 0)
        return predicted

    def __call__(self, new_cases):
        return self.forward(new_cases)
