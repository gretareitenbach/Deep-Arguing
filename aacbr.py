import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class AACBR:

    def __init__(self, X_train, y_train, comparison_func, default_case, default_outcome, use_symmetric_attacks = True, build_parallel = False) -> None:
        self.comparison_func = comparison_func
        self.X_train = X_train
        self.y_train = y_train
        self.default_case = default_case
        self.default_outcome = default_outcome
        self.use_symmetric_attacks = use_symmetric_attacks
        self.add_default_case(default_case, default_outcome)
        if build_parallel:
            self.build_af_parallel()
        else:
            self.build_af()

    def add_default_case(self, default_case, default_outcome):

        self.X_train = np.append(self.X_train, [default_case], axis=0)
        self.y_train = np.append(self.y_train, [default_outcome])
        self.default_index = len(self.X_train) - 1

    def build_af(self):
        # TODO: Use more efficient implementation that sorts topologically as in AA-CBR Library

        A = np.zeros((len(self.X_train), len(self.X_train)), dtype=np.float32)

        for i, attacker in enumerate(self.X_train):
            if i == self.default_index:
                continue
            for j, target in enumerate(self.X_train):
                if self.y_train[i] == self.y_train[j]:
                    continue

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
        symmetric_attacks = np.logical_and(attackers_labels != targets_labels, np.all(attackers == targets, axis=1))
        attacks = np.logical_or(np.logical_and(potential_attacks, np.logical_not(is_blocked)), symmetric_attacks)


        attacks = attacks.reshape((train_size, train_size, train_size))
        attacks = np.all(attacks, axis=-1)
        attacks[self.default_index, :] = False
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
        gr = nx.Graph()
        gr.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(
            gr, prog='twopi', root=str(self.default_index))
        colors = ['red' if self.y_train[x] ==
                  self.default_outcome else 'blue' for x in list(gr.nodes)]
        if self.default_index != None:
            a = list(gr.nodes).index(self.default_index)
            colors[a] = 'green'

        plt.figure(figsize=(20, 20))
        nx.draw(gr, pos, labels={x: x for x in list(gr.nodes)},
                arrowstyle='-|>', arrows=True, node_color=colors)
        plt.show()

    def get_new_case_attacks_mask(self, new_cases):
        if new_cases.ndim == 1:
            new_cases = np.expand_dims(new_cases, axis=0)

        result = np.logical_not(self.comparison_func(
            new_cases[:, np.newaxis, :], self.X_train))
        
        # Default should not be attacked by new case
        result[:, self.default_index] = False

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
            grounded[:, self.default_index], self.default_outcome, 1 - self.default_outcome)
        return predicted

    def __call__(self, new_cases):
        return self.forward(new_cases)
