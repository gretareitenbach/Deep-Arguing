import torch
from deeparguing.semantics.gradual_semantics import GradualSemantics
from deeparguing.irrelevance_edge_weights.compute_irrelevance import IrrelevanceType
from deeparguing.casebase_edge_weights.compute_partial_order import PartialOrderType
from deeparguing.base_scores.compute_base_scores import BaseScoreType 
from deeparguing import GradualAACBR


class SlowGradualAACBR(GradualAACBR):

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
        """
        super().__init__(gradual_semantics, compute_base_score, irrelevance_edge_weights, casebase_edge_weights, 
                         use_symmetric_attacks, defaults_not_attack, use_blockers, use_supports,
                         ) 

    ############################################################################
    # The following contains an implementation of fit that makes limited use of 
    # broadcasting, vectorisation or matrix operations and is used for sanity
    # checking results    
    ############################################################################
    def __casebase_edge_weights_strict(self, attacker, target):
        #This is ineffecient because of repeated calls to casebase_edge_weights
        return self.casebase_edge_weights(attacker, target) * (1 - self.casebase_edge_weights(target, attacker))

    def __casebase_edge_weights_equal(self, attacker, target):
        #This is ineffecient because of repeated calls to casebase_edge_weights
        return self.casebase_edge_weights(attacker, target) * self.casebase_edge_weights(target, attacker)

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, X_default: torch.Tensor, y_default: torch.Tensor, 
            batch_size = None):

        if (X_train is None or y_train is None or len(X_train) != len(y_train)):
            raise(Exception(f"Length of X_train must match length of y_train. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"))

        if (X_default is None or y_default is None or len(X_default) != len(y_default)):
            raise(Exception(f"Length of X_default must match length of y_default. X_default shape: {X_default.shape}, y_default shape: {y_default.shape}"))

        if batch_size is not None:
            print("Warning: batch_size is ignored in this implementation of fit")

        self.A = None
        X_train, y_train, default_indexes = self._add_default_cases(X_train, y_train, X_default, y_default)

        train_size = len(X_train)
        

        X_attackers = X_train.unsqueeze(1).expand(-1, len(X_train), -1)
        X_targets = X_train.unsqueeze(0).expand(len(X_train), -1, -1)


        edge_weights_strict = self.__casebase_edge_weights_strict(X_attackers, X_targets).reshape((train_size, train_size))

        if use_symmetric_attacks:
            edge_weights_equal = self.__casebase_edge_weights_equal(X_attackers, X_targets).reshape((train_size, train_size))
        else:
            edge_weights_equal = torch.zeros_like(edge_weights_strict)

    
        result = torch.zeros_like(edge_weights_strict)

        for attacker_index in range(train_size):

            if defaults_not_attack and attacker_index in default_indexes:
                continue

            for target_index in range(train_size):

                if attacker_index == target_index:
                    continue

                # Could refactor to remove repeats between supports and attacks 
                # but the code here is supposed to be a translation of the fuzzy logic 
                # (and is only used for testing) so have left it as is

                if torch.all(y_train[attacker_index] == y_train[target_index], dim=-1):
                    if not use_supports:
                        continue
                    # Supports
                    blocker_value = 1
                    if use_blockers:

                        for blocker_index in range(train_size):

                            if (defaults_not_attack and blocker_index in default_indexes):
                                continue

                            blocker_value = blocker_value * (1 - (edge_weights_strict[attacker_index, blocker_index] * edge_weights_strict[blocker_index, target_index]))

                    result[attacker_index, target_index] = edge_weights_strict[attacker_index, target_index] * blocker_value 
                else:
                    # Attacks
                    blocker_value = 1
                    if use_blockers:

                        for blocker_index in range(train_size):

                            if torch.all(y_train[attacker_index] != y_train[blocker_index], dim=-1) or (defaults_not_attack and blocker_index in default_indexes):
                                continue

                            blocker_value = blocker_value * (1 - (edge_weights_strict[attacker_index, blocker_index] * edge_weights_strict[blocker_index, target_index]))


                    result[attacker_index, target_index] = -(edge_weights_strict[attacker_index, target_index] * blocker_value + edge_weights_equal[attacker_index, target_index])

        self.A = result
        self.X_train = X_train
        self.y_train = y_train
        self.default_indexes = default_indexes
    




    
