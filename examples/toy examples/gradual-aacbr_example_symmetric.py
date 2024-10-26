import torch
import deeparguing
import deeparguing.semantics.relu_semantics as rs
import deeparguing.base_scores.feature_weighted_base_score as fwbs
import deeparguing.irrelevance_edge_weights.feature_weighted_irrelevance as fwi


"""
                    1: (3, 0)
                       ^
                       |
3: (5, 0) <------> 2: (5, 1)
                   /   |
                  /    v
                 /  0: (1, 0)
                /      |
              |_       v
5: (default, 0)  4: (default, 1)

"""


X_train = torch.tensor([
    [1], # (0, 0)
    [3], # (1, 0)
    [5], # (2, 1)
    [5], # (3, 0)
], dtype=torch.float32)

y_train = torch.tensor([
    [1, 0], 
    [1, 0], 
    [0, 1], 
    [1, 0]
], dtype=torch.float32)

X_default = torch.tensor([
        [0], [0]
], dtype=torch.float32) # 4, 5
y_default = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)

torch.manual_seed(1)

no_features = X_train.shape[-1]
# semantics = ms.MLPBasedSemantics(max_iters=5, epsilon=0)
semantics = rs.ReluSemantics(max_iters=5, epsilon=0)

# edge_weights_test = lambda a, t: torch.where(torch.all(a >= t, axis=1), 1.0, 0.0) 
edge_weights_test = lambda a, t: torch.sigmoid(a[:, 0] - t[:, 0]) 


model = deeparguing.GradualAACBR(semantics, 
                                   fwbs.FeatureWeightedBaseScore(no_features), 
                                   fwi.FeatureWeightedIrrelevance(no_features), 
                                   edge_weights_test
                                   )

model.fit(X_train, y_train, X_default, y_default, use_symmetric_attacks=True)
new_case = torch.tensor([
    [2],
], dtype=torch.float32)

strengths = model(new_case)
print(strengths)


new_fit = model.A
model.show_matrix()
model.show_graph_with_labels()

model.slow_fit(X_train, y_train, X_default, y_default, use_symmetric_attacks=True)
slow_fit = model.A


strengths = model(new_case)
print(strengths)

model.show_matrix()
model.show_graph_with_labels()

assert(torch.all(new_fit == slow_fit))