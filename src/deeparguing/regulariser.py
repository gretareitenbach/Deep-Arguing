import torch



def sparsity_regulariser(model): 
    return torch.sum(torch.abs(model.A))

def community_preservation_regulariser(model): 
    return torch.sum(torch.svd(model.A).S)

def connectivity_regulariser(model, eps=1e-6):
    A = torch.sum(torch.abs(model.A), dim =1) + eps
    return torch.sum(torch.log(A))

def feature_smoothness_regulariser(model):
    D = torch.diag(torch.sum(model.A, dim=1))
    L = D - model.A

    # TODO: We might want to consider feature weightings
    # in which case we need a weighting vector 
    # (like what we use in the base_scores or edge_weight functions)
    X = model.X_train
    
    return torch.trace(X.T @ L @ X)



def regularise(model, regularisers):
    # Each regulariser is a pair (reg_func, weight)
    # [[func, w], [func, w], [func, w]]

    total = 0

    for reg_func, weight in regularisers:
        total += weight * reg_func(model)
    
    return total
