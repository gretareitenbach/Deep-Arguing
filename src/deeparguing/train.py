import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torchviz import make_dot
from typing import Callable
from sklearn.cluster import KMeans


def evaluate_model(model, X_casebase, y_casebase, X_default, y_default, X_new_cases, y_new_cases, print_results=True,
                   show_confusion=False, print_graph=False, print_matrix=False, print_compute_graph=False,
                   use_symmetric_attacks=False, use_blockers=True,  return_predictions = False,
                   post_process_func = lambda x: x,
                   cm_logger=lambda x: x, matrix_logger = lambda x: x, graph_logger = lambda x: x, prevent_show=False,
                   use_supports=False):
    
    """
        Fits and executes the model, then evaluates it on accuracy, precision, 
        recall and f1

        Parameters
        ----------
        model : GradualAACBR
            GradualAACBR model to train
        X_casebase : torch.Tensor
            Input casebase argument characterisations as a tensor. 
            Shape (N, x1, ..., xn) where N is the number of casebase 
            arguments and (x1, ..., xn) is the shape of each argument.
        y_casebase : torch.Tensor
            Input casebase label as a tensor. Shape (N, Y) where N is the 
            number of casebase arguments and Y is the number of labels
        X_default : torch.Tensor
            Default arguments characterisations as a tensor.
            Shape (Y, x1, ..., xn) where Y is the number of labels 
            and (x1, ..., xn) is the shape of each argument.
        y_default : torch.Tensor
            Input casebase label as a tensor. Shape (Y, Y) where Y is the
            number of labels
        X_new_cases : torch.Tensor
            Input new_cases arguments characterisations as a tensor. 
            Shape (M, x1, ..., xn) where M is the number of new cases 
            and (x1, ..., xn) is the shape of each argument.
        y_new_cases : torch.Tensor
            Input casebase label as a tensor. Shape (M, Y) where M is the 
            number of new cases and Y is the number of labels

        print_results : bool, default true
            When true, the accuracy, precision, recall and f1 is printed to
            console

        show_confusion : bool, default false
            When true, the confusion matrix graph is created
        
        print_graph : bool, default false
            When true, the model adjacency matrix is visualised as a connected
            graph
        
        print_matrix : bool, default false
            When true, the model adjacency matrix is visualised as a heatmap
        
        print_compute_graph : bool, default false
            When true, a pdf with the compute graph is outputted

        use_symmetric_attack : bool, default false
            When true, symmetric attacks between cases of the same 
            characterisation are included
        use_blockers : bool, default true
            When true, the model will optimise attacks of minimal between cases
            of minimal difference

        Returns
        -------
        results : Tuple
            returns a tuple containing the accuracy, precision, recall 
            and f1 score
        
    """


    final_strengths = run_gradual_model(model, X_casebase, y_casebase, X_default, y_default,
                                        X_new_cases, use_symmetric_attacks=use_symmetric_attacks, use_blockers=use_blockers, 
                                        post_process_func=post_process_func, use_supports=use_supports)

    y_predicted = final_strengths.cpu().detach().numpy() 
    # print(y_predicted)
    y_predicted = np.argmax(y_predicted, axis=1)
    y_new_cases_orig = np.argmax(y_new_cases.cpu().detach().numpy(), axis=1)

    results = (
        accuracy_score(y_new_cases_orig, y_predicted),
        precision_score(y_new_cases_orig, y_predicted, average='macro', zero_division=0),
        recall_score(y_new_cases_orig, y_predicted, average='macro', zero_division=0),
        f1_score(y_new_cases_orig, y_predicted, average='macro', zero_division=0)
    )

    if print_results:
        print("Accuracy, Precision, Recall, F1")
        print(results)

    if show_confusion:

        cm = confusion_matrix(y_new_cases_orig, y_predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        cm_logger(disp.figure_)
        if not prevent_show:
            plt.show()

    if print_graph:
        model.show_graph(post_process_func=post_process_func, logger=graph_logger, prevent_show=prevent_show)

    if print_matrix:
        model.show_matrix(post_process_func=post_process_func, logger=matrix_logger, prevent_show=prevent_show)

    if print_compute_graph:
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(final_strengths.squeeze(), y_new_cases)
        make_dot(loss, params=dict(model.named_parameters())
                 ).render("gradual_aacbr", format="pdf")
    
    if return_predictions:
        return tuple(list(results) + [y_new_cases_orig, y_predicted])

    return results

def cluster_data(X, y, cluster_size_func: Callable[[int], int]):

    """
        For each label in y, make clusters of X. The number of clusters is 
        dependent on the cluster_size_func

        Parameters
        ----------

        X : array_like 
            The inputs to be clustered

        y : array_like
            The labels of the inputs

        cluster_size_func : Callable[[int], int] 
            A function that accepts the total number of items to be clustered
            and returns the number of clusters to produce


        Returns
        -------
        results : Tuple
            returns a pair of array_likes, the first of which is the cluster 
            centers and the second is the corresponding label for each cluster 
            center
        
    """

    original_shape = X.shape


    X_all_centroids = []
    y_all_centroids = []

    all_y = np.unique(y, axis=0)

    for selected_y in all_y:


        group = X[np.all(selected_y == y, axis=1)]
        group_size = len(group)

        group = group.reshape(group_size, -1)

        # Number of clusters
        k = cluster_size_func(len(group))

        print(f"{k} clusters for {selected_y}")

        # Create a KMeans object
        kmeans = KMeans(n_clusters=k, random_state=0)

        # Fit the model to the data and predict cluster assignments
        cluster_assignments = kmeans.fit_predict(group)

        # Get the centroids
        X_centroids_group = kmeans.cluster_centers_
        y_centroids_group = np.tile(selected_y, (k, 1))

        X_all_centroids.append(X_centroids_group)
        y_all_centroids.append(y_centroids_group)

    # original_shape = list(original_shape)
    # original_shape[0] = -1
    # original_shape = tuple(original_shape)
    X_centroids =  np.concatenate(X_all_centroids, axis=0)
    y_centroids =  np.concatenate(y_all_centroids, axis=0)
    return X_centroids, y_centroids
