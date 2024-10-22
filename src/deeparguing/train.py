import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torchviz import make_dot


def train_step(model,
               X_casebase, y_casebase, X_new_cases, y_new_cases, X_default, y_default,
               optimizer, criterion,
               use_symmetric_attacks,
               use_blockers=True):

    optimizer.zero_grad()

    # TODO: consider efficiency issues with having to rebuild each time
    # Find a way to accumulate gradients update only when necessary?
    model.fit(X_casebase, y_casebase, X_default, y_default,
              use_symmetric_attacks=use_symmetric_attacks, use_blockers=use_blockers)

    predictions = model(X_new_cases).squeeze()
    loss = criterion(predictions, y_new_cases)
    loss.backward()

    optimizer.step()
    return loss


def static_train_model(model, X_casebase, y_casebase, X_default, y_default, optimizer, criterion, epochs, use_symmetric_attacks, X_new_cases=None, y_new_cases=None,
            n_splits = None, use_blockers=True, plot_loss_curve=False, disable_tqdm=False, random_split_state=None):

    if X_new_cases is None or y_new_cases is None:
        raise Exception("X_new_cases and y_new_cases cannot be None")
    if n_splits is not None:
        # TODO: Change to logging
        print("Warning: n_splits is not used by static training")
    if random_split_state is not None:
        # TODO: Change to logging
        print("Warning: random_state is not used by static training")

    losses = []
    pbar = tqdm(range(epochs), disable=disable_tqdm)

    for epoch in pbar:

        loss = train_step(model,
                          X_casebase, y_casebase, X_new_cases, y_new_cases, X_default, y_default,
                          optimizer, criterion,
                          use_symmetric_attacks,
                          use_blockers=use_blockers)

        losses.append(loss.item())

        pbar.set_description(
            f'Epoch {epoch + 1}, Loss: {round(loss.item()/len(X_new_cases), 4)}')
    

    if plot_loss_curve:

        plt.plot(losses)
        plt.show()


def dynamic_train_model(model, X_casebase, y_casebase, X_default, y_default, optimizer, criterion, epochs, use_symmetric_attacks, X_new_cases=None, y_new_cases=None,
            n_splits = None, use_blockers=True, plot_loss_curve=False, disable_tqdm=False, random_split_state=None):

    if X_new_cases is not None or y_new_cases is not None:
        # TODO: Change to logging
        print("Warning: X_new_cases and y_new_cases is not used by dynamic training")
    if n_splits is None:
        raise Exception("n_splits cannot be None")
    if random_split_state is None:
        raise Exception("random_state cannot be None")

    losses = []
    pbar = tqdm(range(epochs), disable=disable_tqdm)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_split_state)

    for epoch in pbar:
        for fold, (casebase_index,  new_cases_index) in enumerate(kf.split(X_casebase)):

            X_sub_casebase = X_casebase[casebase_index]
            y_sub_casebase = y_casebase[casebase_index]

            X_new_cases = X_casebase[new_cases_index]
            y_new_cases = y_casebase[new_cases_index]

            loss = train_step(model,
                              X_sub_casebase, y_sub_casebase, X_new_cases, y_new_cases, X_default, y_default,
                              optimizer, criterion,
                              use_symmetric_attacks,
                              use_blockers=use_blockers)

            losses.append(loss.item()/len(X_new_cases))

        pbar.set_description(
            f'Epoch {epoch + 1}, Loss: {round(loss.item()/len(X_new_cases), 4)}')

    if plot_loss_curve:

        plt.plot(losses)
        plt.show()

def run_gradual_model(model, X_casebase, y_casebase,
                      X_default, y_default, X_new_cases, use_symmetric_attacks, use_blockers=True):


    model.fit(X_casebase, y_casebase, X_default, y_default,
              use_symmetric_attacks=use_symmetric_attacks, use_blockers=use_blockers)

    return model(X_new_cases)


def evaluate_model(model, X_casebase, y_casebase, X_default, y_default, X_new_cases, y_new_cases, print_results=True,
                   show_confusion=False, print_graph=False, print_matrix=False, print_compute_graph=False,
                   use_symmetric_attacks=False, use_blockers=True):

    final_strengths = run_gradual_model(model, X_casebase, y_casebase, X_default, y_default,
                                        X_new_cases, use_symmetric_attacks=use_symmetric_attacks, use_blockers=use_blockers)
    y_predicted = final_strengths.cpu().detach().numpy()

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
        plt.show()

    if print_graph:
        model.show_graph_with_labels()

    if print_matrix:
        model.show_matrix()

    if print_compute_graph:
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(final_strengths.squeeze(), y_new_cases)
        make_dot(loss, params=dict(model.named_parameters())
                 ).render("gradual_aacbr", format="pdf")

    return results
