import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torchviz import make_dot


def train_step(model,
               X_casebase, y_casebase, X_new_cases, y_new_cases, X_defaults, y_defaults,
               optimizer, criterion,
               use_symmetric_attacks,
               use_blockers=True):

    optimizer.zero_grad()

    # TODO: consider efficiency issues with having to rebuild each time
    # Find a way to accumulate gradients update only when necessary?
    model.fit(X_casebase, y_casebase, X_defaults, y_defaults,
              use_symmetric_attacks=use_symmetric_attacks, use_blockers=use_blockers)

    predictions = model(X_new_cases).squeeze()
    loss = criterion(predictions, y_new_cases)
    loss.backward()

    optimizer.step()
    return loss


def train_model(model,
          X_casebase, y_casebase, X_new_cases, y_new_cases, X_defaults, y_defaults,
          optimizer, criterion, epochs,
          use_symmetric_attacks,
          use_blockers=True, plot_loss_curve=False, disable_tqdm=False):

    losses = []
    pbar = tqdm(range(epochs), disable=disable_tqdm)

    for epoch in pbar:

        loss = train_step(model,
                          X_casebase, y_casebase, X_new_cases, y_new_cases, X_defaults, y_defaults,
                          optimizer, criterion,
                          use_symmetric_attacks,
                          use_blockers=use_blockers)

        losses.append(loss.item())

        pbar.set_description(
            f'Epoch {epoch + 1}, Loss: {round(loss.item()/len(X_new_cases), 4)}')
    
    # # print model gradients:
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.grad)


    if plot_loss_curve:

        plt.plot(losses)
        plt.show()


def dynamic_train_model(model,
                  X_train, y_train, X_defaults, y_defaults,
                  optimizer, criterion, epochs,
                  use_symmetric_attacks, n_splits,
                  use_blockers=True, plot_loss_curve=False, disable_tqdm=False, random_state=42):

    losses = []
    pbar = tqdm(range(epochs), disable=disable_tqdm)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for epoch in pbar:
        for fold, (casebase_index,  new_cases_index) in enumerate(kf.split(X_train)):

            X_casebase = X_train[casebase_index]
            y_casebase = y_train[casebase_index]

            X_new_cases = X_train[new_cases_index]
            y_new_cases = y_train[new_cases_index]

            loss = train_step(model,
                              X_casebase, y_casebase, X_new_cases, y_new_cases, X_defaults, y_defaults,
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
                      X_defaults, y_defaults, new_cases, use_symmetric_attacks, use_blockers=False):


    model.fit(X_casebase, y_casebase, X_defaults, y_defaults,
              use_symmetric_attacks=use_symmetric_attacks, use_blockers=use_blockers)

    return model(new_cases)


def evaluate_model(model, X_train, y_train, X_default, y_default, new_cases, new_cases_labels, print_results=True,
                   show_confusion=False, print_graph=False, print_matrix=False, print_compute_graph=False,
                   use_symmetric_attacks=False, use_blockers=False):

    final_strengths = run_gradual_model(model, X_train, y_train, X_default, y_default,
                                        new_cases, use_symmetric_attacks=use_symmetric_attacks, use_blockers=use_blockers)
    predicted = final_strengths.cpu().detach().numpy()

    predicted = np.argmax(predicted, axis=1)
    new_cases_labels_orig = np.argmax(new_cases_labels.cpu().detach().numpy(), axis=1)

    results = (
        accuracy_score(new_cases_labels_orig, predicted),
        precision_score(new_cases_labels_orig, predicted, average='macro', zero_division=0),
        recall_score(new_cases_labels_orig, predicted, average='macro', zero_division=0),
        f1_score(new_cases_labels_orig, predicted, average='macro', zero_division=0)
    )

    if print_results:
        print("Accuracy, Precision, Recall, F1")
        print(results)

    if show_confusion:
        cm = confusion_matrix(new_cases_labels_orig, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    if print_graph:
        model.show_graph_with_labels()

    if print_matrix:
        model.show_matrix()

    if print_compute_graph:
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(final_strengths.squeeze(), new_cases_labels)
        make_dot(loss, params=dict(model.named_parameters())
                 ).render("gradual_aacbr", format="pdf")

    return results
