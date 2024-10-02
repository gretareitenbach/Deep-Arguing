import torch
import torch.optim as optim
from deeparguing.gradual_aacbr import GradualAACBR
import ray
from ray import tune
from deeparguing.train import evaluate_model


def objective(config, X_casebase, y_casebase, X_default, y_default, X_train_new_cases, y_train_new_cases,
              X_eval_new_cases, y_eval_new_cases,
              show_confusion=False, print_matrix=False,
              print_compute_graph=False, print_graph=False, print_results=False, use_blockers=False,
              disable_tqdm = False, plot_loss_curve = False):

    semantics = config["semantics"]
    partial_order = config["partial_order"]
    irrelevance = config["irrelevance"]
    base_score = config["base_score"]

    model = GradualAACBR(semantics,
                         base_score,
                         irrelevance,
                         partial_order)
    
    

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=config["lr"], momentum=config["momentum"])

    config["train"](model,
          X_casebase, y_casebase, X_train_new_cases, y_train_new_cases, X_default, y_default,
          optimizer, criterion, config["epochs"],
          config["symmetric_attacks"],
          use_blockers=False, plot_loss_curve=plot_loss_curve, disable_tqdm=disable_tqdm)


    accuracy, precision, recall, f1 = evaluate_model(model, X_casebase, y_casebase, X_default, y_default, 
                                                     X_eval_new_cases, y_eval_new_cases,
                                                     show_confusion=show_confusion, print_matrix=print_matrix,
                                                     print_compute_graph=print_compute_graph, print_graph=print_graph,
                                                     print_results=print_results, use_symmetric_attacks=config["symmetric_attacks"], 
                                                     use_blockers=use_blockers)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def tune_model(search_space, X_casebase, y_casebase, X_default, y_default, 
               X_train_new_cases, y_train_new_cases, X_eval_new_cases, y_eval_new_cases,
                disable_tqdm = False, num_cpus=12, num_samples=1000, metric="accuracy"):

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)

    tuner = tune.Tuner(
        tune.with_parameters(objective, X_casebase=X_casebase, y_casebase=y_casebase, X_default=X_default,
                             y_default=y_default, 
                             X_train_new_cases=X_train_new_cases, y_train_new_cases=y_train_new_cases, 
                             X_eval_new_cases = X_eval_new_cases, y_eval_new_cases = y_eval_new_cases,
                             disable_tqdm = disable_tqdm),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode="max",
            num_samples=num_samples,
        ),
        param_space=search_space
    )

    results = tuner.fit()
    best_params = results.get_best_result().config
    print("BEST PARAMETERS FOR MODEL ARE:")
    print(best_params)
    print("Best score found was: ", results.get_best_result().metrics[metric])
    return best_params


# search_space = {
#     "semantics": ...,
#     "partial_order": ...,
#     "irrelevance": ...,
#     "base_score": ...,
#     "train": ...,
#     "no_iters": tune.choice([15, 20, 25]),
#     "epochs": tune.choice([250, 750, 1500, 3000]),
#     "lr": tune.loguniform(1e-4, 1e-1),
#     "momentum": tune.loguniform(1e-4, 1e-1),
#     "sharpness": tune.loguniform(1e-1, 1e1),
#     "seed": tune.randint(0, 100),
#     "symmetric_attacks": tune.choice([True, False]),
# }
