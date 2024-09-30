
def objective(config, test_set = False, show_confusion = False, print_matrix = False, 
              print_compute_graph = False, print_graph = False, print_results = False, no_blockers=True):
    torch.manual_seed(config["seed"]) # TRY DIFFERENT INITIAL WEIGHTS 

    no_features = X_train.shape[-1]
    semantics = rs.ReluSemantics(max_iters=config["no_iters"], epsilon=0)
    partial_order = fwpo.FeatureWeightedPartialOrder(no_features, sharpness=config["sharpness"])
    # partial_order = lpo.LearnedPartialOrder(no_features, no_hidden=10, sharpness=2)
    irrelevance = ri.RegularIrrelevance(partial_order)
    # irrelevance = fwi.FeatureWeightedIrrelevance(no_features)
    base_score = fwbs.FeatureWeightedBaseScore(no_features)

    model = gradual_aacbr.GradualAACBR(semantics, 
                                    base_score,
                                    irrelevance,
                                    partial_order)

    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=MOMENTUM)

    train(model, optimizer, criterion, epochs=config["epochs"], use_symmetric_attacks=config["symmetric_attacks"])

    if test_set:
        accuracy, precision, recall, f1 = evaluate_model(model, X_train_full, y_train_full, X_DEFAULTS, Y_DEFAULTS, X_test, y_test, 
                                        show_confusion=show_confusion, print_matrix=print_matrix, 
                                        print_compute_graph=print_compute_graph, print_graph=print_graph, 
                                        print_results=print_results, use_symmetric_attacks=config["symmetric_attacks"], no_blockers=no_blockers)
    else:
        accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_DEFAULTS, Y_DEFAULTS, X_val, y_val, 
                                        show_confusion=show_confusion, print_matrix=print_matrix, 
                                        print_compute_graph=print_compute_graph, print_graph=print_graph, 
                                        print_results=print_results, use_symmetric_attacks=config["symmetric_attacks"], no_blockers=no_blockers)


    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1 }


def tune_model(search_space, num_cpus=12, num_samples=1000, metric="accuracy"):
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)

    tuner = tune.Tuner(
        tune.with_parameters(objective, test_set=False),
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


search_space = {
    "no_iters": tune.choice([15, 20, 25]),
    "epochs": tune.choice([250, 750, 1500, 3000]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "sharpness": tune.loguniform(1e-1, 1e1),
    "seed": tune.randint(0, 100),
    "symmetric_attacks": tune.choice([True, False]),
}
