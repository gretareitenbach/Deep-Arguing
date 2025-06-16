import torch

from deeparguing import GradualAACBR
from deeparguing.base_scores import *
from deeparguing.casebase_edge_weights import *
from deeparguing.cli import (parse_command_line, parse_model_config, plots,
                             read_config_files)
from deeparguing.clustering import *
from deeparguing.evals import evaluate_model, print_results
from deeparguing.feature_extractor import *
from deeparguing.helper import *
from deeparguing.irrelevance_edge_weights import *
from deeparguing.regulariser import *
from deeparguing.semantics import *
from deeparguing.train import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


if __name__ == "__main__":

    args = parse_command_line()
    plot_matrix_before, plot_matrix_after, plot_graph_before, plot_graph_after = plots(
        args
    )

    total = 0

    model_config = read_config_files(args.config)

    for seed_idx, seed in enumerate(args.seed):
        # TODO: Move this logic to separate function
        print("=" * 100)
        print(f"Running With Torch Seed: {seed}")
        torch.manual_seed(seed)

        data_dict, instances = parse_model_config(model_config, device=device)

        assert data_dict
        assert instances

        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val"]

        model: GradualAACBR = instances["model"].to(device)

        trainer: Trainer = instances["trainer"]
        train_settings = instances["train_settings"]

        no_features = data_dict["no_features"]
        labels = data_dict["labels"]

        X_casebase, y_casebase = instances["build_casebase"](X_train, y_train)

        X_defaults = X_train.mean(dim=0).tile(len(labels), 1)
        y_defaults = labels.flip([0])

        if plot_matrix_before:
            model.show_matrix()
        if plot_graph_before:
            model.show_graph()

        trainer.train(
            model,
            X_casebase,
            y_casebase,
            X_defaults,
            y_defaults,
            **train_settings,
            disable_tqdm=args.disable_tqdm,
        )

        acc, prec, rec, f1, cm = evaluate_model(
            model, X_casebase, y_casebase, X_defaults, y_defaults, X_val, y_val
        )
        print_results(acc, prec, rec, f1, cm, "VALIDATION", labels)

        if args.run_test:
            X_test = data_dict["X_test"]
            y_test = data_dict["y_test"]

            acc, prec, rec, f1, cm = evaluate_model(
                model, X_casebase, y_casebase, X_defaults, y_defaults, X_test, y_test
            )
            print_results(acc, prec, rec, f1, cm, "TEST", labels)

        if args.plot_loss:
            trainer.plot_loss_curve()
        if args.plot_gradients:
            trainer.plot_grads()
        if plot_matrix_after:
            model.show_matrix()
        if plot_graph_after:
            model.show_graph()

        print("=" * 100)
        if f1 > 0.7:
            total += 1
        if len(args.seed) > 1:
            print("PERCENT F1", total / (seed_idx + 1) * 100)
