import logging
import os

# This will ensure determism of the KMEANS Clustering if used
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import optuna
import torch
from optuna import Trial

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
from deeparguing.cli.parse_command_line import LOG_LEVELS

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.debug(f"Using device: {device}")


def run(trial: Trial | None = None):

    if trial is not None:
        logging.info(f"Starting Trial {trial.number}")

    model_config = read_config_files(args.config)

    total = 0
    f1s = []

    for seed_idx, seed in enumerate(args.seed):
        # TODO: Move this logic to separate function
        logging.info("=" * 100)
        logging.info(f"Running With Torch Seed: {seed}")
        torch.manual_seed(seed)

        data_dict, instances = parse_model_config(model_config, trial, device=device)

        assert data_dict
        assert instances

        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val"]

        model: GradualAACBR = instances["model"].to(device)

        trainer: Trainer = instances["trainer"]
        train_settings = instances["train_settings"]

        labels = data_dict["labels"]

        X_casebase, y_casebase = instances["build_casebase"](X_train, y_train)
        X_new_cases, y_new_cases = instances["build_new_cases"](
            X_train, y_train, X_casebase, y_casebase
        )

        X_defaults = X_train.mean(dim=0).tile(len(labels), 1)
        y_defaults = labels.flip([0])

        if plot_matrix_before:
            model.show_matrix()
        if plot_graph_before:
            model.show_graph()

        model.train()
        trainer.train(
            model,
            X_casebase,
            y_casebase,
            X_new_cases,
            y_new_cases,
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

            acc_test, prec_test, rec_test, f1_test, cm_test = evaluate_model(
                model, X_casebase, y_casebase, X_defaults, y_defaults, X_test, y_test
            )
            print_results(
                acc_test, prec_test, rec_test, f1_test, cm_test, "TEST", labels
            )

        if args.run_train:
            X_train = data_dict["X_train"]
            y_train = data_dict["y_train"]

            acc_train, prec_train, rec_train, f1_train, cm_train = evaluate_model(
                model, X_casebase, y_casebase, X_defaults, y_defaults, X_train, y_train
            )
            print_results(
                acc_train, prec_train, rec_train, f1_train, cm_train, "TRAIN", labels
            )

        if args.plot_loss:
            trainer.plot_loss_curve()
        if args.plot_gradients:
            trainer.plot_grads()
        if plot_matrix_after:
            model.show_matrix()
        if plot_graph_after:
            model.show_graph()

        logging.info("=" * 100)
        if f1 > 0.7:
            total += 1
        if len(args.seed) > 1:
            logging.info(f"F1 > 0.7 in {total}/{seed_idx + 1} seeds, which is {total / (seed_idx + 1) * 100}%")
        f1s.append(f1)
        logging.info(f"Average f1 score: {np.mean(f1s)}")

    return np.mean(f1s)


if __name__ == "__main__":

    args = parse_command_line()

    logging.basicConfig(
        level=LOG_LEVELS[args.log], format="%(asctime)s - %(levelname)s - %(message)s", force=True,
    )


    plot_matrix_before, plot_matrix_after, plot_graph_before, plot_graph_after = plots(
        args
    )

    if args.tuning:
        study = optuna.create_study(direction="maximize")
        study.optimize(run, n_trials=args.n_trials)
        logging.info("\nBest trial:")
        best = study.best_trial
        logging.info(f"Value: {best.value}")
        logging.info("Params:")
        for key, val in best.params.items():
            logging.info(f"  {key}: {val}")
    else:
        average_f1 = run()
