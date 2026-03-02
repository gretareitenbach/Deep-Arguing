import logging
import os

from optuna.samplers import TPESampler

# This will ensure determinism of KMEANS Clustering and GPU/CUDA operations
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import uuid

import optuna
import torch
from optuna import Trial
from torch.nn.utils import parameters_to_vector

from deeparguing import GradualAACBR
from deeparguing.base_scores import *
from deeparguing.casebase_edge_weights import *
from deeparguing.cli import (parse_command_line, parse_model_config, plots,
                             read_config_files)
from deeparguing.cli.loggers import DummyLogger, ExperimentLogger, WandbLogger
from deeparguing.cli.parse_command_line import LOG_LEVELS
from deeparguing.clustering import *
from deeparguing.evals import (evaluate_model, print_results,
                               visualize_overlayed_loss_landscapes)
from deeparguing.feature_extractor import *
from deeparguing.helper import *
from deeparguing.irrelevance_edge_weights import *
from deeparguing.losses import *
from deeparguing.regularisers import *
from deeparguing.semantics import *
from deeparguing.t_norm import *
from deeparguing.train import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.debug(f"Using device: {device}")

# Ensure determinism for GPU/cuDNN operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def run(project: str = "gradual-aa-cbr"):
    def objective(trial: Trial | None = None):

        ExperimentLogger.set_current(experiment_logger)

        model_config = read_config_files(args.config)

        total = 0
        f1s = []

        trial_id = uuid.uuid4()

        for seed_idx, seed in enumerate(args.seed):

            if trial is not None:
                experiment_logger.init(
                    project=project,
                    group=f"trial-{trial.number}",
                    config={"trial_id": f"{trial_id}_{trial.number}", "seed": seed},
                )
                logging.info(f"Starting Trial {trial.number}")
            else:
                experiment_logger.init(
                    project=project,
                    group=f"trial_{trial_id}",
                    config={"trial_id": f"{trial_id}_-1", "seed": seed},
                )

            for path in args.config:
                filename = os.path.splitext(os.path.basename(path))[0]
                ExperimentLogger.current().log_artifact(filename, path, type="config")

            logging.info("=" * 100)
            logging.info(f"Running With Torch Seed: {seed}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            data_dict, instances = parse_model_config(
                model_config, trial, device=device
            )

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

            n_dims = X_train.dim()
            tile_shape = [1] * n_dims
            tile_shape[0] = len(labels)
            X_defaults = X_train.mean(dim=0).tile(tile_shape)
            y_defaults = labels.flip([0])

            if plot_matrix_before:
                model.show_matrix()
            if plot_graph_before:
                model.show_graph()

            if args.visualise_loss_landscape:
                theta_pre = parameters_to_vector(model.parameters()).clone().detach()

            model.train()
            max_val_acc = trainer.train(
                model,
                X_casebase,
                y_casebase,
                X_new_cases,
                y_new_cases,
                X_defaults,
                y_defaults,
                **train_settings,
                disable_tqdm=args.disable_tqdm,
                X_val=X_val,
                y_val=y_val,
                log_val_loss=args.log_val_loss,
                log_gradients=args.log_gradients,
            )

            batch_size = train_settings.get("batch_size", None)

            model.eval()

            acc, prec, rec, f1, cm = evaluate_model(
                model,
                X_casebase,
                y_casebase,
                X_defaults,
                y_defaults,
                X_val,
                y_val,
                batch_size=batch_size,
            )
            print_results(acc, prec, rec, f1, cm, "VALIDATION", labels)
            ExperimentLogger.current().log_metrics(
                {
                    "seed": seed,
                    "val_accuracy": acc,
                    "val_precision": prec,
                    "val_recall": rec,
                    "val_f1": f1,
                },
            )

            if args.run_test:
                X_test = data_dict["X_test"]
                y_test = data_dict["y_test"]

                acc_test, prec_test, rec_test, f1_test, cm_test = evaluate_model(
                    model,
                    X_casebase,
                    y_casebase,
                    X_defaults,
                    y_defaults,
                    X_test,
                    y_test,
                    batch_size=batch_size,
                )
                print_results(
                    acc_test, prec_test, rec_test, f1_test, cm_test, "TEST", labels
                )

            if args.run_train:
                X_train = data_dict["X_train"]
                y_train = data_dict["y_train"]

                acc_train, prec_train, rec_train, f1_train, cm_train = evaluate_model(
                    model,
                    X_casebase,
                    y_casebase,
                    X_defaults,
                    y_defaults,
                    X_train,
                    y_train,
                    batch_size=batch_size,
                )
                print_results(
                    acc_train,
                    prec_train,
                    rec_train,
                    f1_train,
                    cm_train,
                    "TRAIN",
                    labels,
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
                logging.info(
                    f"F1 > 0.7 in {total}/{seed_idx + 1} seeds, which is {total / (seed_idx + 1) * 100}%"
                )
            f1s.append(f1)
            logging.info(f"Average f1 score: {np.mean(f1s)}")

            if args.visualise_loss_landscape:
                visualize_overlayed_loss_landscapes(
                    model,
                    train_settings["criterion_factory"](),
                    X_train,
                    y_train,
                    X_casebase,
                    y_casebase,
                    X_defaults,
                    y_defaults,
                    theta_pre,
                    train_settings["regulariser"],
                    steps=30,
                )
            ExperimentLogger.current().finish()
        average_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        logging.info(f"Average F1: {average_f1}")
        logging.info(f"F1 STD: {std_f1}")
        # return average_f1
        return max_val_acc

    return objective


if __name__ == "__main__":

    args = parse_command_line()

    logging.basicConfig(
        level=LOG_LEVELS[args.log],
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    plot_matrix_before, plot_matrix_after, plot_graph_before, plot_graph_after = plots(
        args
    )

    if args.experiment_logger == "none":
        experiment_logger = DummyLogger()
    elif args.experiment_logger == "wandb":
        experiment_logger = WandbLogger()
    else:
        raise ValueError(f"No logger implementation for: {args.experiment_logger}")

    if args.tuning:
        study = optuna.create_study(direction="maximize", sampler=TPESampler())
        study.optimize(run(args.project), n_trials=args.n_trials)
        logging.info("\nBest trial:")
        best = study.best_trial
        logging.info(f"Value: {best.value}")
        logging.info("Params:")
        for key, val in best.params.items():
            logging.info(f"  {key}: {val}")
    else:
        average_f1 = run(args.project)()
