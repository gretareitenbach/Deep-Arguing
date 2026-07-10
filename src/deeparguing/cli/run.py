import logging
import os
from pathlib import Path

from optuna.samplers import TPESampler

# This will ensure determinism of KMEANS Clustering and GPU/CUDA operations
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
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
from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.criterion import *
from deeparguing.evals import (evaluate_model, print_results,
                               visualize_overlayed_loss_landscapes)
from deeparguing.feature_extractor import *
from deeparguing.helper import *
from deeparguing.irrelevance_edge_weights import *
from deeparguing.criterion.losses import *
from deeparguing.criterion import *
from deeparguing.models import *
from deeparguing.criterion.regularisers import *
from deeparguing.semantics import *
from deeparguing.t_norm import *
from deeparguing.train import Trainer
from deeparguing.train.neural_trainer import NeuralTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.debug(f"Using device: {device}")

# Ensure determinism for GPU/cuDNN operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def write_markdown_summary(lines: list[str], mode: str = "w") -> None:
    """Render CLI summary log lines as markdown in outputs/summary.md.

    Lines of the form "--- X ---" become level-2 headings; everything else
    becomes a bullet point. ``mode="a"`` appends -- used for the tuning
    best-trial block, which is logged after the per-trial summary this
    function is also called for.
    """
    Path("outputs").mkdir(parents=True, exist_ok=True)
    rendered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("---") and stripped.endswith("---"):
            rendered.append(f"## {stripped.strip('- ').strip()}")
        else:
            rendered.append(f"- {stripped}")

    with open("outputs/summary.md", mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# Run Summary\n\n")
        f.write("\n".join(rendered) + "\n\n")


def run(project: str = "gradual-aa-cbr"):
    def objective(trial: Trial | None = None):

        ExperimentLogger.set_current(experiment_logger)

        model_config = read_config_files(args.config)

        total = 0
        val_f1s = []
        val_accs = []
        max_val_accs = []
        max_val_f1s = []
        test_f1s = []
        test_accs = []
        train_f1s = []
        train_accs = []
        grae_magnitudes_per_seed = []

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
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
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

            # TODO: Handle this differently?
            model_instance = instances["model"]
            if hasattr(model_instance, "to"):
                model = model_instance.to(device)
            else:
                model = model_instance

            trainer: Trainer = instances["trainer"]
            batch_size = getattr(trainer, "batch_size", None)

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

            # acc, prec, rec, f1, cm = evaluate_model(
            #     model,
            #     X_casebase,
            #     y_casebase,
            #     X_defaults,
            #     y_defaults,
            #     X_val,
            #     y_val,
            #     batch_size=batch_size,
            # )
            # print_results(acc, prec, rec, f1, cm, "VALIDATION PRE TRAINING", labels)

            if plot_matrix_before:
                model.fit(X_casebase, y_casebase, X_defaults, y_defaults)
                model.show_matrix()
            if plot_graph_before:
                model.fit(X_casebase, y_casebase, X_defaults, y_defaults)
                model.show_graph()

            if args.visualise_loss_landscape:
                theta_pre = parameters_to_vector(model.parameters()).clone().detach()

            if args.json_out:
                OUT_DIR = "outputs"
                Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
                model.fit(X_casebase, y_casebase, X_defaults, y_defaults)
                image_mean = data_dict.get("image_mean", None)
                image_std = data_dict.get("image_std", None)
                model.export_to_json(
                    f"{OUT_DIR}/pre_training_{args.json_out}.json",
                    image_mean=image_mean,
                    image_std=image_std,
                    new_cases=X_new_cases[: args.num_new_vis],
                    new_cases_labels=y_new_cases[: args.num_new_vis],
                )

            model.train()
            if isinstance(trainer, NeuralTrainer):
                trainer.set_logging_flags(args.log_val_loss, args.log_gradients)

            max_val_acc, max_val_f1 = trainer.train(
                model,
                X_casebase,
                y_casebase,
                X_new_cases,
                y_new_cases,
                X_defaults,
                y_defaults,
                disable_tqdm=args.disable_tqdm,
                X_val=X_val,
                y_val=y_val,
            )

            model.eval()

            # Persist the trained model + fitted casebase so a separate
            # process can reload it and run e.g. counterfactuals/contest.py
            # against a real sample -- state_dict() alone misses
            # model.A/X_train/default_indexes, since fit() sets those as
            # plain attributes, not buffers. Saved unconditionally (not just
            # under --misclassified_log) so a checkpoint is always available.
            OUT_DIR = "outputs"
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
            checkpoint_path = f"{OUT_DIR}/model_checkpoint.pt"
            torch.save(
                {
                    "config_paths": args.config,
                    "state_dict": model.state_dict(),
                    "A": model.A.detach().cpu(),
                    "X_train": model.X_train.detach().cpu(),
                    "y_train": model.y_train.detach().cpu(),
                    "default_indexes": model.default_indexes.detach().cpu(),
                },
                checkpoint_path,
            )
            logging.info(
                f"Saved model checkpoint (weights + fitted casebase) to {checkpoint_path}"
            )

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
                    "evals/val_accuracy": acc,
                    "evals/val_precision": prec,
                    "evals/val_recall": rec,
                    "evals/val_f1": f1,
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
                test_f1s.append(f1_test)
                test_accs.append(acc_test)

                if args.misclassified_log:
                    # ======================================================
                    # MISCLASSIFIED SAMPLE & QBAF TENSOR EXTRACTION
                    # ======================================================
                    logging.info("Isolating misclassified test samples...")
                    model.eval()

                    all_preds = []
                    # Batched inference to prevent Out-Of-Memory errors
                    current_batch_size = batch_size if batch_size is not None else len(X_test)
                    for i in range(0, len(X_test), current_batch_size):
                        batch_preds = model(X_test[i : i + current_batch_size]).cpu().detach().numpy()
                        all_preds.append(batch_preds)

                    y_predicted = np.concatenate(all_preds, axis=0)
                    y_predicted_classes = np.argmax(y_predicted, axis=1)
                    y_true_classes = np.argmax(y_test.cpu().detach().numpy(), axis=1)

                    # Find indices where the model prediction does not match the ground truth
                    misclassified_indices = np.where(y_predicted_classes != y_true_classes)[0]

                    # Isolate up to 100 samples
                    num_to_extract = min(100, len(misclassified_indices))
                    selected_indices = misclassified_indices[:num_to_extract]

                    X_misc = X_test[selected_indices]
                    y_misc = y_test[selected_indices]

                    OUT_DIR = "outputs"
                    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

                    grae_result = None
                    if args.grae_log:
                        # ==================================================
                        # G-RAE (GRADIENT-BASED RELATION ATTRIBUTION)
                        # ==================================================
                        logging.info(
                            f"Computing G-RAEs for {num_to_extract} misclassified samples..."
                        )

                        # default_indexes rows are ordered the same as
                        # X_defaults/y_defaults ("labels.flip([0])"), but
                        # "labels" itself (torch.unique(y, dim=0)) is already
                        # in reverse-class order for one-hot rows, so the two
                        # reversals cancel out: default row for class c sits
                        # at c directly (verified against a real checkpoint's
                        # y_train[default_indexes] -- no offset needed).
                        true_classes_misc = y_true_classes[selected_indices]
                        target_indices = true_classes_misc.tolist()

                        grae_result = compute_grae(
                            model, X_misc, target_indices, per_sample=True
                        )

                        grae_export_path = f"{OUT_DIR}/misclassified_grae.pt"
                        torch.save(
                            {
                                "casebase_edges": grae_result.casebase_edges,
                                "new_case_edges": grae_result.new_case_edges,
                                "target_indices": grae_result.target_indices,
                                "selected_indices": selected_indices,
                            },
                            grae_export_path,
                        )
                        logging.info(
                            f"Successfully exported G-RAEs for {num_to_extract} "
                            f"misclassified samples to {grae_export_path}"
                        )

                        # Break down G-RAE magnitude by edge type: sign of the
                        # underlying adjacency entry (model.A / model.new_cases_attacks_adjacency)
                        # determines attack (negative) vs support (positive); zero entries are
                        # non-edges and excluded from both buckets.
                        def _mean_abs_grae(grae_values: torch.Tensor, mask: torch.Tensor) -> float:
                            mask = mask.expand_as(grae_values)
                            selected = grae_values.abs()[mask]
                            return selected.mean().item() if selected.numel() > 0 else float("nan")

                        A_signs = model.A.detach()
                        new_case_signs = model.new_cases_attacks_adjacency.detach()

                        grae_magnitudes = {
                            "evals/grae_casebase_attack_magnitude": _mean_abs_grae(
                                grae_result.casebase_edges, A_signs < 0
                            ),
                            "evals/grae_casebase_support_magnitude": _mean_abs_grae(
                                grae_result.casebase_edges, A_signs > 0
                            ),
                            "evals/grae_new_case_attack_magnitude": _mean_abs_grae(
                                grae_result.new_case_edges, new_case_signs < 0
                            ),
                            "evals/grae_new_case_support_magnitude": _mean_abs_grae(
                                grae_result.new_case_edges, new_case_signs > 0
                            ),
                        }
                        logging.info(
                            "Average G-RAE magnitude per edge type -- "
                            f"self.A attacks: {grae_magnitudes['evals/grae_casebase_attack_magnitude']:.6g}, "
                            f"self.A supports: {grae_magnitudes['evals/grae_casebase_support_magnitude']:.6g}, "
                            f"new-case attacks: {grae_magnitudes['evals/grae_new_case_attack_magnitude']:.6g}, "
                            f"new-case supports: {grae_magnitudes['evals/grae_new_case_support_magnitude']:.6g}"
                        )
                        ExperimentLogger.current().log_metrics(grae_magnitudes)
                        grae_magnitudes_per_seed.append(grae_magnitudes)
                        # ==================================================

                    # Export to JSON
                    export_path = f"{OUT_DIR}/misclassified_qbaf.json"

                    image_mean = data_dict.get("image_mean", None)
                    image_std = data_dict.get("image_std", None)

                    # This triggers a forward pass internally to populate new_cases_base_scores
                    # and new_cases_attacks_adjacency before exporting
                    model.export_to_json(
                        export_path,
                        image_mean=image_mean,
                        image_std=image_std,
                        new_cases=X_misc,
                        new_cases_labels=y_misc,
                        grae_casebase_edges=(
                            grae_result.casebase_edges if grae_result else None
                        ),
                        grae_new_case_edges=(
                            grae_result.new_case_edges if grae_result else None
                        ),
                        grae_target_indices=(
                            grae_result.target_indices if grae_result else None
                        ),
                    )
                    logging.info(f"Successfully exported {num_to_extract} misclassified samples and their QBAF tensors to {export_path}")
                    # ======================================================

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
                train_f1s.append(f1_train)
                train_accs.append(acc_train)

            if args.plot_loss and isinstance(trainer, NeuralTrainer):
                trainer.plot_loss_curve()
            if args.plot_gradients and isinstance(trainer, NeuralTrainer):
                trainer.plot_grads()
            if plot_matrix_after:
                model.show_matrix()
            if plot_graph_after:
                model.show_graph()

            logging.info("=" * 100)
            if f1 > 0.7:
                total += 1
            if len(args.seed) > 1:
                # Count how many seeds f1 > 0.7 - informal way to check for model collapse
                logging.debug(
                    f"F1 > 0.7 in {total}/{seed_idx + 1} seeds, which is {total / (seed_idx + 1) * 100}%"
                )
            val_f1s.append(f1)
            val_accs.append(acc)
            max_val_accs.append(max_val_acc)
            max_val_f1s.append(max_val_f1)
            logging.info(f"Average validation f1 score: {np.mean(val_f1s)}")

            if args.visualise_loss_landscape and isinstance(trainer, NeuralTrainer):
                visualize_overlayed_loss_landscapes(
                    model,
                    trainer.criterion,  # Assuming it holds the instance itself or a similar structure
                    X_train,
                    y_train,
                    X_casebase,
                    y_casebase,
                    X_defaults,
                    y_defaults,
                    theta_pre,
                    trainer.regulariser,
                    steps=30,
                )
            ExperimentLogger.current().finish()

            if args.json_out:
                model.fit(X_casebase, y_casebase, X_defaults, y_defaults)
                image_mean = data_dict.get("image_mean", None)
                image_std = data_dict.get("image_std", None)
                model.export_to_json(
                    f"{OUT_DIR}/post_training_{args.json_out}.json",
                    image_mean=image_mean,
                    image_std=image_std,
                    new_cases=X_new_cases[: args.num_new_vis],
                    new_cases_labels=y_new_cases[: args.num_new_vis],
                )

        average_max_val_acc = np.mean(max_val_accs)
        average_max_val_f1 = np.mean(max_val_f1s)

        summary_lines = [
            "--- VALIDATION RESULTS ---",
            f"Average Val Acc: {np.mean(val_accs)}",
            f"Val F1 STD: {np.std(val_accs)}",
            f"Average Val F1: {np.mean(val_f1s)}",
            f"Val F1 STD: {np.std(val_f1s)}",
            f"Average Max Val Acc: {average_max_val_acc}",
            f"Average Max Val F1: {average_max_val_f1}",
        ]

        if args.run_train and train_f1s:
            summary_lines += [
                "--- TRAIN RESULTS ---",
                f"Average Train Acc: {np.mean(train_accs)}",
                f"Train Acc STD: {np.std(train_accs)}",
                f"Average Train F1: {np.mean(train_f1s)}",
                f"Train F1 STD: {np.std(train_f1s)}",
            ]

        if args.run_test and test_f1s:
            summary_lines += [
                "--- TEST RESULTS ---",
                f"Average Test Acc: {np.mean(test_accs)}",
                f"Test Acc STD: {np.std(test_accs)}",
                f"Average Test F1: {np.mean(test_f1s)}",
                f"Test F1 STD: {np.std(test_f1s)}",
            ]

        if grae_magnitudes_per_seed:
            grae_labels = {
                "evals/grae_casebase_attack_magnitude": "Average self.A attack G-RAE magnitude",
                "evals/grae_casebase_support_magnitude": "Average self.A support G-RAE magnitude",
                "evals/grae_new_case_attack_magnitude": "Average new-case attack G-RAE magnitude",
                "evals/grae_new_case_support_magnitude": "Average new-case support G-RAE magnitude",
            }
            summary_lines += ["--- GRAE RESULTS ---"]
            for key, label in grae_labels.items():
                values = [seed_mags[key] for seed_mags in grae_magnitudes_per_seed]
                mean_value = np.nanmean(values) if not np.all(np.isnan(values)) else float("nan")
                summary_lines.append(f"{label}: {mean_value}")

        for line in summary_lines:
            logging.info(line)
        write_markdown_summary(summary_lines)

        if args.ht_obj == "f1":
            return average_max_val_f1
        else:
            return average_max_val_acc

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
        best_trial_lines = ["--- BEST TRIAL ---", f"Value: {best.value}", "Params:"]
        for key, val in best.params.items():
            logging.info(f"  {key}: {val}")
            best_trial_lines.append(f"{key}: {val}")
        write_markdown_summary(best_trial_lines, mode="a")
    else:
        average_f1 = run(args.project)()
