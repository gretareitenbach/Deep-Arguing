"""Pretrain the LSTM sequence embedding used as the frozen ``feature_weights``
extractor in the brainwear AACBR model -- mirrors ``pretrain_resnet.py``'s
role: train a plain classifier on the windowed data, then keep only the
embedding backbone.

``LSTMFeatureExtractor`` has no built-in classification head (unlike
``ResNetCIFAR``'s ``use_classification_head``), so this script bolts on a
throwaway ``nn.Linear`` head, trains both jointly as a 5-way activity
classifier on ``data/brainwear/brainwear_trainval.csv``, then saves only the
LSTM's ``state_dict()`` -- the richer embedding (``--embedding-size``, wider
than the 5-class output) survives; the head does not.

Loads via the same ``load_tabular_data``/``train_test_split(test_size=0.2,
random_state=42)`` calls ``load_data_dict``'s ``tabular``+``test_path``
branch uses, so the train/val split and feature scaling here exactly match
what the downstream AACBR run will see -- the frozen embedding is never
handed out-of-distribution inputs relative to what it was pretrained on.

Usage::

    python -m scripts.pretrain_lstm
    python -m scripts.pretrain_lstm --epochs 20 --hidden-size 64 --embedding-size 32
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from deeparguing.feature_extractor.lstm import LSTMFeatureExtractor
from deeparguing.helper import load_tabular_data
from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import output_path, today_output_dir

DEFAULT_TRAINVAL_PATH = "data/brainwear/brainwear_trainval.csv"
DEFAULT_WINDOW_EPOCHS = 15  # winner of the window-size sweep, see tuning/brainwear/hyperparameters_brainwear.yaml
DEFAULT_CHECKPOINT_NAME = "lstm_brainwear.pt"

# Categorical palette, slot 1 -- see dataviz skill's reference palette.
_PLOT_SERIES = "#2a78d6"
_PLOT_SURFACE = "#fcfcfb"
_PLOT_GRIDLINE = "#e1e0d9"
_PLOT_AXIS = "#c3c2b7"
_PLOT_TICK_INK = "#898781"
_PLOT_PRIMARY_INK = "#0b0b0b"
_PLOT_SECONDARY_INK = "#52514e"


def class_weights(y_idx: np.ndarray, num_classes: int) -> Tensor:
    """Inverse-frequency weights (sklearn's 'balanced' formula) -- without
    this, cross-entropy on this label distribution (~51% sleep, ~2.5%
    moderate) mostly just learns to predict the majority classes."""
    counts = np.bincount(y_idx, minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    embedder: LSTMFeatureExtractor, head: nn.Module,
    X_train: Tensor, y_train_idx: Tensor,
    optimizer: optim.Optimizer, criterion: nn.Module, batch_size: int,
) -> float:
    embedder.train()
    head.train()
    n = X_train.shape[0]
    permutation = torch.randperm(n, device=X_train.device)
    running_loss = 0.0

    for i in range(0, n, batch_size):
        indices = permutation[i : i + batch_size]
        inputs = X_train[indices]
        labels = y_train_idx[indices]

        optimizer.zero_grad()
        logits = head(embedder(inputs))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / n


@torch.no_grad()
def evaluate(
    embedder: LSTMFeatureExtractor, head: nn.Module,
    X: Tensor, y_idx: Tensor, criterion: nn.Module, batch_size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    embedder.eval()
    head.eval()
    running_loss = 0.0
    preds, labels = [], []

    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size]
        yb = y_idx[i : i + batch_size]
        logits = head(embedder(xb))
        running_loss += (criterion(logits, yb) * xb.size(0)).item()
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(yb.cpu().numpy())

    loss = running_loss / X.shape[0]
    return loss, np.concatenate(preds), np.concatenate(labels)


def _plot_per_class_f1(
    label_names: list[str], f1_per_class: np.ndarray, png_path
) -> None:
    x = np.arange(len(label_names))

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=_PLOT_SURFACE)
    ax.set_facecolor(_PLOT_SURFACE)

    bars = ax.bar(x, f1_per_class, width=0.5, color=_PLOT_SERIES, zorder=3)
    for bar, val in zip(bars, f1_per_class):
        ax.annotate(
            f"{val:.2f}", (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=9, color=_PLOT_SECONDARY_INK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(label_names, color=_PLOT_PRIMARY_INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Val F1", color=_PLOT_PRIMARY_INK)
    ax.set_title("Pretrained LSTM: per-class val F1", color=_PLOT_PRIMARY_INK, fontsize=11)
    ax.grid(True, axis="y", color=_PLOT_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_PLOT_AXIS)
    ax.spines["bottom"].set_color(_PLOT_AXIS)
    ax.tick_params(colors=_PLOT_TICK_INK)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, facecolor=_PLOT_SURFACE)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainval-path", default=DEFAULT_TRAINVAL_PATH)
    parser.add_argument("--window-epochs", type=int, default=DEFAULT_WINDOW_EPOCHS)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=32,
                         help="LSTMFeatureExtractor's output_features -- the "
                              "embedding handed to the AACBR model, wider "
                              "than num_classes so it survives as a real "
                              "embedding rather than collapsing to logits.")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2,
                         help="Applied only to the throwaway classification "
                              "head, not saved with the embedder.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--md-log-path", default="", help="Empty string for the default dated path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    logging.info(f"Loading {args.trainval_path}")
    X, y, state = load_tabular_data(
        path=args.trainval_path,
        target_field=args.window_epochs,
        has_header=True,
        continuous_cols=list(range(args.window_epochs)),
        device=str(device),
    )
    # Matches load_data_dict's tabular + test_path branch exactly, so the
    # train/val rows here are the same ones the downstream AACBR run gets.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    label_names = list(state["target_encoder"].categories_[0])
    num_classes = len(label_names)
    y_train_idx = y_train.argmax(dim=1)
    y_val_idx = y_val.argmax(dim=1)
    logging.info(f"train={X_train.shape[0]:,} val={X_val.shape[0]:,} classes={label_names}")

    embedder = LSTMFeatureExtractor(
        input_size=1, hidden_size=args.hidden_size,
        output_features=args.embedding_size, num_layers=args.num_layers,
    ).to(device)
    head = nn.Sequential(
        nn.Dropout(args.dropout), nn.Linear(args.embedding_size, num_classes)
    ).to(device)

    weights = class_weights(y_train_idx.cpu().numpy(), num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        list(embedder.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    pbar = tqdm(range(args.epochs), dynamic_ncols=True)
    for epoch in pbar:
        train_loss = train_epoch(
            embedder, head, X_train, y_train_idx, optimizer, criterion, args.batch_size
        )
        val_loss, val_preds, val_labels = evaluate(
            embedder, head, X_val, y_val_idx, criterion, args.batch_size
        )
        val_acc = (val_preds == val_labels).mean()
        pbar.set_description(
            f"Epoch {epoch}, Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
        )
    logging.info("Training complete.")

    _, val_preds, val_labels = evaluate(embedder, head, X_val, y_val_idx, criterion, args.batch_size)
    report = classification_report(
        val_labels, val_preds, target_names=label_names, zero_division=0, digits=3
    )
    cm = confusion_matrix(val_labels, val_preds)
    cm_str = "Confusion matrix (rows=true, cols=pred):\n" + "\n".join(
        f"{label_names[i]:>12}: " + " ".join(f"{v:6d}" for v in row)
        for i, row in enumerate(cm)
    )
    f1_per_class = f1_score(val_labels, val_preds, average=None, zero_division=0)
    logging.info("\n" + report)
    logging.info("\n" + cm_str)

    save_path = output_path(args.checkpoint_name)
    torch.save(embedder.state_dict(), save_path)
    logging.info(f"Wrote {save_path}")

    png_path = output_path("lstm_brainwear_per_class_f1.png")
    _plot_per_class_f1(label_names, f1_per_class, png_path)
    logging.info(f"Wrote {png_path}")

    if args.md_log_path != "":
        md_path = args.md_log_path or str(today_output_dir() / "pretrain_lstm.md")
        lines = [
            "--- LSTM PRETRAINING ---",
            f"hidden_size={args.hidden_size}, embedding_size={args.embedding_size}, "
            f"num_layers={args.num_layers}, epochs={args.epochs}, lr={args.lr}",
            f"train={X_train.shape[0]:,}, val={X_val.shape[0]:,}, "
            f"final val acc={ (val_preds == val_labels).mean():.4f}",
            "--- CLASSIFICATION REPORT ---",
            report,
            "--- " + cm_str.split(chr(10))[0].upper() + " ---",
            "\n".join(cm_str.split(chr(10))[1:]),
            f"Saved embedding to {save_path}",
        ]
        write_markdown_log(lines, md_path, mode="w")
        logging.info(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
