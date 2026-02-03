import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import Tensor
from tqdm import tqdm

import wandb
from deeparguing.feature_extractor.resnet import Resnet32
from deeparguing.helper import load_torch_images, split_data


def evaluate(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    batch_size: int,
    device: torch.device,
    classification_head: nn.Module | None = None,
):
    model.eval()
    if classification_head is not None:
        classification_head.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size]
            yb = y[i : i + batch_size]

            if classification_head is not None:
                features = model(xb, use_classification_head=False)
                logits = classification_head(features)
            else:
                logits = model(xb, use_classification_head=True)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labs = torch.argmax(yb, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, prec, rec, f1, cm


def train_linear_probe(
    model: nn.Module,
    linear_head: nn.Module,
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
):
    model.eval()
    linear_head.train()

    optimizer = optim.Adam(linear_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_samples = X_train.shape[0]

    pbar = tqdm(range(epochs), dynamic_ncols=True, desc="Linear Probe Training")
    for epoch in pbar:
        permutation = torch.randperm(n_samples, device=device)
        running_loss = 0.0

        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            inputs = X_train[indices]
            labels = torch.argmax(y_train[indices], dim=1)

            optimizer.zero_grad()

            with torch.no_grad():
                features = model(inputs, use_classification_head=False)

            logits = linear_head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / n_samples

        with torch.no_grad():
            val_loss = 0.0
            for i in range(0, X_val.shape[0], batch_size):
                xb = X_val[i : i + batch_size]
                yb = torch.argmax(y_val[i : i + batch_size], dim=1)
                features = model(xb, use_classification_head=False)
                logits = linear_head(features)
                val_loss += criterion(logits, yb).item() * xb.size(0)
            val_loss = val_loss / X_val.shape[0]

        wandb.log(
            {
                "linear_probe_train_loss": train_loss,
                "linear_probe_val_loss": val_loss,
                "linear_probe_epoch": epoch,
            }
        )

        pbar.set_description(
            f"Epoch {epoch}, Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )


def print_test_results(
    test_name: str,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    threshold: float,
):
    status = "PASS" if acc >= threshold else "FAIL"
    print(f"\n{test_name}")
    print("-" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Status:    {status} (threshold: {threshold * 100:.0f}%)")
    return status == "PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify a pretrained ResNet model on CIFAR-10"
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        required=True,
        help="Path to the pretrained ResNet weights (.pt file)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Dataset to use for verification (default: CIFAR10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation/training (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--linear-probe-epochs",
        type=int,
        default=20,
        help="Number of epochs for linear probe training (default: 20)",
    )
    parser.add_argument(
        "--linear-probe-lr",
        type=float,
        default=0.01,
        help="Learning rate for linear probe (default: 0.01)",
    )

    args = parser.parse_args()

    config = {
        "weights_path": args.weights_path,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "linear_probe_epochs": args.linear_probe_epochs,
        "linear_probe_lr": args.linear_probe_lr,
    }

    wandb.init(project="deeparguing-resnet-verify", config=config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    X, y = load_torch_images(args.dataset, device, shuffle=True, seed=args.seed)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, seed=args.seed)

    num_classes = y.shape[1]
    feature_dim = 64

    model = Resnet32(
        num_classes=num_classes,
        weights_path=args.weights_path,
        freeze_weights=False,
    )
    model = model.to(device)

    print("=" * 60)
    print("ResNet Verification Results")
    print("=" * 60)
    print(f"Weights: {args.weights_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device:  {device}")

    # Test 1: Direct classification with built-in head
    acc1, prec1, rec1, f11, cm1 = evaluate(model, X_val, y_val, args.batch_size, device)

    wandb.log(
        {
            "direct_accuracy": acc1,
            "direct_precision": prec1,
            "direct_recall": rec1,
            "direct_f1": f11,
        }
    )

    test1_pass = print_test_results(
        "TEST 1: Direct Classification (with built-in head)",
        acc1,
        prec1,
        rec1,
        f11,
        threshold=0.85,
    )

    # Test 2: Linear probe with frozen features
    for p in model.parameters():
        p.requires_grad = False

    linear_head = nn.Linear(feature_dim, num_classes).to(device)

    train_linear_probe(
        model,
        linear_head,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.linear_probe_epochs,
        lr=args.linear_probe_lr,
        batch_size=args.batch_size,
        device=device,
    )

    acc2, prec2, rec2, f12, cm2 = evaluate(
        model, X_val, y_val, args.batch_size, device, classification_head=linear_head
    )

    wandb.log(
        {
            "linear_probe_accuracy": acc2,
            "linear_probe_precision": prec2,
            "linear_probe_recall": rec2,
            "linear_probe_f1": f12,
        }
    )

    test2_pass = print_test_results(
        "TEST 2: Linear Probe (frozen features + trained linear head)",
        acc2,
        prec2,
        rec2,
        f12,
        threshold=0.80,
    )

    print("\n" + "=" * 60)
    overall = "PASS" if (test1_pass and test2_pass) else "FAIL"
    print(f"Overall: {overall}")
    print("=" * 60)

    wandb.log({"overall_pass": test1_pass and test2_pass})
    wandb.finish()
