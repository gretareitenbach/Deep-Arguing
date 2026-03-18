import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from deeparguing.feature_extractor.simple_cnn import SimpleCNN
from deeparguing.helper import load_torch_images, split_data


BEST_MODEL_PATH = "best_simple_cnn.pt"
best_accuracy_global = 0.0   # updated across sweep runs


# -----------------------------------------------------------------------------
# TRAINING FUNCTION FOR W&B SWEEP
# -----------------------------------------------------------------------------
def train_sweep(config=None):
    global best_accuracy_global

    with wandb.init(config=config):
        config = wandb.config
        run_id = wandb.run.id

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ------------------------------
        # Load dataset
        # ------------------------------
        X, y, _, _ = load_torch_images("CIFAR10", device, shuffle=True)
        X_train, y_train, X_val, y_val, _, _ = split_data(X, y, seed=42)

        batch_size = config.batch_size

        # ------------------------------
        # Instantiate model (dropout added)
        # ------------------------------
        model = SimpleCNN(
            in_channels=3,
            output_features=10,
            dropout=config.dropout
        ).to(device)

        classification_head = nn.Linear(10, 10).to(device)

        all_params = list(model.parameters()) + list(classification_head.parameters())

        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        optimizer = optim.AdamW(
            all_params,
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # ------------------------------
        # Training loop
        # ------------------------------
        epochs = config.epochs
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            permutation = torch.randperm(n_samples, device=device)

            for i in range(0, n_samples, batch_size):
                idx = permutation[i:i + batch_size]
                xb = X_train[idx]
                yb = torch.argmax(y_train[idx], dim=1)

                optimizer.zero_grad()
                out = classification_head(F.relu(model(xb)))
                loss = criterion(out, yb)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(all_params, max_norm=config.grad_clip)
                optimizer.step()

            wandb.log({"train_loss": loss.item()})

        # ------------------------------
        # Validation Loop
        # ------------------------------
        model.eval()
        classification_head.eval()

        val_preds = []
        val_labels = []

        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                xb = X_val[i:i + batch_size]
                yb = y_val[i:i + batch_size]

                out = classification_head(F.relu(model(xb)))

                preds = torch.argmax(out, dim=1).cpu().numpy()
                labs = torch.argmax(yb, dim=1).cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labs)

        # Compute metrics
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, average="macro", zero_division=0)
        rec = recall_score(val_labels, val_preds, average="macro", zero_division=0)
        f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        wandb.log({
            "final_accuracy": acc,
            "final_precision": prec,
            "final_recall": rec,
            "final_f1": f1,
        })

        # ------------------------------
        # Save best model across sweep
        # ------------------------------
        if acc > best_accuracy_global:
            best_accuracy_global = acc
            print(f"\n✨ New best model found! Accuracy = {acc:.4f}")
            print(f"Saving best model to {BEST_MODEL_PATH}")

            torch.save(model.state_dict(), BEST_MODEL_PATH)
            wandb.save(BEST_MODEL_PATH)

        print(f"Run {run_id} accuracy = {acc:.4f} — Best so far = {best_accuracy_global:.4f}")


# -----------------------------------------------------------------------------
# LAUNCH SWEEP PROGRAMMATICALLY
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "final_accuracy", "goal": "maximize"},
        "parameters": {
            "lr": {"min": 0.0001, "max": 0.0005},
            "weight_decay": {"values": [0.0005, 0.001, 0.01]},
            "batch_size": {"values": [128, 256]},
            "dropout": {"values": [0.2, 0.3, 0.4]},
            "epochs": {"values": [15, 20, 30, 50, 100]},
            "label_smoothing": {"values": [0.0, 0.05]},
            "grad_clip": {"values": [0.5, 1.0, 5.0]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="deeparguing-cnn-tuning")
    wandb.agent(sweep_id, function=train_sweep, count=72)

