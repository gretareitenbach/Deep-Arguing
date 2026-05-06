import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from deeparguing.feature_extractor.simple_cnn import SimpleCNN
from deeparguing.helper import load_torch_images, split_data

# ------------------------------
# HYPERPARAMETERS
# ------------------------------
config = {
    "dataset": "MNIST",
    "batch_size": 256,
    "dropout": 0.2,
    "output_features": 64,
    "seed": 42,
    "epochs": 10,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "grad_clip": 5.0,
    "in_channels": 1
}

wandb.init(project=f"deeparguing-cnn-{config["dataset"]}", config=config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config["seed"])

# ------------------------------
# Load data
# ------------------------------
X, y, _, _ = load_torch_images(config["dataset"], device, shuffle=True)
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, seed=config["seed"])

batch_size = config["batch_size"]

# ------------------------------
# Model setup
# ------------------------------
model = SimpleCNN(in_channels=config["in_channels"], output_features=config["output_features"], dropout = config["dropout"]).to(device)
classification_head = nn.Linear(config["output_features"], 10).to(device)

all_params = list(model.parameters()) + list(classification_head.parameters())

criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
optimizer = optim.AdamW(
    all_params,
    lr=config["lr"],
    weight_decay=config["weight_decay"],
)

epochs = config["epochs"]
n_samples = X_train.shape[0]

# ------------------------------
# Training loop
# ------------------------------
pbar = tqdm(range(epochs), dynamic_ncols=True)

for epoch in pbar:
    permutation = torch.randperm(n_samples, device=device)

    for i in range(0, n_samples, batch_size):
        indices = permutation[i: i + batch_size]
        inputs = X_train[indices]
        labels = torch.argmax(y_train[indices], dim=1)

        optimizer.zero_grad()
        outputs = classification_head(F.relu(model(inputs)))
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            all_params,
            max_norm=config["grad_clip"],
            error_if_nonfinite=False,
        )

        optimizer.step()

    # Log training loss
    with torch.no_grad():
        total_loss = 0
        for i in range(0, X_val.shape[0], batch_size):
            xb = X_val[i : i + batch_size]
            yb = y_val[i : i + batch_size]
    
            out = classification_head(F.relu(model(xb)))
            total_loss += criterion(out, yb).item() * len(xb)
        avg_loss = total_loss / len(X_val)

    wandb.log({"train_loss": loss.item(), "val_loss": avg_loss})

    pbar.set_description(f"Epoch {epoch}, Loss: {round(loss.item(), 6)}, Val loss: {round(avg_loss, 6)}")

print("Finished Training")

# ==================================================================
# VALIDATION (BATCHED) + FINAL METRICS (LOGGED ONCE)
# ==================================================================
model.eval()
classification_head.eval()

val_preds = []
val_labels = []

with torch.no_grad():
    for i in range(0, X_val.shape[0], batch_size):
        xb = X_val[i : i + batch_size]
        yb = y_val[i : i + batch_size]

        out = classification_head(F.relu(model(xb)))

        preds = torch.argmax(out, dim=1).cpu().numpy()
        labs = torch.argmax(yb, dim=1).cpu().numpy()

        val_preds.extend(preds)
        val_labels.extend(labs)

# Sklearn metrics
acc = accuracy_score(val_labels, val_preds)
prec = precision_score(val_labels, val_preds, average="macro", zero_division=0)
rec = recall_score(val_labels, val_preds, average="macro", zero_division=0)
f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
cm = confusion_matrix(val_labels, val_preds)

# Log final metrics ONCE (after training)
wandb.log({
    "final_accuracy": acc,
    "final_precision": prec,
    "final_recall": rec,
    "final_f1": f1,
})

# Confusion matrix plot
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=val_labels,
    preds=val_preds,
)})


save_path = "simple_cnn_64.pt"
torch.save(model.state_dict(), save_path)

wandb.finish()


