from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from deeparguing.helper import load_torch_images, split_data
from deeparguing.feature_extractor.resnet import ResNetCIFAR, Resnet32


def train(model: ResNetCIFAR, X_train: Tensor, y_train: Tensor, X_val: Tensor, y_val: Tensor, config: Any):

    n_samples: int = X_train.shape[0]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])

    batch_size: int = config["batch_size"]

    # Standard CIFAR LR schedule
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    pbar = tqdm(range(config["epochs"]), dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        permutation = torch.randperm(n_samples, device=device)

        for i in range(0, n_samples, batch_size):
            indices = permutation[i: i + batch_size]
            inputs = X_train[indices]
            labels = torch.argmax(y_train[indices], dim=1)

            optimizer.zero_grad()
            logits = model(inputs, use_classification_head = True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        train_loss = running_loss / len(X_train)

        with torch.no_grad():
            val_loss = 0.0
            for i in range(0, X_val.shape[0], batch_size):
                xb = X_val[i : i + batch_size]
                yb = torch.argmax(y_val[i : i + batch_size], dim=1)

                logits = model(xb, use_classification_head = True)
                val_loss += (criterion(logits, yb) * xb.size(0)).item()

            val_loss = val_loss / len(X_val)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        pbar.set_description(f"Epoch {epoch}, Loss: {round(train_loss, 6)}, Val loss: {round(val_loss, 6)}")

    print("Training complete.")


if __name__ == "__main__":
    config = {
        "dataset": "CIFAR10",
        "batch_size": 128,
        "seed": 42,
        "epochs": 200,
        "lr": 0.1,
        "weight_decay": 5e-4,
        # "label_smoothing": 0.05,
        # "grad_clip": 5.0,
    }

    wandb.init(project="deeparguing-resnet", config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])

    X, y, _, _ = load_torch_images(config["dataset"], device, shuffle=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, seed=config["seed"])
    batch_size = config["batch_size"]

    model = Resnet32(num_classes=10)
    model = model.to(device)
    train(model, X_train, y_train, X_val, y_val, config)

    
    model.eval()
    
    val_preds = []
    val_labels = []

    batch_size: int = config["batch_size"]
    
    with torch.no_grad():
        for i in range(0, X_val.shape[0], batch_size):
            xb = X_val[i : i + batch_size]
            yb = y_val[i : i + batch_size]
    
            out = model(xb, use_classification_head = True)
    
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
    
    
    save_path = "resnet_30.pt"
    torch.save(model.state_dict(), save_path)
    
    wandb.finish()



