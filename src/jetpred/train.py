import argparse
import random

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jetpred import config
from jetpred.data import prepare_data, save_scaler
from jetpred.model import JetClassifier


def set_seed(seed):
    """Fixe toutes les sources d'aléatoire pour des résultats reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X, y, batch_size, shuffle):
    """Emballe les tableaux numpy dans un DataLoader qui sert des batches."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()                                   # active le dropout
    total_loss, n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()                       # 1. remise à zéro des gradients
        logits = model(X_batch)                     # 2. passe avant
        loss = loss_fn(logits, y_batch)
        loss.backward()                             # 3. rétropropagation
        optimizer.step()                            # 4. mise à jour des poids
        total_loss += loss.item() * len(X_batch)
        n += len(X_batch)
    return total_loss / n


@torch.no_grad()                                    # pas de gradients en évaluation
def evaluate(model, loader, loss_fn, device):
    model.eval()                                    # désactive le dropout
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += loss_fn(logits, y_batch).item() * len(X_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        n += len(X_batch)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser(description="Entraîne le classificateur de jets")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_data()
    train_loader = make_loader(data["X_train"], data["y_train"], args.batch_size, shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"], args.batch_size, shuffle=False)

    model = JetClassifier(dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(vars(args))               # hyperparamètres de ce run
        mlflow.log_param("device", device)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        model_path = config.ARTIFACTS_DIR / "model.pt"

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

            # Métriques enregistrées à chaque époque : MLflow tracera les courbes
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
                step=epoch,
            )
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

            # Arrêt anticipé : on sauvegarde le meilleur modèle selon la validation
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Arrêt anticipé à l'époque {epoch}")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        scaler_path = save_scaler(data["scaler"])
        mlflow.log_artifact(str(model_path))        # poids du meilleur modèle
        mlflow.log_artifact(str(scaler_path))       # scaler indispensable à l'inférence


if __name__ == "__main__":
    main()