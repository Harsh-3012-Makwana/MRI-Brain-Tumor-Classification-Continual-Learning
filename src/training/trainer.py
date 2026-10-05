"""Training Engine for Brain Tumor MRI Classification with Validation Monitoring & Early Stopping."""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.evaluator import evaluate_model


class ModelTrainer:
    """Handles model training, validation monitoring, early stopping, and checkpoint saving."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
        max_epochs: int = 25,
        patience: int = 7,
        checkpoint_dir: Path = Path("artifacts/checkpoints/resnet18_baseline"),
        seed: int = 42
    ):
        """
        Args:
            model: PyTorch model instance.
            train_loader: DataLoader for training split ONLY.
            val_loader: DataLoader for validation split ONLY.
            optimizer: PyTorch optimizer instance.
            criterion: Loss function (default: CrossEntropyLoss).
            scheduler: Optional LR scheduler.
            device: Target torch.device (default: cuda if available else cpu).
            max_epochs: Maximum training epochs.
            patience: Early stopping patience.
            checkpoint_dir: Directory for saving model checkpoints.
            seed: Random seed for training reproducibility.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"

    def train_epoch(self) -> Tuple[float, float]:
        """Runs a single training epoch."""
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == targets).sum().item()
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = correct_preds / total_samples if total_samples > 0 else 0.0
        return epoch_loss, epoch_acc

    def fit(self) -> Dict[str, Any]:
        """Executes full training loop with validation monitoring and early stopping."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "lr": []
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0
        start_time = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc = self.train_epoch()

            # Run validation evaluation
            val_metrics = evaluate_model(self.model, self.val_loader, self.device)
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]

            # Standard loss calculation for validation
            self.model.eval()
            val_running_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for images, targets in self.val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                    val_running_loss += loss.item() * images.size(0)
                    val_samples += images.size(0)
            val_loss = val_running_loss / val_samples if val_samples > 0 else 0.0

            current_lr = self.optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_acc"].append(round(val_acc, 4))
            history["val_macro_f1"].append(round(val_f1, 4))
            history["lr"].append(current_lr)

            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping check based strictly on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_acc": val_acc,
                    "best_val_f1": val_f1
                }, self.best_checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"[Trainer] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                break

        total_train_time = time.perf_counter() - start_time

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "total_train_time_seconds": round(total_train_time, 2),
            "checkpoint_path": str(self.best_checkpoint_path)
        }

    def load_best_checkpoint(self):
        """Reloads weights from the best saved checkpoint."""
        if not self.best_checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at: {self.best_checkpoint_path}")

        checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
