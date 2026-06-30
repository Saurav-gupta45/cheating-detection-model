"""
ModelTrainer: PyTorch MLP gaze model training pipeline.

Trains a GazeClassifier MLP on safe_weighted.csv and cheat_weighted.csv datasets.
Uses BCELoss + Adam optimizer with early stopping on validation loss.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from src.utils import get_artifact_path
from src.exception import CustomException
from src.logger import logging


class GazeDataset(Dataset):
    """PyTorch Dataset wrapping gaze coordinate arrays."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class ModelTrainerConfig:
    model_save_path: str = None
    safe_data_path: str = None
    cheat_data_path: str = None
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    def __post_init__(self):
        if self.model_save_path is None:
            self.model_save_path = get_artifact_path("proctor_model.pth")
        if self.safe_data_path is None:
            self.safe_data_path = get_artifact_path("safe_weighted.csv")
        if self.cheat_data_path is None:
            self.cheat_data_path = get_artifact_path("cheat_weighted.csv")


class ModelTrainer:
    """
    Trains the GazeClassifier MLP and saves the best checkpoint.
    
    Usage:
        trainer = ModelTrainer()
        r2 = trainer.train()
    """

    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()

    def _load_dataset(self):
        """Load and combine safe + cheat gaze coordinate CSVs."""
        logging.info("Loading gaze datasets for training")

        if not os.path.exists(self.config.safe_data_path):
            raise FileNotFoundError(f"Safe dataset not found: {self.config.safe_data_path}")
        if not os.path.exists(self.config.cheat_data_path):
            raise FileNotFoundError(f"Cheat dataset not found: {self.config.cheat_data_path}")

        safe_data = np.loadtxt(self.config.safe_data_path, delimiter=",")
        cheat_data = np.loadtxt(self.config.cheat_data_path, delimiter=",")

        combined = np.vstack((safe_data, cheat_data))
        X = combined[:, :-1]
        y = combined[:, -1]

        logging.info(f"Loaded {len(safe_data)} safe + {len(cheat_data)} cheat = {len(combined)} total samples")
        return X, y

    def train(self):
        """Run the full training pipeline. Returns best validation accuracy."""
        try:
            from src.components.gaze_detector import GazeClassifier

            device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            logging.info(f"Training on device: {device}")

            X, y = self._load_dataset()

            # Shuffle
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            # 80/20 split
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            train_loader = DataLoader(GazeDataset(X_train, y_train), batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(GazeDataset(X_val, y_val), batch_size=self.config.batch_size, shuffle=False)

            model = GazeClassifier(input_dim=X.shape[1]).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

            best_val_loss = float('inf')
            best_val_acc = 0.0

            for epoch in range(1, self.config.epochs + 1):
                # Training
                model.train()
                running_loss, correct, total = 0.0, 0, 0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * bx.size(0)
                    correct += ((out >= 0.5).float() == by).sum().item()
                    total += by.size(0)

                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(device), by.to(device)
                        out = model(bx)
                        loss = criterion(out, by)
                        val_loss += loss.item() * bx.size(0)
                        val_correct += ((out >= 0.5).float() == by).sum().item()
                        val_total += by.size(0)

                epoch_val_loss = val_loss / val_total
                epoch_val_acc = val_correct / val_total

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_val_acc = epoch_val_acc
                    torch.save(model.state_dict(), self.config.model_save_path)
                    logging.info(f"Epoch {epoch}: Saved best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

            logging.info(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
            return best_val_acc

        except Exception as e:
            raise CustomException(e, sys)
