"""
Training loop for DangerNet.

Usage:
    python -m cracked.training.trainer --log data/game_log.jsonl

Requires PyTorch: pip install 'cracked[ml]'
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cracked.training.data import load_dataset
from cracked.training.model import DangerNet, save_model

DEFAULT_MODEL_PATH = Path("models/danger_net.pt")


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    val_split: float = 0.1
    seed: int = 42


def train(
    log_file: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    config: Optional[TrainConfig] = None,
    verbose: bool = True,
) -> float:
    """
    Train DangerNet on recorded simulation examples and save the best checkpoint.

    Returns the best validation loss achieved.
    Raises ImportError if torch is not installed.
    Raises ValueError if the log file contains no examples.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Run: pip install 'cracked[ml]'"
        ) from exc

    if config is None:
        config = TrainConfig()

    torch.manual_seed(config.seed)

    examples = load_dataset(log_file)
    if not examples:
        raise ValueError(f"No training examples found in {log_file}")

    if verbose:
        print(f"Loaded {len(examples)} examples from {log_file}")

    X = torch.tensor(
        np.stack([e.features for e in examples]), dtype=torch.float32
    )
    y = torch.tensor(
        [[e.win_rate, e.shoot_rate, e.expected_gain] for e in examples],
        dtype=torch.float32,
    )

    dataset = TensorDataset(X, y)
    n_val = max(1, int(len(dataset) * config.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = DangerNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, model_path)

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{config.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

    if verbose:
        print(f"Best val MSE: {best_val_loss:.4f}  →  {model_path}")

    return best_val_loss


def _cli():
    parser = argparse.ArgumentParser(description="Train DangerNet on recorded game data.")
    parser.add_argument("--log", type=Path, default=Path("data/game_log.jsonl"),
                        help="JSONL log file produced by 'cracked recommend --deep'")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )
    train(args.log, model_path=args.out, config=cfg)


if __name__ == "__main__":
    _cli()
