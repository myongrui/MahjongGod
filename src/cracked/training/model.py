"""
DangerNet: residual MLP that maps a game-state feature vector to
(win_rate, shoot_rate, expected_gain).

Requires PyTorch: pip install 'cracked[ml]'
"""

from __future__ import annotations

from pathlib import Path

from cracked.training.features import N_FEATURES

N_OUTPUTS = 3      # win_rate, shoot_rate, expected_gain
HIDDEN_SIZE = 256
N_LAYERS = 3


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for model use. Run: pip install 'cracked[ml]'"
        ) from exc


class ResidualBlock:
    """Defined at import time only when torch is available."""


def _build_classes():
    torch = _require_torch()
    import torch.nn as nn

    class _ResidualBlock(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(size, size),
                nn.LayerNorm(size),
                nn.ReLU(),
                nn.Linear(size, size),
                nn.LayerNorm(size),
            )
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class _DangerNet(nn.Module):
        """
        Residual MLP: input projection → N_LAYERS residual blocks → output head.

        Input:  feature vector of length N_FEATURES
        Output: [win_rate_pred, shoot_rate_pred, expected_gain_pred]
        """
        def __init__(
            self,
            n_features: int = N_FEATURES,
            hidden: int = HIDDEN_SIZE,
            n_layers: int = N_LAYERS,
        ):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )
            self.blocks = nn.Sequential(
                *[_ResidualBlock(hidden) for _ in range(n_layers)]
            )
            self.head = nn.Linear(hidden, N_OUTPUTS)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.blocks(x)
            return self.head(x)

    return _DangerNet


def DangerNet(
    n_features: int = N_FEATURES,
    hidden: int = HIDDEN_SIZE,
    n_layers: int = N_LAYERS,
):
    """Factory that builds a DangerNet instance (requires torch)."""
    cls = _build_classes()
    return cls(n_features=n_features, hidden=hidden, n_layers=n_layers)


def save_model(model, path: Path) -> None:
    """Save model weights and construction hyperparameters to a .pt file."""
    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": model.input_proj[0].in_features,
            "hidden": model.blocks[0].net[0].in_features,
            "n_layers": len(model.blocks),
        },
        path,
    )


def load_model(path: Path):
    """Load a DangerNet from a saved .pt checkpoint."""
    torch = _require_torch()
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = DangerNet(
        n_features=ckpt["n_features"],
        hidden=ckpt["hidden"],
        n_layers=ckpt["n_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
