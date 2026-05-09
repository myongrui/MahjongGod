"""Game log recording and dataset management for ML training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cracked.game_state import GameState
from cracked.simulator import SimulationResult
from cracked.training.features import extract_features

DEFAULT_LOG_FILE = Path("data/game_log.jsonl")


@dataclass
class TrainingExample:
    """One labeled example: (state + candidate_discard) → simulation outcomes."""
    features: np.ndarray   # shape (N_FEATURES,), float32
    win_rate: float
    shoot_rate: float
    expected_gain: float


def record_simulation(
    state: GameState,
    sim_result: SimulationResult,
    log_file: Path = DEFAULT_LOG_FILE,
) -> None:
    """
    Append one simulation result to the JSONL log file.

    Each line is a JSON object with keys: features, win_rate, shoot_rate,
    expected_gain.  The log file is created (with parent dirs) if absent.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    feat = extract_features(state, sim_result.tile_id)
    entry = {
        "features": feat.tolist(),
        "win_rate": sim_result.win_rate,
        "shoot_rate": sim_result.shoot_rate,
        "expected_gain": sim_result.expected_gain,
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_dataset(log_file: Path = DEFAULT_LOG_FILE) -> list[TrainingExample]:
    """Load all training examples from a JSONL log file."""
    log_file = Path(log_file)
    if not log_file.exists():
        return []
    examples: list[TrainingExample] = []
    with log_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            examples.append(TrainingExample(
                features=np.array(d["features"], dtype=np.float32),
                win_rate=d["win_rate"],
                shoot_rate=d["shoot_rate"],
                expected_gain=d["expected_gain"],
            ))
    return examples


def dataset_stats(log_file: Path = DEFAULT_LOG_FILE) -> dict:
    """Return summary statistics about the training dataset."""
    examples = load_dataset(log_file)
    if not examples:
        return {"n_examples": 0}
    gains = [e.expected_gain for e in examples]
    return {
        "n_examples": len(examples),
        "mean_gain": float(np.mean(gains)),
        "std_gain": float(np.std(gains)),
        "min_gain": float(np.min(gains)),
        "max_gain": float(np.max(gains)),
    }
