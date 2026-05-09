"""Tests for the ML training data recording pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from cracked.tiles import tile_id, Wind, tiles_from_names
from cracked.hand import HandState
from cracked.game_state import GameState, PlayerView
from cracked.simulator import SimulationResult
from cracked.training.data import (
    TrainingExample, record_simulation, load_dataset, dataset_stats,
    DEFAULT_LOG_FILE,
)
from cracked.training.features import N_FEATURES


def _make_state() -> GameState:
    all_winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
    return GameState(
        my_hand=HandState(seat_wind=Wind.EAST),
        my_seat=Wind.EAST,
        prevailing_wind=Wind.EAST,
        opponents=[PlayerView(seat=w) for w in all_winds if w != Wind.EAST],
    )


def _full_state() -> GameState:
    state = _make_state()
    state.my_hand.concealed = tiles_from_names(
        ["b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","gd"]
    )
    return state


def _fake_sim_result(tile: str, win=0.2, shoot=0.1, gain=0.5) -> SimulationResult:
    return SimulationResult(
        tile_id=tile_id(tile),
        n_games=10,
        win_count=int(win * 10),
        shoot_count=int(shoot * 10),
        draw_count=10 - int(win * 10) - int(shoot * 10),
        total_net=gain * 10,
    )


# ---------------------------------------------------------------------------
# record_simulation
# ---------------------------------------------------------------------------

def test_record_creates_file(tmp_path):
    log = tmp_path / "test.jsonl"
    state = _full_state()
    sr = _fake_sim_result("gd")
    record_simulation(state, sr, log_file=log)
    assert log.exists()


def test_record_writes_valid_json(tmp_path):
    log = tmp_path / "test.jsonl"
    state = _full_state()
    sr = _fake_sim_result("gd")
    record_simulation(state, sr, log_file=log)
    line = log.read_text().strip()
    d = json.loads(line)
    assert "features" in d
    assert "win_rate" in d
    assert "shoot_rate" in d
    assert "expected_gain" in d


def test_record_appends_multiple(tmp_path):
    log = tmp_path / "test.jsonl"
    state = _full_state()
    record_simulation(state, _fake_sim_result("gd"), log_file=log)
    record_simulation(state, _fake_sim_result("rd"), log_file=log)
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_record_feature_length(tmp_path):
    log = tmp_path / "test.jsonl"
    state = _full_state()
    record_simulation(state, _fake_sim_result("gd"), log_file=log)
    d = json.loads(log.read_text().strip())
    assert len(d["features"]) == N_FEATURES


def test_record_creates_parent_dirs(tmp_path):
    log = tmp_path / "deep" / "nested" / "log.jsonl"
    state = _full_state()
    record_simulation(state, _fake_sim_result("gd"), log_file=log)
    assert log.exists()


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_empty_returns_empty_list(tmp_path):
    log = tmp_path / "missing.jsonl"
    examples = load_dataset(log)
    assert examples == []


def test_load_roundtrip(tmp_path):
    log = tmp_path / "test.jsonl"
    state = _full_state()
    sr = _fake_sim_result("gd", win=0.3, shoot=0.05, gain=1.2)
    record_simulation(state, sr, log_file=log)
    examples = load_dataset(log)
    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, TrainingExample)
    assert ex.features.shape == (N_FEATURES,)
    assert ex.features.dtype == np.float32
    assert ex.win_rate == pytest.approx(sr.win_rate)
    assert ex.shoot_rate == pytest.approx(sr.shoot_rate)
    assert ex.expected_gain == pytest.approx(sr.expected_gain)


def test_load_multiple_examples(tmp_path):
    log = tmp_path / "test.jsonl"
    state = _full_state()
    for tile in ("gd", "rd", "ew"):
        record_simulation(state, _fake_sim_result(tile), log_file=log)
    examples = load_dataset(log)
    assert len(examples) == 3


# ---------------------------------------------------------------------------
# dataset_stats
# ---------------------------------------------------------------------------

def test_stats_empty(tmp_path):
    log = tmp_path / "missing.jsonl"
    stats = dataset_stats(log)
    assert stats == {"n_examples": 0}


def test_stats_populated(tmp_path):
    log = tmp_path / "test.jsonl"
    state = _full_state()
    record_simulation(state, _fake_sim_result("gd", gain=1.0), log_file=log)
    record_simulation(state, _fake_sim_result("rd", gain=-1.0), log_file=log)
    stats = dataset_stats(log)
    assert stats["n_examples"] == 2
    assert stats["mean_gain"] == pytest.approx(0.0, abs=1e-5)
    assert "std_gain" in stats
    assert stats["min_gain"] == pytest.approx(-1.0, abs=1e-3)
    assert stats["max_gain"] == pytest.approx(1.0, abs=1e-3)
