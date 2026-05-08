"""Tests for GameState, PlayerView serialization and tile tracking."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cracked.tiles import Wind, tile_id, full_wall, new_hand_array
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView, save_state, load_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(my_seat=Wind.EAST, prev_wind=Wind.EAST) -> GameState:
    all_winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
    opponents = [PlayerView(seat=w) for w in all_winds if w != my_seat]
    return GameState(
        my_hand=HandState(seat_wind=my_seat),
        my_seat=my_seat,
        prevailing_wind=prev_wind,
        opponents=opponents,
    )


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_new_state_has_three_opponents():
    state = _make_state()
    assert len(state.opponents) == 3


def test_new_state_default_wall():
    state = _make_state()
    assert state.wall_tiles_remaining == 136


def test_opponent_by_seat_found():
    state = _make_state(my_seat=Wind.EAST)
    opp = state.opponent_by_seat(Wind.SOUTH)
    assert opp.seat == Wind.SOUTH


def test_opponent_by_seat_not_found_raises():
    state = _make_state(my_seat=Wind.EAST)
    with pytest.raises(ValueError):
        state.opponent_by_seat(Wind.EAST)  # our own seat, not an opponent


# ---------------------------------------------------------------------------
# visible_tiles / unknown_tiles
# ---------------------------------------------------------------------------

def test_visible_tiles_empty_hand():
    state = _make_state()
    vis = state.visible_tiles()
    assert vis.sum() == 0


def test_visible_tiles_includes_our_concealed():
    state = _make_state()
    state.my_hand.concealed[tile_id("b1")] = 2
    vis = state.visible_tiles()
    assert vis[tile_id("b1")] == 2


def test_visible_tiles_includes_opponent_discards():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.discards.append(tile_id("c5"))
    vis = state.visible_tiles()
    assert vis[tile_id("c5")] == 1


def test_visible_tiles_includes_exposed_meld():
    state = _make_state()
    meld = Meld(MeldType.PONG, (tile_id("d3"), tile_id("d3"), tile_id("d3")))
    state.my_hand.melds.append(meld)
    vis = state.visible_tiles()
    assert vis[tile_id("d3")] == 3


def test_unknown_tiles_full_wall_minus_visible():
    state = _make_state()
    state.my_hand.concealed[tile_id("b1")] = 3
    unknown = state.unknown_tiles()
    # full wall has 4 of each; we see 3, so 1 unknown
    assert unknown[tile_id("b1")] == 1
    # unrelated tile still has 4 unknown
    assert unknown[tile_id("c9")] == 4


def test_unknown_tiles_cannot_go_negative():
    state = _make_state()
    # Put all 4 copies in our hand
    state.my_hand.concealed[tile_id("ew")] = 4
    unknown = state.unknown_tiles()
    assert unknown[tile_id("ew")] == 0


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

def test_round_trip_empty_state():
    state = _make_state()
    d = state.to_dict()
    restored = GameState.from_dict(d)
    assert restored.my_seat == state.my_seat
    assert restored.prevailing_wind == state.prevailing_wind
    assert len(restored.opponents) == 3
    assert restored.wall_tiles_remaining == 136


def test_round_trip_with_hand_tiles():
    state = _make_state()
    state.my_hand.concealed[tile_id("b3")] = 2
    state.my_hand.concealed[tile_id("rd")] = 1
    d = state.to_dict()
    restored = GameState.from_dict(d)
    assert restored.my_hand.concealed[tile_id("b3")] == 2
    assert restored.my_hand.concealed[tile_id("rd")] == 1


def test_round_trip_with_opponent_discards():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.WEST)
    opp.discards = [tile_id("c1"), tile_id("c2")]
    d = state.to_dict()
    restored = GameState.from_dict(d)
    restored_opp = restored.opponent_by_seat(Wind.WEST)
    assert restored_opp.discards == [tile_id("c1"), tile_id("c2")]


def test_round_trip_with_meld():
    state = _make_state()
    tid = tile_id("rd")
    meld = Meld(MeldType.PONG, (tid, tid, tid))
    state.my_hand.add_meld(meld)
    state.my_hand.concealed[tid] -= 3  # remove from concealed to keep consistent
    d = state.to_dict()
    restored = GameState.from_dict(d)
    assert len(restored.my_hand.melds) == 1
    assert restored.my_hand.melds[0].type == MeldType.PONG


def test_round_trip_with_flowers_and_animals():
    from cracked.tiles import FLOWER_SPRING, ANIMAL_CAT
    state = _make_state()
    state.my_hand.flowers = [FLOWER_SPRING]
    state.my_hand.animals = [ANIMAL_CAT]
    d = state.to_dict()
    restored = GameState.from_dict(d)
    assert restored.my_hand.flowers == [FLOWER_SPRING]
    assert restored.my_hand.animals == [ANIMAL_CAT]


# ---------------------------------------------------------------------------
# File persistence (save_state / load_state)
# ---------------------------------------------------------------------------

def test_save_and_load_state(tmp_path):
    p = tmp_path / "game.json"
    state = _make_state(my_seat=Wind.SOUTH)
    save_state(state, path=p)
    assert p.exists()

    loaded = load_state(path=p)
    assert loaded.my_seat == Wind.SOUTH


def test_load_state_missing_file(tmp_path):
    p = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        load_state(path=p)


def test_save_load_preserves_turn_number(tmp_path):
    p = tmp_path / "game.json"
    state = _make_state()
    state.turn_number = 7
    save_state(state, path=p)
    loaded = load_state(path=p)
    assert loaded.turn_number == 7


def test_env_var_state_file(tmp_path, monkeypatch):
    p = tmp_path / "env_game.json"
    monkeypatch.setenv("CRACKED_STATE_FILE", str(p))
    state = _make_state()
    save_state(state)
    loaded = load_state()
    assert loaded.my_seat == state.my_seat
