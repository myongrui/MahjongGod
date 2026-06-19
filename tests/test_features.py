"""Tests for the ML feature extractor."""

import numpy as np
import pytest

from cracked.tiles import tile_id, Wind, tiles_from_names
from cracked.hand import HandState
from cracked.game_state import GameState, PlayerView
from cracked.training.features import (
    extract_features, N_FEATURES,
    _STATE_BLOCK_SIZE, _OPP_BLOCK_SIZE,
)


def _make_state(my_seat=Wind.EAST, prevailing=Wind.EAST) -> GameState:
    all_winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
    opponents = [PlayerView(seat=w) for w in all_winds if w != my_seat]
    return GameState(
        my_hand=HandState(seat_wind=my_seat),
        my_seat=my_seat,
        prevailing_wind=prevailing,
        opponents=opponents,
    )


def _set_hand(state: GameState, *names: str) -> None:
    state.my_hand.concealed = tiles_from_names(list(names))


# Full test hand — 14 tiles
_FULL_HAND = ("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","gd")


# ---------------------------------------------------------------------------
# Shape, dtype, and value bounds
# ---------------------------------------------------------------------------

def test_feature_vector_length():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    feat = extract_features(state, tile_id("gd"))
    assert feat.shape == (N_FEATURES,)


def test_feature_vector_dtype():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    feat = extract_features(state, tile_id("gd"))
    assert feat.dtype == np.float32


def test_feature_vector_bounded():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    feat = extract_features(state, tile_id("gd"))
    assert feat.min() >= -1e-6
    assert feat.max() <= 1.0 + 1e-6


def test_feature_deterministic():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    feat1 = extract_features(state, tile_id("gd"))
    feat2 = extract_features(state, tile_id("gd"))
    assert np.array_equal(feat1, feat2)


# ---------------------------------------------------------------------------
# Candidate discard one-hot (indices _STATE_BLOCK_SIZE .. +33)
# ---------------------------------------------------------------------------

def test_candidate_discard_onehot_set():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    tid = tile_id("gd")
    feat = extract_features(state, tid)
    assert feat[_STATE_BLOCK_SIZE + tid] == pytest.approx(1.0)


def test_candidate_discard_onehot_exclusive():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    tid = tile_id("gd")
    feat = extract_features(state, tid)
    for t in range(34):
        if t != tid:
            assert feat[_STATE_BLOCK_SIZE + t] == pytest.approx(0.0)


def test_different_discards_give_different_features():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    feat_rd = extract_features(state, tile_id("rd"))
    feat_gd = extract_features(state, tile_id("gd"))
    assert not np.array_equal(feat_rd, feat_gd)


# ---------------------------------------------------------------------------
# Seat wind and prevailing wind (indices 72–79)
# ---------------------------------------------------------------------------

def test_my_seat_east_onehot():
    state = _make_state(my_seat=Wind.EAST)
    _set_hand(state, *_FULL_HAND)
    feat = extract_features(state, tile_id("gd"))
    assert feat[72] == pytest.approx(1.0)   # East
    assert feat[73] == pytest.approx(0.0)
    assert feat[74] == pytest.approx(0.0)
    assert feat[75] == pytest.approx(0.0)


def test_prevailing_south_onehot():
    state = _make_state(prevailing=Wind.SOUTH)
    _set_hand(state, *_FULL_HAND)
    feat = extract_features(state, tile_id("gd"))
    assert feat[76] == pytest.approx(0.0)   # East
    assert feat[77] == pytest.approx(1.0)   # South
    assert feat[78] == pytest.approx(0.0)
    assert feat[79] == pytest.approx(0.0)


def test_different_seat_gives_different_features():
    state_e = _make_state(my_seat=Wind.EAST)
    state_s = _make_state(my_seat=Wind.SOUTH)
    _set_hand(state_e, "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","sw","rd","gd")
    _set_hand(state_s, "b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","sw","rd","gd")
    f_e = extract_features(state_e, tile_id("gd"))
    f_s = extract_features(state_s, tile_id("gd"))
    assert not np.array_equal(f_e, f_s)


# ---------------------------------------------------------------------------
# Concealed tile counts (indices 0–33)
# ---------------------------------------------------------------------------

def test_concealed_single_tile_normalized():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    c1_id = tile_id("c1")
    feat = extract_features(state, tile_id("gd"))
    assert feat[c1_id] == pytest.approx(1.0 / 4.0)


def test_concealed_quad_normalized():
    # Four b1s in hand → feat[b1_id] = 4/4 = 1.0
    state = _make_state()
    _set_hand(state, "b1","b1","b1","b1","c1","c2","c3","d1","d2","d3","ew","ew","rd","gd")
    b1_id = tile_id("b1")
    feat = extract_features(state, tile_id("gd"))
    assert feat[b1_id] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tiles-away after discard (index _STATE_BLOCK_SIZE + 34)
# ---------------------------------------------------------------------------

_TILES_AWAY_IDX = _STATE_BLOCK_SIZE + 34

def test_tiles_away_after_waiting_discard():
    # Discarding gd from this hand leaves tiles_away=0 → (0+1)/9
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    feat = extract_features(state, tile_id("gd"))
    assert feat[_TILES_AWAY_IDX] == pytest.approx(1.0 / 9.0, abs=1e-5)


def test_tiles_away_after_is_nonnegative():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    feat = extract_features(state, tile_id("rd"))
    assert feat[_TILES_AWAY_IDX] >= 0.0


# ---------------------------------------------------------------------------
# Opponent encoding (starts at _STATE_BLOCK_SIZE + 35)
# ---------------------------------------------------------------------------

_OPP_BASE = _STATE_BLOCK_SIZE + 35

def test_opponent_discard_reflected_in_features():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.discards = [tile_id("b5"), tile_id("b5")]  # 2 copies
    feat = extract_features(state, tile_id("gd"))
    # South is the first opponent (index 0); discards start at +5
    b5_id = tile_id("b5")
    assert feat[_OPP_BASE + 5 + b5_id] == pytest.approx(2.0 / 4.0)


def test_opponent_waiting_prob_encoded():
    state = _make_state()
    _set_hand(state, *_FULL_HAND)
    state.turn_number = 0
    feat_low = extract_features(state, tile_id("gd"))

    state2 = _make_state()
    _set_hand(state2, *_FULL_HAND)
    state2.turn_number = 30
    feat_high = extract_features(state2, tile_id("gd"))

    # waiting_prob at base+39; higher turn → higher prob
    assert feat_high[_OPP_BASE + 39] >= feat_low[_OPP_BASE + 39]


def test_n_features_constant_matches_computed():
    """Verify N_FEATURES equals 89 + 35 + 3×47 = 265."""
    assert N_FEATURES == _STATE_BLOCK_SIZE + 35 + 3 * _OPP_BLOCK_SIZE
