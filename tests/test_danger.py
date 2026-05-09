"""
Tests for opponent_model.py and danger.py.

Scenarios are constructed so the expected danger outcome is unambiguous:
  - Exhausted tiles are always safe
  - A tile in an opponent's flush suit is dangerous
  - A tile the opponent discarded is safe
  - Tenpai probability scales correctly with meld count
  - Expected shooting cost is higher when facing a flush opponent
"""

import pytest
import numpy as np

from cracked.tiles import tile_id, Wind, DRAGON_START
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView
from cracked.opponent_model import (
    OpponentModel, tenpai_probability, detect_suit_bias,
    model_opponent, model_all_opponents,
)
from cracked.danger import TileDanger, tile_danger_scores, expected_shooting_cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(my_seat=Wind.EAST, prevailing=Wind.EAST) -> GameState:
    all_winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
    opponents = [PlayerView(seat=w) for w in all_winds if w != my_seat]
    return GameState(
        my_hand=HandState(seat_wind=my_seat),
        my_seat=my_seat,
        prevailing_wind=prevailing,
        opponents=opponents,
    )


def _circle_meld(rank_start: int) -> Meld:
    """Exposed chow of three consecutive circles starting at rank_start."""
    base = 18  # CIRCLE_START
    tiles = (base + rank_start - 1, base + rank_start, base + rank_start + 1)
    return Meld(MeldType.CHOW, tiles)


def _pong_meld(name: str) -> Meld:
    tid = tile_id(name)
    return Meld(MeldType.PONG, (tid, tid, tid))


# ---------------------------------------------------------------------------
# tenpai_probability
# ---------------------------------------------------------------------------

def test_tenpai_prob_no_melds_early_is_low():
    assert tenpai_probability(0, 2) < 0.20


def test_tenpai_prob_three_melds_is_high():
    assert tenpai_probability(3, 5) > 0.55


def test_tenpai_prob_increases_with_melds():
    assert tenpai_probability(0, 0) < tenpai_probability(1, 0) < tenpai_probability(2, 0)


def test_tenpai_prob_increases_with_turns():
    assert tenpai_probability(0, 0) < tenpai_probability(0, 15)


def test_tenpai_prob_capped_below_one():
    assert tenpai_probability(4, 50) <= 0.95


# ---------------------------------------------------------------------------
# detect_suit_bias — meld evidence
# ---------------------------------------------------------------------------

def test_suit_bias_from_circle_melds():
    opp = PlayerView(seat=Wind.SOUTH)
    opp.melds = [_circle_meld(1), _circle_meld(4)]
    bias, honor = detect_suit_bias(opp)
    assert bias == 2  # circles


def test_suit_bias_from_bamboo_pong():
    opp = PlayerView(seat=Wind.SOUTH)
    opp.melds = [_pong_meld("b3")]
    bias, _ = detect_suit_bias(opp)
    assert bias == 0  # bamboo


def test_no_bias_with_mixed_suit_melds():
    opp = PlayerView(seat=Wind.SOUTH)
    opp.melds = [_circle_meld(1), _pong_meld("b3")]
    bias, _ = detect_suit_bias(opp)
    assert bias is None


# ---------------------------------------------------------------------------
# detect_suit_bias — discard evidence
# ---------------------------------------------------------------------------

def test_suit_bias_from_discards_absent_suit():
    # Opponent discards bamboo and characters, never circles → circles bias
    opp = PlayerView(seat=Wind.SOUTH)
    opp.discards = [
        tile_id("b1"), tile_id("b3"), tile_id("b5"),
        tile_id("c2"), tile_id("c4"),
    ]
    bias, _ = detect_suit_bias(opp)
    assert bias == 2  # circles absent from discards


def test_no_bias_when_too_few_discards():
    opp = PlayerView(seat=Wind.SOUTH)
    opp.discards = [tile_id("b1"), tile_id("c1")]
    bias, _ = detect_suit_bias(opp)
    assert bias is None


def test_honor_bias_detected():
    # Opponent only discards suited tiles, never honors (after 6+ discards)
    opp = PlayerView(seat=Wind.SOUTH)
    opp.discards = [
        tile_id("b1"), tile_id("b2"), tile_id("b3"),
        tile_id("b4"), tile_id("b5"), tile_id("b6"),
    ]
    _, honor = detect_suit_bias(opp)
    assert honor is True


# ---------------------------------------------------------------------------
# model_opponent — dangerous and safe tile sets
# ---------------------------------------------------------------------------

def test_discarded_tile_is_safe():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.discards = [tile_id("d5")]
    model = model_opponent(opp, state)
    assert tile_id("d5") in model.safe_tiles


def test_exhausted_tile_is_safe():
    state = _make_state()
    # Put all 4 copies of b1 in our hand to exhaust it
    state.my_hand.concealed[tile_id("b1")] = 4
    opp = state.opponent_by_seat(Wind.SOUTH)
    model = model_opponent(opp, state)
    assert tile_id("b1") in model.safe_tiles


def test_flush_suit_tiles_are_dangerous():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.melds = [_circle_meld(1), _circle_meld(4)]
    model = model_opponent(opp, state)
    # All circle tiles not yet discarded or exhausted should be in dangerous set
    for rank in range(1, 10):
        tid = tile_id(f"d{rank}")
        if tid not in model.safe_tiles:
            assert tid in model.dangerous_tiles


def test_adjacent_to_meld_is_dangerous():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    # Opponent has exposed [b4, b5, b6] chow
    opp.melds = [Meld(MeldType.CHOW, (tile_id("b4"), tile_id("b5"), tile_id("b6")))]
    model = model_opponent(opp, state)
    # b3 and b7 are within 2 ranks of meld tiles
    assert tile_id("b3") in model.dangerous_tiles or tile_id("b7") in model.dangerous_tiles


def test_honor_tiles_not_dangerous_vs_suited_flush_opponent():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.melds = [_circle_meld(1), _circle_meld(4)]
    model = model_opponent(opp, state)
    # East wind is not in circles — should not appear in dangerous_tiles due to flush
    assert tile_id("ew") not in model.dangerous_tiles


def test_est_tai_higher_for_flush_with_dragon_pong():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.melds = [_circle_meld(1), _pong_meld("rd")]
    model_flush = model_opponent(opp, state)

    opp2 = state.opponent_by_seat(Wind.WEST)
    model_plain = model_opponent(opp2, state)

    assert model_flush.est_tai > model_plain.est_tai


# ---------------------------------------------------------------------------
# model_all_opponents
# ---------------------------------------------------------------------------

def test_model_all_opponents_count():
    state = _make_state(my_seat=Wind.EAST)
    models = model_all_opponents(state)
    assert len(models) == 3


def test_model_all_opponents_seats_correct():
    state = _make_state(my_seat=Wind.EAST)
    models = model_all_opponents(state)
    seats = {m.seat for m in models}
    assert seats == {Wind.SOUTH, Wind.WEST, Wind.NORTH}


# ---------------------------------------------------------------------------
# tile_danger_scores
# ---------------------------------------------------------------------------

def test_exhausted_tile_has_zero_danger():
    state = _make_state()
    state.my_hand.concealed[tile_id("b1")] = 4  # exhaust b1
    models = model_all_opponents(state)
    scores = tile_danger_scores(state, models)
    assert scores[tile_id("b1")].score == 0.0
    assert scores[tile_id("b1")].exhausted is True
    assert scores[tile_id("b1")].expected_cost == 0.0


def test_no_opponents_all_scores_zero():
    state = _make_state()
    scores = tile_danger_scores(state, models=[])
    for tid in range(34):
        assert scores[tid].score == 0.0
        assert scores[tid].expected_cost == 0.0


def test_flush_opponent_raises_circle_danger():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.melds = [_circle_meld(1), _circle_meld(4)]
    state.turn_number = 8
    models = model_all_opponents(state)
    scores = tile_danger_scores(state, models)

    # Circles should be more dangerous than bamboo (opponent going for circles)
    circle_score = scores[tile_id("d5")].score
    bamboo_score = scores[tile_id("b1")].score
    assert circle_score > bamboo_score


def test_discarded_tile_lower_danger_than_unknown():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.discards = [tile_id("c5")]
    state.turn_number = 8
    models = model_all_opponents(state)
    scores = tile_danger_scores(state, models)

    # c5 was discarded by the opponent → safer than an unknown tile
    assert scores[tile_id("c5")].score < scores[tile_id("c9")].score


# ---------------------------------------------------------------------------
# expected_shooting_cost
# ---------------------------------------------------------------------------

def test_expected_cost_zero_for_exhausted():
    state = _make_state()
    state.my_hand.concealed[tile_id("d3")] = 4
    models = model_all_opponents(state)
    cost = expected_shooting_cost(tile_id("d3"), state, models)
    assert cost == 0.0


def test_expected_cost_higher_for_flush_opponent():
    state_flush = _make_state()
    opp_f = state_flush.opponent_by_seat(Wind.SOUTH)
    opp_f.melds = [_circle_meld(1), _circle_meld(4)]
    state_flush.turn_number = 10
    models_flush = model_all_opponents(state_flush)
    cost_flush = expected_shooting_cost(tile_id("d5"), state_flush, models_flush)

    state_plain = _make_state()
    state_plain.turn_number = 10
    models_plain = model_all_opponents(state_plain)
    cost_plain = expected_shooting_cost(tile_id("d5"), state_plain, models_plain)

    assert cost_flush > cost_plain


def test_expected_cost_increases_with_tenpai_prob():
    # Late game (more turns) → higher tenpai prob → higher cost
    state_early = _make_state()
    state_early.turn_number = 1
    state_late = _make_state()
    state_late.turn_number = 20

    models_early = model_all_opponents(state_early)
    models_late = model_all_opponents(state_late)

    cost_early = expected_shooting_cost(tile_id("b5"), state_early, models_early)
    cost_late = expected_shooting_cost(tile_id("b5"), state_late, models_late)
    assert cost_late > cost_early
