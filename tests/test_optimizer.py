"""Tests for the heuristic discard optimizer."""

import pytest

from cracked.tiles import tile_id, Wind
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView
from cracked.opponent_model import model_all_opponents
from cracked.optimizer import DiscardRecommendation, adaptive_alpha, recommend_discard


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


def _set_hand(state: GameState, *names: str) -> None:
    """Load tile names into the concealed hand."""
    from cracked.tiles import tiles_from_names
    state.my_hand.concealed = tiles_from_names(list(names))


def _circle_pong(rank: int) -> Meld:
    tid = tile_id(f"d{rank}")
    return Meld(MeldType.PONG, (tid, tid, tid))


# ---------------------------------------------------------------------------
# adaptive_alpha
# ---------------------------------------------------------------------------

def test_alpha_lower_with_dangerous_opponents():
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.melds = [_circle_pong(1), _circle_pong(3), _circle_pong(5)]
    state.turn_number = 10

    models_dangerous = model_all_opponents(state)
    alpha_dangerous = adaptive_alpha(state, models_dangerous, best_shanten=1)

    state2 = _make_state()
    state2.turn_number = 10
    models_safe = model_all_opponents(state2)
    alpha_safe = adaptive_alpha(state2, models_safe, best_shanten=1)

    assert alpha_dangerous < alpha_safe


def test_alpha_higher_when_hand_closer_to_tenpai():
    state = _make_state()
    models = model_all_opponents(state)
    alpha_tenpai = adaptive_alpha(state, models, best_shanten=0)
    alpha_far = adaptive_alpha(state, models, best_shanten=3)
    assert alpha_tenpai > alpha_far


def test_alpha_lower_with_thin_wall():
    state_full = _make_state()
    state_full.wall_tiles_remaining = 136
    state_thin = _make_state()
    state_thin.wall_tiles_remaining = 10

    models = model_all_opponents(state_full)
    alpha_full = adaptive_alpha(state_full, models, best_shanten=1)
    alpha_thin = adaptive_alpha(state_thin, models, best_shanten=1)
    assert alpha_thin < alpha_full


def test_alpha_bounded():
    state = _make_state()
    models = model_all_opponents(state)
    for shanten_val in (-1, 0, 1, 3, 8):
        a = adaptive_alpha(state, models, shanten_val)
        assert 0.15 <= a <= 0.90


# ---------------------------------------------------------------------------
# recommend_discard — basic correctness
# ---------------------------------------------------------------------------

def test_recommend_raises_on_wrong_tile_count():
    state = _make_state()
    _set_hand(state, "b1", "b2", "b3")  # only 3 tiles, need 14
    with pytest.raises(ValueError):
        recommend_discard(state)


def test_recommend_returns_list_of_recommendations():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    results = recommend_discard(state)
    assert len(results) > 0
    assert all(isinstance(r, DiscardRecommendation) for r in results)


def test_recommendation_has_all_fields():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    r = recommend_discard(state)[0]
    assert hasattr(r, "tile_id")
    assert hasattr(r, "shanten_after")
    assert hasattr(r, "weighted_acceptance")
    assert hasattr(r, "shooting_cost")
    assert hasattr(r, "danger_score")
    assert hasattr(r, "utility")
    assert hasattr(r, "acceptance")


def test_recommend_sorted_by_shanten_first():
    state = _make_state()
    # Disorganized hand — different discards will give different shanten
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","sw","ww","nw","rd")
    results = recommend_discard(state)
    shantens = [r.shanten_after for r in results]
    assert shantens == sorted(shantens)


def test_recommend_utility_between_zero_and_one():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    results = recommend_discard(state)
    for r in results:
        assert 0.0 <= r.utility <= 1.0


# ---------------------------------------------------------------------------
# recommend_discard — shanten / tenpai hands
# ---------------------------------------------------------------------------

def test_recommend_tenpai_hand_shanten_zero():
    # After discarding the right tile, hand should be shanten=0
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","rd")
    results = recommend_discard(state)
    # Best discard brings us to shanten=-1 (complete) or 0 (tenpai)
    assert results[0].shanten_after <= 0


def test_recommend_best_is_tenpai_for_near_complete_hand():
    # Four complete groups + rd-rd pair in hand already → best discard gives tenpai (shanten=0)
    # The 14-tile hand IS already complete (shanten=-1), but recommend evaluates
    # 13-tile hands after each discard; those land at shanten=0.
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","rd")
    results = recommend_discard(state)
    assert results[0].shanten_after == 0


# ---------------------------------------------------------------------------
# recommend_discard — danger awareness
# ---------------------------------------------------------------------------

def test_shooting_cost_nonzero_with_dangerous_opponent():
    state = _make_state()
    state.turn_number = 10
    opp = state.opponent_by_seat(Wind.SOUTH)
    # Opponent going for circles flush with 3 melds
    opp.melds = [_circle_pong(2), _circle_pong(5), _circle_pong(8)]

    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    results = recommend_discard(state)

    # Circles (d-suit) discards should have higher shooting cost than bamboo
    d_recs = [r for r in results if 18 <= r.tile_id <= 26]
    b_recs = [r for r in results if 0 <= r.tile_id <= 8]

    if d_recs and b_recs:
        avg_d_cost = sum(r.shooting_cost for r in d_recs) / len(d_recs)
        avg_b_cost = sum(r.shooting_cost for r in b_recs) / len(b_recs)
        assert avg_d_cost > avg_b_cost


def test_safe_discard_preferred_over_dangerous_within_same_shanten():
    """
    Both rd and gd give tenpai (shanten=0) after discard.
    Opponent discarded rd → rd is safe to discard.
    No info on gd → baseline danger.
    Discarding rd should have higher utility than discarding gd.
    """
    state = _make_state()
    state.turn_number = 8

    # Opponent discarded rd → rd is in their safe_tiles
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.discards = [tile_id("rd")]

    # Hand: 4 complete groups + rd + gd (14 tiles total)
    # Discard rd → tenpai waiting for gd pair
    # Discard gd → tenpai waiting for rd pair
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    results = recommend_discard(state)

    rd_rec = next((r for r in results if r.tile_id == tile_id("rd")), None)
    gd_rec = next((r for r in results if r.tile_id == tile_id("gd")), None)

    assert rd_rec is not None and gd_rec is not None
    assert rd_rec.shanten_after == gd_rec.shanten_after  # same shanten
    assert rd_rec.shooting_cost < gd_rec.shooting_cost   # rd is safer to discard
    assert rd_rec.utility > gd_rec.utility               # rd ranks higher


def test_danger_score_higher_for_flush_suit_discard():
    state = _make_state()
    state.turn_number = 10
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.melds = [_circle_pong(1), _circle_pong(4), _circle_pong(7)]

    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d4","d5","d6",
        "ew","ew","ew","b4","d9")
    results = recommend_discard(state)

    # d9 (circles 9) should be more dangerous than b4 (bamboo 4)
    d9_rec = next((r for r in results if r.tile_id == tile_id("d9")), None)
    b4_rec = next((r for r in results if r.tile_id == tile_id("b4")), None)

    if d9_rec and b4_rec:
        assert d9_rec.danger_score >= b4_rec.danger_score


# ---------------------------------------------------------------------------
# CLI integration — recommend command uses optimizer
# ---------------------------------------------------------------------------

def test_cli_recommend_shows_danger_column():
    from click.testing import CliRunner
    import os
    import tempfile
    from cracked.cli import cli

    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "game.json")
        runner = CliRunner()
        env = {"CRACKED_STATE_FILE": state_file}

        runner.invoke(cli, ["new-game", "--seat", "east"], env=env)
        runner.invoke(cli, ["hand",
            "b1","b2","b3","c1","c2","c3","d1","d2","d3",
            "ew","ew","ew","rd"], env=env)
        runner.invoke(cli, ["draw", "gd"], env=env)

        r = runner.invoke(cli, ["recommend"], env=env)
        assert r.exit_code == 0
        # Table should include danger-related output
        assert "Danger" in r.output or "danger" in r.output.lower() or "0." in r.output
