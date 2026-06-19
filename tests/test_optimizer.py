"""Tests for the heuristic discard optimizer."""

import pytest

from cracked.tiles import tile_id, Wind
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView
from cracked.opponent_model import OpponentModel, model_all_opponents
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
    alpha_dangerous = adaptive_alpha(state, models_dangerous, best_tiles_away=1)

    state2 = _make_state()
    state2.turn_number = 10
    models_safe = model_all_opponents(state2)
    alpha_safe = adaptive_alpha(state2, models_safe, best_tiles_away=1)

    assert alpha_dangerous < alpha_safe


def test_alpha_higher_when_hand_closer_to_waiting():
    state = _make_state()
    models = model_all_opponents(state)
    alpha_waiting = adaptive_alpha(state, models, best_tiles_away=0)
    alpha_far = adaptive_alpha(state, models, best_tiles_away=3)
    assert alpha_waiting > alpha_far


def test_alpha_lower_with_thin_wall():
    state_full = _make_state()
    state_full.wall_tiles_remaining = 136
    state_thin = _make_state()
    state_thin.wall_tiles_remaining = 10

    models = model_all_opponents(state_full)
    alpha_full = adaptive_alpha(state_full, models, best_tiles_away=1)
    alpha_thin = adaptive_alpha(state_thin, models, best_tiles_away=1)
    assert alpha_thin < alpha_full


def test_alpha_bounded():
    state = _make_state()
    models = model_all_opponents(state)
    for tiles_away_val in (-1, 0, 1, 3, 8):
        a = adaptive_alpha(state, models, tiles_away_val)
        assert 0.15 <= a <= 0.90


def test_alpha_lower_against_high_value_waiting_opponent():
    # Same waiting probability, higher estimated tai → α must drop (the
    # exponential, shooter-pays-all cost dominates). Point 3.
    state = _make_state()
    state.wall_tiles_remaining = 80
    big = [OpponentModel(seat=Wind.SOUTH, waiting_prob=0.9, est_tai=4.0,
                         suit_bias=None, honor_bias=False)]
    small = [OpponentModel(seat=Wind.SOUTH, waiting_prob=0.9, est_tai=1.0,
                           suit_bias=None, honor_bias=False)]
    assert adaptive_alpha(state, big, best_tiles_away=1) < \
           adaptive_alpha(state, small, best_tiles_away=1)


def test_alpha_lower_with_big_chip_lead():
    # Protect a lead (lower α); gamble when behind (higher α). Point 6.
    state = _make_state()
    models = model_all_opponents(state)
    a_lead = adaptive_alpha(state, models, best_tiles_away=1, chip_lead=0.5)
    a_behind = adaptive_alpha(state, models, best_tiles_away=1, chip_lead=-0.5)
    assert a_lead < a_behind


def test_recommendation_is_placement_aware():
    # Same hand vs a circle-flush threat. With a big chip lead the optimizer
    # protects it by choosing the safe discard; far behind it gambles on the
    # higher-acceptance (more dangerous) tile. Point 6.
    def top(my_chips: int) -> DiscardRecommendation:
        state = _make_state()
        state.turn_number = 14
        opp = state.opponent_by_seat(Wind.SOUTH)
        opp.melds = [_circle_pong(2), _circle_pong(5), _circle_pong(8)]
        _set_hand(state,
            "b1","b5","b9","c4","c4","c5","d3","d3","d6","d7","d8","d9","sw","sw")
        return recommend_discard(state, my_chips=my_chips, opponent_chips=[500, 500, 500])[0]

    lead = top(750)     # chip_lead = +0.5 → protect the lead
    behind = top(250)   # chip_lead = -0.5 → chase

    assert lead.tiles_away_after == behind.tiles_away_after
    assert lead.tile_id != behind.tile_id
    assert lead.shooting_cost < behind.shooting_cost


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
    assert hasattr(r, "tiles_away_after")
    assert hasattr(r, "weighted_acceptance")
    assert hasattr(r, "shooting_cost")
    assert hasattr(r, "danger_score")
    assert hasattr(r, "utility")
    assert hasattr(r, "acceptance")


def test_recommend_sorted_by_tiles_away_first():
    state = _make_state()
    # Disorganized hand — different discards will give different tiles_away
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","sw","ww","nw","rd")
    results = recommend_discard(state)
    tiles_aways = [r.tiles_away_after for r in results]
    assert tiles_aways == sorted(tiles_aways)


def test_recommend_utility_between_zero_and_one():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    results = recommend_discard(state)
    for r in results:
        assert 0.0 <= r.utility <= 1.0


# ---------------------------------------------------------------------------
# recommend_discard — tiles_away / waiting hands
# ---------------------------------------------------------------------------

def test_recommend_waiting_hand_tiles_away_zero():
    # After discarding the right tile, hand should be tiles_away=0
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","rd")
    results = recommend_discard(state)
    # Best discard brings us to tiles_away=-1 (complete) or 0 (waiting)
    assert results[0].tiles_away_after <= 0


def test_recommend_best_is_waiting_for_near_complete_hand():
    # Four complete groups + rd-rd pair in hand already → best discard gives waiting (tiles_away=0)
    # The 14-tile hand IS already complete (tiles_away=-1), but recommend evaluates
    # 13-tile hands after each discard; those land at tiles_away=0.
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","rd")
    results = recommend_discard(state)
    assert results[0].tiles_away_after == 0


# ---------------------------------------------------------------------------
# recommend_discard — danger awareness
# ---------------------------------------------------------------------------

def test_shooting_cost_nonzero_with_dangerous_opponent():
    state = _make_state()
    state.turn_number = 10
    opp = state.opponent_by_seat(Wind.SOUTH)
    # Opponent going for circles flush with 3 melds. Pong ranks 4/5/6 (not held
    # in our hand) so no circle tile reaches 4 visible — this isolates flush
    # danger from the wall-read effect (which legitimately frees walled tiles).
    opp.melds = [_circle_pong(4), _circle_pong(5), _circle_pong(6)]

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


def test_safe_discard_preferred_over_dangerous_within_same_tiles_away():
    """
    Both rd and gd give waiting (tiles_away=0) after discard.
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
    # Discard rd → waiting waiting for gd pair
    # Discard gd → waiting waiting for rd pair
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    results = recommend_discard(state)

    rd_rec = next((r for r in results if r.tile_id == tile_id("rd")), None)
    gd_rec = next((r for r in results if r.tile_id == tile_id("gd")), None)

    assert rd_rec is not None and gd_rec is not None
    assert rd_rec.tiles_away_after == gd_rec.tiles_away_after  # same tiles_away
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
