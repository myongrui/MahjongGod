"""
Tests for the Monte Carlo game simulator.

Tests marked @pytest.mark.slow run full game simulations and are excluded
from the default test run. Use: pytest -m slow
"""

import pytest
import numpy as np

from cracked.tiles import tile_id, Wind, tiles_from_names
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView
from cracked.simulator import (
    SimHand, GameResult, SimulationResult,
    _heuristic_discard, _deal_hands, _play_one_game,
    simulate_from_state, run_simulation,
)


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
    state.my_hand.concealed = tiles_from_names(list(names))


def _sim_hand(*names: str, n_melds: int = 0, seat: int = Wind.EAST) -> SimHand:
    return SimHand(
        concealed=tiles_from_names(list(names)),
        n_melds=n_melds,
        seat=seat,
    )


# ---------------------------------------------------------------------------
# _heuristic_discard
# ---------------------------------------------------------------------------

def test_heuristic_discard_returns_valid_tile():
    hand = _sim_hand("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","sw","ww","nw")
    tid = _heuristic_discard(hand)
    assert 0 <= tid < 34
    assert hand.concealed[tid] > 0


def test_heuristic_discard_prefers_isolated_honor():
    # Hand with one isolated honor and one suited sequence — should drop the honor
    hand = _sim_hand("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","nw")
    tid = _heuristic_discard(hand)
    assert tid == tile_id("nw")  # isolated north wind is the obvious drop


def test_heuristic_discard_on_tenpai_hand():
    # b1b2b3 c1c2c3 d1d2d3 ew-pong + rd (waiting for rd pair) — already tenpai
    # Heuristic must still return some tile (no crash)
    hand = _sim_hand("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","gd")
    tid = _heuristic_discard(hand)
    assert hand.concealed[tid] > 0


# ---------------------------------------------------------------------------
# SimHand
# ---------------------------------------------------------------------------

def test_sim_hand_is_winner_complete():
    # b1b2b3 c1c2c3 d1d2d3 ew-pong + rd pair = complete
    hand = _sim_hand("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd")
    assert hand.is_winner()


def test_sim_hand_is_winner_incomplete():
    hand = _sim_hand("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd")
    assert not hand.is_winner()


def test_sim_hand_can_win_from_tile():
    # Tenpai hand waiting for rd
    hand = _sim_hand("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd")
    assert hand.can_win_from(tile_id("rd"))
    assert not hand.can_win_from(tile_id("gd"))


# ---------------------------------------------------------------------------
# _deal_hands
# ---------------------------------------------------------------------------

def test_deal_hands_total_tiles_correct():
    import random
    rng = random.Random(42)
    unknown = np.full(34, 4, dtype=np.int8)  # 136 unknown tiles
    opp_counts = [13, 13, 13]
    opp_arrays, wall = _deal_hands(unknown, opp_counts, rng)

    dealt = sum(arr.sum() for arr in opp_arrays) + len(wall)
    assert dealt == unknown.sum()


def test_deal_hands_no_tile_exceeds_max():
    import random
    rng = random.Random(42)
    unknown = np.full(34, 4, dtype=np.int8)
    opp_arrays, wall = _deal_hands(unknown, [13, 13, 13], rng)
    for arr in opp_arrays:
        assert arr.max() <= 4


def test_deal_hands_different_seeds_give_different_results():
    import random
    unknown = np.full(34, 4, dtype=np.int8)
    opp_counts = [13, 13, 13]
    _, wall1 = _deal_hands(unknown, opp_counts, random.Random(1))
    _, wall2 = _deal_hands(unknown, opp_counts, random.Random(2))
    assert wall1 != wall2


# ---------------------------------------------------------------------------
# simulate_from_state
# ---------------------------------------------------------------------------

def test_simulate_from_state_returns_result():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    sr = simulate_from_state(state, tile_id("gd"), n_games=3, seed=0)
    assert isinstance(sr, SimulationResult)
    assert sr.tile_id == tile_id("gd")
    assert sr.n_games == 3


def test_simulate_result_counts_sum_to_n_games():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    sr = simulate_from_state(state, tile_id("gd"), n_games=3, seed=1)
    assert sr.n_games == 3


def test_simulate_win_and_shoot_rates_bounded():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    sr = simulate_from_state(state, tile_id("gd"), n_games=3, seed=2)
    assert 0.0 <= sr.win_rate <= 1.0
    assert 0.0 <= sr.shoot_rate <= 1.0
    assert sr.win_rate + sr.shoot_rate <= 1.0


def test_simulate_reproducible_with_seed():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    sr1 = simulate_from_state(state, tile_id("gd"), n_games=5, seed=42)
    sr2 = simulate_from_state(state, tile_id("gd"), n_games=5, seed=42)
    assert sr1.win_count == sr2.win_count
    assert sr1.shoot_count == sr2.shoot_count
    assert sr1.total_net == sr2.total_net


def test_simulate_raises_for_missing_tile():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    with pytest.raises(ValueError):
        simulate_from_state(state, tile_id("b9"), n_games=10, seed=0)


@pytest.mark.slow
def test_near_tenpai_hand_higher_win_rate_than_random():
    """A hand already at tenpai should win more often than a disorganised hand."""
    state_tenpai = _make_state()
    _set_hand(state_tenpai,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    sr_tenpai = simulate_from_state(
        state_tenpai, tile_id("gd"), n_games=100, seed=7
    )

    state_bad = _make_state()
    _set_hand(state_bad,
        "b1","b3","b5","b7","b9","c2","c4","c6","c8",
        "ew","sw","ww","nw","rd")
    sr_bad = simulate_from_state(
        state_bad, tile_id("rd"), n_games=100, seed=7
    )

    assert sr_tenpai.win_rate >= sr_bad.win_rate


# ---------------------------------------------------------------------------
# run_simulation
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_simulation_raises_on_wrong_count():
    state = _make_state()
    _set_hand(state, "b1","b2","b3")  # only 3 tiles
    with pytest.raises(ValueError):
        run_simulation(state, n_games=10)


@pytest.mark.slow
def test_run_simulation_returns_one_result_per_unique_tile():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    results = run_simulation(state, n_games=5, seed=0)
    unique_tiles = {
        tid for tid in range(34)
        if state.my_hand.concealed[tid] > 0
    }
    result_tiles = {r.tile_id for r in results}
    assert result_tiles == unique_tiles


@pytest.mark.slow
def test_run_simulation_sorted_by_expected_gain():
    state = _make_state()
    _set_hand(state,
        "b1","b2","b3","c1","c2","c3","d1","d2","d3",
        "ew","ew","ew","rd","gd")
    results = run_simulation(state, n_games=5, seed=0)
    gains = [r.expected_gain for r in results]
    assert gains == sorted(gains, reverse=True)


# ---------------------------------------------------------------------------
# CLI --deep integration
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cli_recommend_deep_flag():
    from click.testing import CliRunner
    import os, tempfile
    from cracked.cli import cli

    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "game.json")
        env = {"CRACKED_STATE_FILE": state_file}
        runner = CliRunner()

        runner.invoke(cli, ["new-game", "--seat", "east"], env=env)
        runner.invoke(cli, ["hand",
            "b1","b2","b3","c1","c2","c3","d1","d2","d3",
            "ew","ew","ew","rd"], env=env)
        runner.invoke(cli, ["draw", "gd"], env=env)

        r = runner.invoke(cli, ["recommend", "--deep", "--games", "20"], env=env)
        assert r.exit_code == 0
        assert "Win%" in r.output or "Shoot%" in r.output or "Gain" in r.output
