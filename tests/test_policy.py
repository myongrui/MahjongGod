"""Tests for seat policies and their integration with the engine."""

import numpy as np
import pytest

from cracked.tiles import tile_id, tiles_from_names, Wind, NTILES
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView
from cracked.optimizer import recommend_discard
from cracked.policy import HumanPolicy, HeuristicPolicy
from cracked.engine import GameEngine, EventType


EAST = int(Wind.EAST)
SOUTH = int(Wind.SOUTH)


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


def _circle_pong(rank: int) -> Meld:
    tid = tile_id(f"d{rank}")
    return Meld(MeldType.PONG, (tid, tid, tid))


class FixedPolicy:
    """Test policy that always discards a preset tile (must be in hand)."""

    def __init__(self, tile: int):
        self.tile = tile

    def choose_discard(self, view):
        return self.tile

    def wants_pong(self, view, tile):
        return False

    def wants_kong(self, view, tile):
        return False

    def choose_chow(self, view, tile, options):
        return None


# ---------------------------------------------------------------------------
# HeuristicPolicy — discards
# ---------------------------------------------------------------------------

def test_heuristic_discard_matches_optimizer():
    state = _make_state()
    _set_hand(state, "b1", "b2", "b3", "c1", "c2", "c3", "d1", "d2", "d3",
              "ew", "ew", "ew", "rd", "gd")
    expected = recommend_discard(state)[0].tile_id
    assert HeuristicPolicy().choose_discard(state) == expected


def test_heuristic_discard_avoids_dangerous_tile_when_tied():
    """With two discards at equal tiles-away, throw the safe one, not the tile
    in the opponent's flush suit."""
    state = _make_state()
    opp = state.opponent_by_seat(Wind.SOUTH)
    opp.melds = [_circle_pong(1), _circle_pong(3), _circle_pong(6)]  # circles flush
    state.turn_number = 10
    # 4 complete sets + two isolated singles: d5 (a circle → dangerous) and
    # b9 (a terminal bamboo → safe). Discarding either keeps the hand waiting.
    _set_hand(state, "b1", "b2", "b3", "b5", "b6", "b7",
              "c1", "c2", "c3", "c5", "c6", "c7", "d5", "b9")
    choice = HeuristicPolicy().choose_discard(state)
    assert choice == tile_id("b9"), f"threw dangerous tile {choice}"


# ---------------------------------------------------------------------------
# HeuristicPolicy — claims
# ---------------------------------------------------------------------------

def test_heuristic_pong_guard_below_two_copies():
    state = _make_state()
    _set_hand(state, "b1", "b2", "b3", "c1", "c2", "c3", "d1", "d2", "d3",
              "ew", "rd", "gd", "wd")  # 13 tiles, single ew
    assert HeuristicPolicy().wants_pong(state, tile_id("ew")) is False


def test_heuristic_pong_declines_on_winning_hand():
    state = _make_state()
    _set_hand(state, "b1", "b2", "b3", "c1", "c2", "c3", "d1", "d2", "d3",
              "ew", "ew", "ew", "rd", "rd")  # already a complete hand
    assert HeuristicPolicy().wants_pong(state, tile_id("ew")) is False


def test_heuristic_wants_pong_when_it_advances():
    # 3 sets + a useful pair to pong + scattered honors → ponging the pair into
    # a 4th set should not worsen tiles away.
    state = _make_state()
    _set_hand(state, "b1", "b2", "b3", "c1", "c2", "c3", "d1", "d2", "d3",
              "c5", "c5", "ew", "ww")  # 13 tiles, c5 pair
    assert HeuristicPolicy().wants_pong(state, tile_id("c5")) is True


def test_heuristic_chow_requires_improvement():
    state = _make_state()
    _set_hand(state, "b1", "b2", "b3", "c1", "c2", "c3", "d1", "d2", "d3",
              "b4", "b5", "ew", "ww")  # 13 tiles; b4 b5 can chow a b3/b6
    options = [(tile_id("b4"), tile_id("b5"), tile_id("b6"))]
    # chow on b6: uses b4,b5 → forms b4b5b6, a 4th set → strict improvement
    assert HeuristicPolicy().choose_chow(state, tile_id("b6"), options) == options[0]


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def test_engine_default_bots_are_heuristic():
    engine = GameEngine(seed=42)
    engine.deal()
    assert all(isinstance(engine.policies[s], HeuristicPolicy) for s in engine.policies)


def test_engine_human_seat_uses_human_policy():
    engine = GameEngine(human_seats={EAST}, seed=42)
    assert isinstance(engine.policies[EAST], HumanPolicy)
    assert isinstance(engine.policies[SOUTH], HeuristicPolicy)


def test_engine_uses_injected_policy():
    engine = GameEngine(seed=42)
    engine.deal()
    target = int(np.argmax(engine.players[EAST].hand.concealed > 0))  # a tile in hand
    engine.policies[EAST] = FixedPolicy(target)
    events = engine.step()  # East draws (target stays in hand) then discards
    discards = [e.tile for e in events if e.type == EventType.DISCARD]
    assert discards and discards[0] == target


def test_engine_full_game_completes_with_heuristic_bots():
    engine = GameEngine(seed=7)
    engine.deal()
    for _ in range(1000):
        if engine.is_finished:
            break
        engine.step()
    assert engine.is_finished


# ---------------------------------------------------------------------------
# ModelPolicy (requires torch)
# ---------------------------------------------------------------------------

def test_model_policy_slots_into_engine():
    pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic
    from cracked.training.policy_model import ModelPolicy

    engine = GameEngine(seed=3, policies={EAST: ModelPolicy(ActorCritic())})
    engine.deal()
    assert isinstance(engine.policies[EAST], ModelPolicy)
    for _ in range(1000):
        if engine.is_finished:
            break
        engine.step()
    assert engine.is_finished
