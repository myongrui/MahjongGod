"""Tests for the turn-by-turn game engine."""
import pytest
from cracked.tiles import Wind, NTILES
from cracked.engine import GameEngine, EventType


EAST = int(Wind.EAST)
SOUTH = int(Wind.SOUTH)
WEST = int(Wind.WEST)
NORTH = int(Wind.NORTH)


# ---------------------------------------------------------------------------
# Deal
# ---------------------------------------------------------------------------

def test_deal_tile_counts():
    engine = GameEngine(seed=42)
    engine.deal()
    for seat in [EAST, SOUTH, WEST, NORTH]:
        assert engine.players[seat].hand.total_concealed == 13

def test_deal_wall_remaining():
    engine = GameEngine(seed=42)
    engine.deal()
    # 136 total - 52 dealt = 84
    assert engine.wall_remaining == 84

def test_deal_no_tile_overflow():
    """No tile type should appear more than 4 times across all hands."""
    engine = GameEngine(seed=42)
    engine.deal()
    import numpy as np
    total = sum(p.hand.concealed for p in engine.players.values())
    assert all(total <= 4), "A tile appears more than 4 times across all hands"

def test_deal_resets_state():
    engine = GameEngine(seed=1)
    engine.deal()
    engine.step()
    engine.deal()
    assert engine.turn_number == 0
    assert engine.wall_remaining == 84
    assert not engine.is_finished


# ---------------------------------------------------------------------------
# Step — spectator mode (all AI)
# ---------------------------------------------------------------------------

def test_step_ai_returns_draw_and_discard():
    engine = GameEngine(seed=42)
    engine.deal()
    events = engine.step()
    types = [e.type for e in events]
    assert EventType.DRAW in types
    assert EventType.DISCARD in types

def test_step_advances_seat():
    engine = GameEngine(seed=42)
    engine.deal()
    assert engine.current_seat == EAST
    engine.step()
    assert engine.current_seat == SOUTH

def test_step_turn_number_increments():
    engine = GameEngine(seed=42)
    engine.deal()
    engine.step()
    assert engine.turn_number == 1
    engine.step()
    assert engine.turn_number == 2

def test_step_finished_returns_empty():
    engine = GameEngine(seed=42)
    engine.deal()
    engine._phase = "finished"
    assert engine.step() == []

def test_wall_exhausted_event():
    engine = GameEngine(seed=42)
    engine.deal()
    # Drain the wall artificially
    engine._wall_idx = len(engine._wall)
    events = engine.step()
    assert any(e.type == EventType.WALL_EXHAUSTED for e in events)
    assert engine.is_finished

def test_full_game_completes():
    """A full AI game must eventually end."""
    engine = GameEngine(seed=7)
    engine.deal()
    for _ in range(500):
        if engine.is_finished:
            break
        engine.step()
    assert engine.is_finished


# ---------------------------------------------------------------------------
# Human seat
# ---------------------------------------------------------------------------

def test_human_seat_returns_await_discard():
    engine = GameEngine(human_seats={EAST}, seed=42)
    engine.deal()
    events = engine.step()
    types = [e.type for e in events]
    assert EventType.DRAW in types
    assert EventType.AWAIT_DISCARD in types
    assert engine.awaiting_human_discard

def test_submit_discard_removes_tile():
    engine = GameEngine(human_seats={EAST}, seed=42)
    engine.deal()
    engine.step()  # East draws → AWAIT_DISCARD
    player = engine.players[EAST]
    tiles_before = player.hand.total_concealed
    # Pick any tile in hand
    tid = next(t for t in range(NTILES) if player.hand.concealed[t] > 0)
    engine.submit_discard(tid)
    assert player.hand.total_concealed == tiles_before - 1
    assert tid in player.discards

def test_submit_discard_wrong_tile_raises():
    engine = GameEngine(human_seats={EAST}, seed=42)
    engine.deal()
    engine.step()
    player = engine.players[EAST]
    # Find a tile NOT in hand
    missing = next(t for t in range(NTILES) if player.hand.concealed[t] == 0)
    with pytest.raises(ValueError):
        engine.submit_discard(missing)

def test_submit_discard_not_awaiting_raises():
    engine = GameEngine(seed=42)
    engine.deal()
    with pytest.raises(ValueError):
        engine.submit_discard(0)

def test_after_human_discard_advances_to_south():
    engine = GameEngine(human_seats={EAST}, seed=42)
    engine.deal()
    engine.step()  # East: AWAIT_DISCARD
    player = engine.players[EAST]
    tid = next(t for t in range(NTILES) if player.hand.concealed[t] > 0)
    engine.submit_discard(tid)
    assert engine.current_seat == SOUTH
    assert not engine.awaiting_human_discard


# ---------------------------------------------------------------------------
# player_view_for
# ---------------------------------------------------------------------------

def test_player_view_hides_opponents_concealed():
    engine = GameEngine(seed=42)
    engine.deal()
    view = engine.player_view_for(EAST)
    assert view.my_seat == EAST
    assert view.my_hand.total_concealed == 13
    assert len(view.opponents) == 3

def test_player_view_wall_remaining():
    engine = GameEngine(seed=42)
    engine.deal()
    view = engine.player_view_for(EAST)
    assert view.wall_tiles_remaining == 84


# ---------------------------------------------------------------------------
# get_recommendations
# ---------------------------------------------------------------------------

def test_get_recommendations_requires_await():
    engine = GameEngine(seed=42)
    engine.deal()
    with pytest.raises(ValueError):
        engine.get_recommendations()

def test_get_recommendations_returns_results():
    engine = GameEngine(human_seats={EAST}, seed=42)
    engine.deal()
    engine.step()  # East draws → AWAIT_DISCARD
    recs = engine.get_recommendations()
    assert len(recs) > 0
    assert all(r.tile_id is not None for r in recs)
