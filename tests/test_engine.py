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
    # 148 total (136 standard + 12 bonus) - 52 standard dealt - bonus consumed during deal
    # Bonus tiles consumed during dealing shrink the wall, so remainder is in [84, 96]
    assert 84 <= engine.wall_remaining <= 96

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
    assert 84 <= engine.wall_remaining <= 96
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
    assert 84 <= view.wall_tiles_remaining <= 96


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


# ---------------------------------------------------------------------------
# Discard-win priority (head-bump): closest player to the discarder wins
# ---------------------------------------------------------------------------

def test_discard_win_priority_closest_to_discarder():
    from cracked.tiles import tiles_from_names, tile_id
    # Half-flush bamboo wait on b1/b4; both EAST and NORTH are ready.
    wait = ["b1","b2","b3","b4","b5","b6","b7","b8","b9","b2","b3","rd","rd"]
    eng = GameEngine(seed=1)
    eng.deal()
    for w in (EAST, NORTH):
        eng.players[w].hand.concealed[:] = tiles_from_names(wait)
    # WEST discards b4. Turn order E->S->W->N: NORTH plays right after WEST,
    # so NORTH should win, not EAST (absolute order would wrongly pick EAST).
    eng.players[WEST].hand.concealed[tile_id("b4")] = 1
    eng._execute_discard(WEST, forced=tile_id("b4"))
    assert eng.winner == NORTH


# ---------------------------------------------------------------------------
# Eight-flower instant win (花胡) and first-round timing
# ---------------------------------------------------------------------------

def test_eight_flower_instant_win():
    eng = GameEngine(seed=1)
    eng.deal()
    # Give EAST seven flowers; stack the eighth as the next wall tile.
    eng.players[EAST].hand.flowers = [34, 35, 36, 37, 38, 39, 40]  # missing 41
    eng._wall[eng._wall_idx] = 41
    events, exhausted = eng._draw_tile(EAST)
    assert not exhausted
    assert eng.is_finished
    assert eng.winner == EAST
    assert any(e.detail.get("eight_flowers") for e in events if e.detail)

def test_discard_win_timing_earthly_vs_humanly():
    eng = GameEngine(seed=1)
    eng.deal()
    # Dealer's (EAST) very first discard, turn 1: a non-dealer win is earthly.
    eng.turn_number = 1
    eng.players[EAST].discards.append(0)  # one discard total
    is_earthly, is_humanly = eng._discard_win_timing(EAST, SOUTH)
    assert is_earthly and not is_humanly
    # A first-round win on a non-dealer's discard, claimer hasn't discarded: humanly.
    eng2 = GameEngine(seed=1); eng2.deal()
    eng2.turn_number = 2
    eng2.players[EAST].discards.append(0)
    eng2.players[SOUTH].discards.append(0)
    e2, h2 = eng2._discard_win_timing(SOUTH, WEST)
    assert h2 and not e2


# ---------------------------------------------------------------------------
# Own-turn kongs (暗杠 / 加杠) and robbing the kong (搶槓)
# ---------------------------------------------------------------------------

def test_do_concealed_kong_mechanic():
    from cracked.tiles import tile_id
    from cracked.hand import MeldType
    eng = GameEngine(seed=1); eng.deal()
    c5 = tile_id("c5")
    eng.players[EAST].hand.concealed[c5] = 4
    n_before = len(eng.players[EAST].hand.melds)
    eng._do_concealed_kong(EAST, c5)
    melds = eng.players[EAST].hand.melds
    assert len(melds) == n_before + 1
    assert melds[-1].type == MeldType.KONG and melds[-1].concealed
    assert eng.players[EAST].hand.concealed[c5] == 0  # all four moved into the meld

def test_do_promoted_kong_mechanic():
    from cracked.tiles import tile_id
    from cracked.hand import Meld, MeldType
    eng = GameEngine(seed=1); eng.deal()
    c5 = tile_id("c5")
    eng.players[EAST].hand.melds = [Meld(MeldType.PONG, (c5, c5, c5), concealed=False,
                                         source_player=NORTH)]
    eng.players[EAST].hand.concealed[:] = 0
    eng.players[EAST].hand.concealed[c5] = 1
    eng._do_promoted_kong(EAST, c5)
    melds = eng.players[EAST].hand.melds
    assert melds[0].type == MeldType.KONG and not melds[0].concealed
    assert eng.players[EAST].hand.concealed[c5] == 0

def test_robbing_the_kong():
    from cracked.tiles import tile_id, tiles_from_names
    from cracked.hand import Meld, MeldType
    eng = GameEngine(seed=1); eng.deal()
    c5 = tile_id("c5")
    # EAST has an exposed pong of c5 and holds the 4th → promotes (加杠).
    eng.players[EAST].hand.melds = [Meld(MeldType.PONG, (c5, c5, c5), concealed=False,
                                         source_player=NORTH)]
    eng.players[EAST].hand.concealed[:] = 0
    eng.players[EAST].hand.concealed[c5] = 1
    # SOUTH waits on c5 (kanchan c4_c6), half-flush characters.
    eng.players[SOUTH].hand.concealed[:] = tiles_from_names(
        ["c1","c2","c3","c4","c6","c7","c8","c9","ew","ew","ew","rd","rd"])
    events = eng._offer_self_kongs(EAST)
    assert eng.winner == SOUTH
    assert any(e.detail.get("robbing_kong") for e in events if e.detail)


# ---------------------------------------------------------------------------
# Human win prompt (hu) + decline
# ---------------------------------------------------------------------------

def test_human_ron_prompts_and_submit_wins():
    from cracked.tiles import tiles_from_names, tile_id
    eng = GameEngine(human_seats={NORTH}, seed=1)
    eng.deal()
    wait = ["b1","b2","b3","b4","b5","b6","b7","b8","b9","b2","b3","rd","rd"]  # half-flush, waits b1/b4
    eng.players[NORTH].hand.concealed[:] = tiles_from_names(wait)
    eng.players[WEST].hand.concealed[tile_id("b4")] = 1
    events = eng._execute_discard(WEST, forced=tile_id("b4"))
    assert eng.awaiting_human_win
    assert any(e.type == EventType.AWAIT_WIN for e in events)
    assert eng.winner is None            # not auto-declared
    win_events = eng.submit_win()
    assert eng.winner == NORTH
    assert any(e.type == EventType.WIN_DISCARD for e in win_events)

def test_human_ron_decline_continues_play():
    from cracked.tiles import tiles_from_names, tile_id
    eng = GameEngine(human_seats={NORTH}, seed=1)
    eng.deal()
    wait = ["b1","b2","b3","b4","b5","b6","b7","b8","b9","b2","b3","rd","rd"]
    eng.players[NORTH].hand.concealed[:] = tiles_from_names(wait)
    eng.players[WEST].hand.concealed[tile_id("b4")] = 1
    eng._execute_discard(WEST, forced=tile_id("b4"))
    assert eng.awaiting_human_win
    eng.decline_win()
    assert not eng.awaiting_human_win
    assert eng.winner is None            # declined → no win, play continues

def test_human_self_draw_decline_then_discard():
    from cracked.tiles import tiles_from_names, tile_id
    eng = GameEngine(human_seats={EAST}, seed=1)
    eng.deal()
    # EAST waits on b4 (full flush); stack b4 as the next wall tile so the draw wins.
    eng.players[EAST].hand.concealed[:] = tiles_from_names(
        ["b1","b2","b3","b4","b5","b6","b7","b8","b9","b1","b2","b3","b4"])
    eng._wall[eng._wall_idx] = tile_id("b4")
    events = eng.step()
    assert eng.awaiting_human_win
    assert any(e.type == EventType.AWAIT_WIN for e in events)
    out = eng.decline_win()
    # Declined self-draw → EAST must now discard.
    assert eng.awaiting_human_discard
    assert any(e.type == EventType.AWAIT_DISCARD for e in out)


# ---------------------------------------------------------------------------
# Sacred / missed discard prohibition (回头牌 / 过水牌)
# ---------------------------------------------------------------------------

def test_sacred_discard_blocks_ron_same_go_around():
    from cracked.tiles import tiles_from_names, tile_id
    eng = GameEngine(seed=1); eng.deal()
    b4 = tile_id("b4")
    # NORTH is ready on b1/b4 but has just discarded b4 itself → 回头牌.
    eng.players[NORTH].hand.concealed[:] = tiles_from_names(
        ["b1","b2","b3","b4","b5","b6","b7","b8","b9","b2","b3","rd","rd"])
    eng._prohibited[NORTH].add(b4)
    assert not eng._can_ron(NORTH, WEST, b4)        # barred from winning b4
    eng._prohibited[NORTH].clear()
    assert eng._can_ron(NORTH, WEST, b4)            # allowed again next turn

def test_missed_pong_prohibits_pong_until_next_turn():
    from cracked.tiles import tile_id
    eng = GameEngine(seed=1); eng.deal()
    c5 = tile_id("c5")
    eng.players[SOUTH].hand.concealed[:] = 0
    eng.players[SOUTH].hand.concealed[c5] = 2       # SOUTH could pong c5
    # SOUTH passes a c5 pong → 过水牌 for c5.
    eng._prohibited[SOUTH].add(c5)
    dps = eng._claim_decision_points(c5, EAST)
    assert all(seat != SOUTH for seat, _ in dps)    # SOUTH not offered the pong
    eng._prohibited[SOUTH].clear()
    dps2 = eng._claim_decision_points(c5, EAST)
    assert any(seat == SOUTH and "pong" in k for seat, k in dps2)


# ---------------------------------------------------------------------------
# Nested claim: AI pongs, then the AI's discard is claimable by the human.
# The human's AWAIT_CLAIM must NOT be wiped by _claim_done (regression).
# ---------------------------------------------------------------------------

def test_nested_human_claim_not_orphaned():
    from cracked.tiles import tiles_from_names, tile_id
    from cracked.hand import Meld, MeldType
    import numpy as np

    # Force SOUTH (AI) to pong East's c9, then discard a tile NORTH (human) can pong.
    eng = GameEngine(human_seats={NORTH}, seed=1)
    eng.deal()
    c9 = tile_id("c9"); ww = tile_id("ww")
    # SOUTH: c9 pair (to pong, but NOT a winning hand) + a ww it will shed after ponging.
    eng.players[SOUTH].hand.concealed[:] = tiles_from_names(
        ["c9","c9","b1","b1","b1","c1","c1","c1","d3","d5","d7","ww","ww"])
    # NORTH (human) holds a ww pair → can pong whatever ww SOUTH throws.
    eng.players[NORTH].hand.concealed[:] = tiles_from_names(
        ["ww","ww","b2","b3","b4","b5","b6","c2","c3","c4","d7","d8","d9"])
    # Make SOUTH's policy deterministically pong and then discard ww.
    class _PongThenWW:
        def choose_discard(self, view): return ww
        def wants_pong(self, view, tile): return tile == c9
        def wants_kong(self, view, tile): return False
        def choose_chow(self, view, tile, options): return None
        def choose_self_kong(self, view): return None
    eng.policies[SOUTH] = _PongThenWW()

    eng._seat_idx = 0  # East to act
    eng.players[EAST].hand.concealed[c9] = 1   # ensure East holds the tile it discards
    eng._execute_discard(EAST, forced=c9)

    # SOUTH ponged c9 and discarded ww; NORTH must be prompted to claim ww — and the
    # engine must actually be paused (flag set), not just show a stale event.
    assert eng.awaiting_human_claim, "human claim was orphaned by _claim_done"
    assert eng.pending_claim_kinds and "pong" in eng.pending_claim_kinds
