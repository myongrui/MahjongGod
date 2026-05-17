"""Opponent hand inference for Singapore Mahjong."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cracked.tiles import (
    NTILES, WIND_START, WIND_END, DRAGON_START, DRAGON_END,
    is_suited, is_honor, suit_of,
)
from cracked.game_state import GameState, PlayerView


@dataclass
class OpponentModel:
    """Observable inference about one opponent's hand and danger profile."""
    seat: int
    tenpai_prob: float           # estimated probability opponent is tenpai (0–1)
    est_tai: float               # estimated tai value if they win
    suit_bias: Optional[int]     # suspected flush suit: 0=bamboo, 1=chars, 2=circles
    honor_bias: bool             # opponent appears to be collecting honors
    dangerous_tiles: set[int]    # tile IDs likely to complete their hand
    safe_tiles: set[int]         # tile IDs unlikely to be wanted by them


def tenpai_probability(n_melds: int, turn: int) -> float:
    """Heuristic tenpai estimate from meld count and turn number."""
    base = [0.05, 0.12, 0.28, 0.60, 0.90][min(n_melds, 4)]
    pressure = min(turn * 0.007, 0.30)
    return min(base + pressure, 0.95)


def detect_suit_bias(opp: PlayerView) -> tuple[Optional[int], bool]:
    """
    Infer which suit (if any) the opponent appears to be building.

    Meld evidence is primary: if all exposed melds share one suit, that's the bias.
    Discard evidence is secondary: if one suit never appears in discards, they may be hoarding it.

    Returns (suit_bias, honor_bias).
    """
    # Meld evidence (strong signal)
    meld_suits: set[int] = set()
    for meld in opp.melds:
        for t in meld.tiles:
            if is_suited(t):
                meld_suits.add(suit_of(t))
    if len(meld_suits) == 1:
        return meld_suits.pop(), False

    discards = opp.discards
    if len(discards) < 4:
        return None, False

    # Discard evidence: if one suit is entirely absent from discards, they may be hoarding it
    suit_seen = [False, False, False]
    honor_discard_count = 0
    for t in discards:
        if is_suited(t):
            suit_seen[suit_of(t)] = True
        else:
            honor_discard_count += 1

    absent = [s for s in range(3) if not suit_seen[s]]
    honor_bias = honor_discard_count == 0 and len(discards) >= 6

    if len(absent) == 1:
        return absent[0], honor_bias

    return None, honor_bias


def _estimate_tai(opp: PlayerView, suit_bias: Optional[int], prevailing_wind: int) -> float:
    """Rough tai estimate from observable meld patterns."""
    tai = 0.0

    for meld in opp.melds:
        t = meld.tiles[0]
        if DRAGON_START <= t < DRAGON_END:
            tai += 1.0
        elif WIND_START <= t < WIND_END:
            if t == prevailing_wind:
                tai += 1.0
            if t == opp.seat:
                tai += 1.0

    # Flush signal: all melds one suit with a suit bias
    meld_suits = {suit_of(t) for m in opp.melds for t in m.tiles if is_suited(t)}
    if len(meld_suits) == 1 and suit_bias is not None:
        tai += 2.5  # half flush minimum; could be full flush

    return max(tai, 1.0)


def _adjacent_to_melds(tid: int, melds: list) -> bool:
    """True if tid is within sequence range (1–2 ranks) of any meld tile in the same suit."""
    if not is_suited(tid):
        return False
    tid_rank = tid % 9
    for meld in melds:
        for mt in meld.tiles:
            if not is_suited(mt) or suit_of(mt) != suit_of(tid):
                continue
            if abs(tid_rank - (mt % 9)) in (1, 2):
                return True
    return False


def _dragon_danger_flags(
    opp: PlayerView,
    visible: np.ndarray,
) -> tuple[set[int], set[int]]:
    """
    Return (extra_dangerous, extra_safe) sets for dragon tiles.

    extra_safe:
      The 4th copy of a ponged/konged dragon — claiming it only grants a kong
      replacement draw, not a win.

    extra_dangerous (requires 4+ discards before drawing conclusions):
      - Opponent has discarded no dragons: they are holding all three
        intentionally; every undiscarded dragon is a potential pair/triplet wait.
      - Opponent has a visible dragon pong/kong: hints at a three-dragon hand;
        other undiscarded, non-ponged dragons become suspicious.
    """
    discarded_set = set(opp.discards)

    ponged_dragons: set[int] = {
        m.tiles[0]
        for m in opp.melds
        if DRAGON_START <= m.tiles[0] < DRAGON_END
        and m.type.value in ("pong", "kong")
    }

    dragon_discard_count = sum(
        1 for d in range(DRAGON_START, DRAGON_END) if d in discarded_set
    )
    undiscarded_free = [
        d for d in range(DRAGON_START, DRAGON_END)
        if d not in discarded_set and int(visible[d]) < 4 and d not in ponged_dragons
    ]

    extra_safe: set[int] = set(ponged_dragons)
    extra_dangerous: set[int] = set()

    if len(opp.discards) < 4:
        return extra_dangerous, extra_safe

    if dragon_discard_count == 0:
        extra_dangerous.update(undiscarded_free)

    if ponged_dragons:
        extra_dangerous.update(undiscarded_free)

    return extra_dangerous, extra_safe


def _wind_danger_flags(
    opp: PlayerView,
    visible: np.ndarray,
    prevailing_wind: int,
    honor_bias: bool,
) -> tuple[set[int], set[int]]:
    """
    Return (extra_dangerous, extra_safe) sets for wind tiles.

    extra_safe:
      4th copy of a ponged/konged wind — cannot complete a win for them.

    extra_dangerous (requires 4+ discards):
      - Seat wind or prevailing wind not discarded: likely held for tai value.
      - honor_bias is True (6+ discards, zero honors thrown): opponent is
        hoarding honors broadly; all un-discarded winds are suspicious.
      - Visible wind pong: hints at an honor-collecting hand; other un-discarded
        winds escalate.
    """
    discarded_set = set(opp.discards)

    ponged_winds: set[int] = {
        m.tiles[0]
        for m in opp.melds
        if WIND_START <= m.tiles[0] < WIND_END
        and m.type.value in ("pong", "kong")
    }

    extra_safe: set[int] = set(ponged_winds)
    extra_dangerous: set[int] = set()

    if len(opp.discards) < 4:
        return extra_dangerous, extra_safe

    def _free_wind(w: int) -> bool:
        return w not in discarded_set and int(visible[w]) < 4 and w not in ponged_winds

    # Seat wind / prevailing wind kept this long → probably holding for tai
    for w in {opp.seat, prevailing_wind}:
        if WIND_START <= w < WIND_END and _free_wind(w):
            extra_dangerous.add(w)

    # No honors thrown at all → collecting honors broadly
    if honor_bias:
        for w in range(WIND_START, WIND_END):
            if _free_wind(w):
                extra_dangerous.add(w)

    # Visible wind pong → honor-collecting hand likely; other winds suspicious
    if ponged_winds:
        for w in range(WIND_START, WIND_END):
            if _free_wind(w):
                extra_dangerous.add(w)

    return extra_dangerous, extra_safe


def _compute_tile_sets(
    opp: PlayerView,
    suit_bias: Optional[int],
    visible: np.ndarray,
    prevailing_wind: int,
    honor_bias: bool,
) -> tuple[set[int], set[int]]:
    """Compute (dangerous, safe) tile sets for one opponent."""
    dangerous: set[int] = set()
    safe: set[int] = set()
    discarded = set(opp.discards)

    dragon_dangerous, dragon_safe = _dragon_danger_flags(opp, visible)
    wind_dangerous, wind_safe = _wind_danger_flags(opp, visible, prevailing_wind, honor_bias)

    for tid in range(NTILES):
        if visible[tid] >= 4:
            safe.add(tid)
            continue
        if tid in discarded:
            safe.add(tid)
            continue
        if tid in dragon_safe or tid in wind_safe:
            safe.add(tid)
            continue
        if tid in dragon_dangerous or tid in wind_dangerous:
            dangerous.add(tid)
            continue
        if suit_bias is not None and is_suited(tid) and suit_of(tid) == suit_bias:
            dangerous.add(tid)
            continue
        if _adjacent_to_melds(tid, opp.melds):
            dangerous.add(tid)

    return dangerous, safe


def model_opponent(opp: PlayerView, state: GameState) -> OpponentModel:
    """Build an OpponentModel from the current observable game state."""
    visible = state.visible_tiles()
    suit_bias, honor_bias = detect_suit_bias(opp)
    dangerous, safe = _compute_tile_sets(
        opp, suit_bias, visible, state.prevailing_wind, honor_bias
    )
    tp = tenpai_probability(len(opp.melds), state.turn_number)
    est_tai = _estimate_tai(opp, suit_bias, state.prevailing_wind)

    return OpponentModel(
        seat=opp.seat,
        tenpai_prob=tp,
        est_tai=est_tai,
        suit_bias=suit_bias,
        honor_bias=honor_bias,
        dangerous_tiles=dangerous,
        safe_tiles=safe,
    )


def model_all_opponents(state: GameState) -> list[OpponentModel]:
    """Build an OpponentModel for each opponent in the game state."""
    return [model_opponent(opp, state) for opp in state.opponents]
