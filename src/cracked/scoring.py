"""
Singapore Mahjong tai (台) scoring engine.

Evaluates a completed winning hand and returns an itemised tai breakdown.
House rules are configurable via HouseRules; defaults match the most common
Singapore ruleset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cracked.tiles import (
    NTILES, WIND_START, WIND_END, DRAGON_START, DRAGON_END,
    SUITED_END, is_suited, is_honor, is_terminal, suit_of,
    Wind, Dragon, tile_name,
    SEAT_FLOWER, SEAT_SEASON, new_hand_array,
)
from cracked.hand import HandState, Meld, MeldType


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HouseRules:
    tai_cap: int = 5
    min_tai: int = 1
    seven_pairs_base: int = 3       # some houses use 2
    flowers_above_cap: bool = False  # flower/season tai added after cap is applied


DEFAULT_RULES = HouseRules()


# ---------------------------------------------------------------------------
# Chip payment system
# ---------------------------------------------------------------------------

STARTING_CHIPS: int = 500

# Shooter scale:  4 / 8 / 16 / 32 / 64  (tai 1–5)
# Zimo per player: 2 / 4 /  8 / 16 / 32  (tai 1–5)
_SHOOTER_BASE = 4
_ZIMO_BASE = 2


def chip_payment(tai: int) -> tuple[int, int]:
    """
    Returns (shooter_pay, zimo_pay_per_player) for the given tai level.

    shooter_pay         — what the discarder pays to the winner (ron)
    zimo_pay_per_player — what each opponent pays for a self-draw win
    """
    multiplier = 1 << max(tai - 1, 0)
    return _SHOOTER_BASE * multiplier, _ZIMO_BASE * multiplier


# ---------------------------------------------------------------------------
# Win context
# ---------------------------------------------------------------------------

@dataclass
class WinContext:
    winning_tile: int
    is_self_draw: bool = False
    is_last_tile: bool = False      # 海底撈月
    is_replacement: bool = False    # 嶺上開花 (after kong draw)
    is_robbing_kong: bool = False   # 搶槓
    prevailing_wind: int = Wind.EAST


# ---------------------------------------------------------------------------
# Internal decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Group:
    gtype: str          # "chow", "pong", "kong", "pair"
    tiles: tuple[int, ...]
    concealed: bool = True

    def is_triplet(self) -> bool:
        return self.gtype in ("pong", "kong")

    def head(self) -> int:
        return self.tiles[0]


@dataclass
class _Decomposition:
    groups: list[_Group]    # exactly 4 complete groups
    pair: _Group


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TaiResult:
    total: int
    breakdown: list[tuple[str, int]]
    capped: bool = False

    def is_valid_win(self, rules: HouseRules = DEFAULT_RULES) -> bool:
        return self.total >= rules.min_tai


# ---------------------------------------------------------------------------
# Hand decomposition helpers
# ---------------------------------------------------------------------------

def _extract_groups(tiles: np.ndarray, n: int) -> list[list[_Group]]:
    """
    Recursively find all ways to extract exactly n groups from tiles.
    tiles is modified in-place and restored; caller should pass a copy.
    """
    if n == 0:
        return [[]] if tiles.sum() == 0 else []

    results: list[list[_Group]] = []

    for i in range(NTILES):
        if tiles[i] == 0:
            continue

        # Pong
        if tiles[i] >= 3:
            tiles[i] -= 3
            for sub in _extract_groups(tiles, n - 1):
                results.append([_Group("pong", (i, i, i))] + sub)
            tiles[i] += 3

        # Sequence (suited, same suit, consecutive)
        if (i < SUITED_END and i + 2 < SUITED_END
                and suit_of(i) == suit_of(i + 1) == suit_of(i + 2)
                and tiles[i + 1] > 0 and tiles[i + 2] > 0):
            tiles[i] -= 1; tiles[i + 1] -= 1; tiles[i + 2] -= 1
            for sub in _extract_groups(tiles, n - 1):
                results.append([_Group("chow", (i, i + 1, i + 2))] + sub)
            tiles[i] += 1; tiles[i + 1] += 1; tiles[i + 2] += 1

        break  # only process from the first non-zero tile

    return results


def _decompose_concealed(concealed: np.ndarray, n_groups: int) -> list[_Decomposition]:
    """All valid (n_groups groups + 1 pair) decompositions of concealed tiles."""
    results = []
    arr = concealed.copy()

    for pair_tid in range(NTILES):
        if arr[pair_tid] < 2:
            continue
        arr[pair_tid] -= 2
        pair = _Group("pair", (pair_tid, pair_tid))
        for groups in _extract_groups(arr.copy(), n_groups):
            results.append(_Decomposition(groups, pair))
        arr[pair_tid] += 2

    return results


def _meld_to_group(meld: Meld) -> _Group:
    gtype = {
        MeldType.CHOW: "chow",
        MeldType.PONG: "pong",
        MeldType.KONG: "kong",
    }[meld.type]
    return _Group(gtype, meld.tiles, concealed=meld.concealed)


# ---------------------------------------------------------------------------
# Limit-hand detection (checked before full decomposition)
# ---------------------------------------------------------------------------

def _is_limit_hand(hand: HandState, ctx: WinContext) -> Optional[str]:
    """Return a limit-hand name if detected, else None."""
    concealed = hand.concealed
    all_tiles = hand.concealed_tiles_list()
    exposed_tiles = [t for m in hand.melds for t in m.tiles]
    every_tile = all_tiles + exposed_tiles

    # Thirteen orphans
    from cracked.shanten import shanten_thirteen_orphans
    if not hand.melds and shanten_thirteen_orphans(concealed) == -1:
        return "Thirteen orphans (十三幺)"

    # All honors: every tile is wind or dragon
    if all(is_honor(t) for t in every_tile):
        return "All honors (字一色)"

    # Big three dragons: all three dragon types appear as pong/kong
    meld_groups = [_meld_to_group(m) for m in hand.melds]
    dragon_pong_tids = {
        g.head() for g in meld_groups
        if g.is_triplet() and DRAGON_START <= g.head() < DRAGON_END
    }
    for t in range(DRAGON_START, DRAGON_END):
        if concealed[t] >= 3:
            dragon_pong_tids.add(t)
    if len(dragon_pong_tids) == 3:
        return "Big three dragons (大三元)"

    # Big four winds
    wind_pong_tids = {
        g.head() for g in meld_groups
        if g.is_triplet() and WIND_START <= g.head() < WIND_END
    }
    for t in range(WIND_START, WIND_END):
        if concealed[t] >= 3:
            wind_pong_tids.add(t)
    if len(wind_pong_tids) == 4:
        return "Big four winds (大四喜)"

    # All kongs: all 4 groups are kongs
    if len(hand.melds) == 4 and all(m.type == MeldType.KONG for m in hand.melds):
        return "All kongs (槓上槓)"

    return None


# ---------------------------------------------------------------------------
# Scoring a single decomposition
# ---------------------------------------------------------------------------

def _score_decomp(
    decomp: _Decomposition,
    hand: HandState,
    ctx: WinContext,
    rules: HouseRules,
) -> tuple[int, list[tuple[str, int]]]:
    """Score one decomposition. Returns (base_tai, breakdown) — not yet capped."""
    bd: list[tuple[str, int]] = []
    groups = decomp.groups
    pair = decomp.pair
    pair_tile = pair.head()

    all_grp_tiles = [t for g in groups for t in g.tiles]
    all_hand_tiles = all_grp_tiles + list(pair.tiles)

    # --- Win context ---
    if ctx.is_self_draw:
        bd.append(("Self-draw (自摸)", 1))
    if ctx.is_last_tile:
        bd.append(("Last tile — 海底撈月", 1))
    if ctx.is_replacement:
        bd.append(("Replacement tile — 嶺上開花", 1))
    if ctx.is_robbing_kong:
        bd.append(("Robbing a kong — 搶槓", 1))

    # --- Dragon pongs ---
    dragon_pong_count = 0
    for g in groups:
        if g.is_triplet() and DRAGON_START <= g.head() < DRAGON_END:
            bd.append((f"{tile_name(g.head())} dragon pong", 1))
            dragon_pong_count += 1
    dragon_pair = DRAGON_START <= pair_tile < DRAGON_END

    # Small three dragons (2 dragon pongs + dragon pair)
    if dragon_pong_count == 2 and dragon_pair:
        bd.append(("Small three dragons (小三元)", 3))

    # --- Wind pongs ---
    for g in groups:
        if not g.is_triplet():
            continue
        t = g.head()
        if not (WIND_START <= t < WIND_END):
            continue
        tai = 0
        if t == hand.seat_wind:
            tai += 1
        if t == ctx.prevailing_wind:
            tai += 1
        if tai > 0:
            wind_names = {27: "East", 28: "South", 29: "West", 30: "North"}
            suffix = " (double wind)" if tai == 2 else ""
            bd.append((f"{wind_names[t]} wind pong{suffix}", tai))

    # --- All pongs ---
    if all(g.is_triplet() for g in groups):
        bd.append(("All pongs (對對胡)", 2))

    # --- Flush detection ---
    suits_in_hand = {suit_of(t) for t in all_hand_tiles if is_suited(t)}
    honors_in_hand = any(is_honor(t) for t in all_hand_tiles)

    if not honors_in_hand and len(suits_in_hand) == 1:
        bd.append(("Full flush (清一色)", 4))
    elif honors_in_hand and len(suits_in_hand) == 1:
        bd.append(("Half flush (混一色)", 2))

    # --- Small four winds (3 wind pongs + wind pair) ---
    wind_pong_count = sum(
        1 for g in groups
        if g.is_triplet() and WIND_START <= g.head() < WIND_END
    )
    wind_pair = WIND_START <= pair_tile < WIND_END
    if wind_pong_count == 3 and wind_pair:
        bd.append(("Small four winds (小四喜)", rules.tai_cap))

    total = sum(t for _, t in bd)
    return total, bd


# ---------------------------------------------------------------------------
# Seven pairs scoring
# ---------------------------------------------------------------------------

def _score_seven_pairs(
    hand: HandState,
    ctx: WinContext,
    rules: HouseRules,
) -> tuple[int, list[tuple[str, int]]]:
    """Score a seven-pairs winning hand."""
    bd: list[tuple[str, int]] = [("Seven pairs (七對子)", rules.seven_pairs_base)]

    concealed = hand.concealed
    all_hand_tiles = hand.concealed_tiles_list()

    if ctx.is_self_draw:
        bd.append(("Self-draw (自摸)", 1))
    if ctx.is_last_tile:
        bd.append(("Last tile — 海底撈月", 1))
    if ctx.is_replacement:
        bd.append(("Replacement tile — 嶺上開花", 1))
    if ctx.is_robbing_kong:
        bd.append(("Robbing a kong — 搶槓", 1))

    # Dragon pairs
    for t in range(DRAGON_START, DRAGON_END):
        if concealed[t] >= 2:
            bd.append((f"{tile_name(t)} dragon pair", 1))

    # Wind pairs
    for t in range(WIND_START, WIND_END):
        if concealed[t] < 2:
            continue
        tai = 0
        if t == hand.seat_wind:
            tai += 1
        if t == ctx.prevailing_wind:
            tai += 1
        if tai:
            wind_names = {27: "East", 28: "South", 29: "West", 30: "North"}
            bd.append((f"{wind_names[t]} wind pair", tai))

    # Flush
    suits = {suit_of(t) for t in all_hand_tiles if is_suited(t)}
    honors = any(is_honor(t) for t in all_hand_tiles)
    if not honors and len(suits) == 1:
        bd.append(("Full flush (清一色)", 4))
    elif honors and len(suits) == 1:
        bd.append(("Half flush (混一色)", 2))

    total = sum(t for _, t in bd)
    return total, bd


# ---------------------------------------------------------------------------
# Bonus tile scoring: flowers, seasons, animals (added on top of capped base tai)
# ---------------------------------------------------------------------------

def _score_bonus_tiles(hand: HandState) -> tuple[int, list[tuple[str, int]]]:
    """
    Returns (bonus_tai, breakdown_items).
    bonus_tai is added after the base tai cap is applied.

    Rules:
    - Matching flower (seat's flower): 1 tai
    - Matching season (seat's season): 1 tai
    - Non-matching flowers/seasons: 0 tai
    - Each animal tile: 1 tai always
    """
    bd: list[tuple[str, int]] = []

    seat = hand.seat_wind
    seat_flower = SEAT_FLOWER.get(seat)
    seat_season = SEAT_SEASON.get(seat)

    for f in hand.flowers:
        if f == seat_flower:
            bd.append(("Seat flower", 1))
        elif f == seat_season:
            bd.append(("Seat season", 1))
        # non-matching flower/season: 0 tai, not added

    for _ in hand.animals:
        bd.append(("Animal", 1))

    bonus_tai = sum(t for _, t in bd)
    return bonus_tai, bd


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calculate_tai(
    hand: HandState,
    ctx: WinContext,
    rules: HouseRules = DEFAULT_RULES,
) -> TaiResult:
    """
    Calculate the tai for a completed winning hand.

    hand.concealed must contain the final 14 tiles (including the winning tile).
    ctx provides win circumstances.
    """
    # --- Limit hands (fast path) ---
    limit = _is_limit_hand(hand, ctx)
    if limit:
        flower_tai, flower_bd = _score_bonus_tiles(hand)
        base_bd: list[tuple[str, int]] = [(limit, rules.tai_cap)]
        ctx_tai = 0
        if ctx.is_self_draw:
            ctx_tai += 1
            base_bd.append(("Self-draw (自摸)", 1))
        if ctx.is_last_tile:
            ctx_tai += 1
            base_bd.append(("Last tile — 海底撈月", 1))
        base = min(rules.tai_cap + ctx_tai, rules.tai_cap)
        total = base + flower_tai
        return TaiResult(total, base_bd + flower_bd, capped=True)

    # --- Seven pairs ---
    from cracked.shanten import shanten_seven_pairs
    seven_pairs_total = -999
    seven_pairs_bd: list[tuple[str, int]] = []
    if not hand.melds and shanten_seven_pairs(hand.concealed) == -1:
        seven_pairs_total, seven_pairs_bd = _score_seven_pairs(hand, ctx, rules)

    # --- Standard form: find all decompositions ---
    n_concealed_groups = 4 - len(hand.melds)
    exposed = [_meld_to_group(m) for m in hand.melds]

    best_base = -999
    best_bd: list[tuple[str, int]] = []

    for cd in _decompose_concealed(hand.concealed, n_concealed_groups):
        full_decomp = _Decomposition(exposed + cd.groups, cd.pair)
        base, bd = _score_decomp(full_decomp, hand, ctx, rules)
        if base > best_base:
            best_base = base
            best_bd = bd

    # Pick the overall best (standard vs seven pairs)
    if seven_pairs_total > best_base:
        best_base = seven_pairs_total
        best_bd = seven_pairs_bd

    # --- Apply base tai cap ---
    capped = best_base > rules.tai_cap
    base_tai = min(best_base, rules.tai_cap)

    # --- Flowers added on top of cap ---
    flower_tai, flower_bd = _score_bonus_tiles(hand)
    total = base_tai + (flower_tai if rules.flowers_above_cap else 0)

    return TaiResult(
        total=total,
        breakdown=best_bd + flower_bd,
        capped=capped,
    )
