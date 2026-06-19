"""Tile danger scoring for Singapore Mahjong discard assessment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cracked.tiles import NTILES, is_suited, suit_of
from cracked.game_state import GameState
from cracked.opponent_model import OpponentModel


@dataclass
class TileDanger:
    """Danger assessment for one tile type."""
    tile_id: int
    score: float          # aggregate danger 0.0 (safe) → 1.0 (very dangerous)
    exhausted: bool       # all 4 copies visible — definitely safe
    expected_cost: float  # E[payment in base units] if this tile is discarded


def _base_payment(est_tai: float) -> float:
    """Payment per loser at a given tai estimate (doubles each tai)."""
    return math.pow(2.0, max(est_tai - 1.0, 0.0))


def _p_dangerous(tid: int, model: OpponentModel) -> float:
    """
    Probability that tile tid completes this opponent's hand, given they are waiting.

    Calibration (Singapore has no sacred-discard rule, so a tile they threw is
    strong but NOT absolute evidence of safety):
      safe_tiles       → 0.00  (all 4 copies seen, or 4th copy of a ponged honor)
      dangerous + thrown → 0.40  (in their suit/adjacent but they once discarded it)
      dangerous_tiles  → 0.55  (in their flush suit or adjacent to their melds)
      discarded_tiles  → 0.05  (they threw it — likely safe, not guaranteed)
      unknown          → 0.15  (baseline risk for tiles we know nothing about)
    """
    if tid in model.safe_tiles:
        return 0.0
    discarded = tid in model.discarded_tiles
    if tid in model.dangerous_tiles:
        return 0.40 if discarded else 0.55
    if discarded:
        return 0.05
    return 0.15


def wall_wait_factor(tid: int, visible: np.ndarray) -> float:
    """
    Wall-read / one-chance danger multiplier in [0.3, 1.0] for suited tiles.

    A suited tile is completed by sequence waits built from adjacent tiles:
    a closed wait {t-1, t+1}, or a two-sided/edge wait holding {t+1, t+2} or
    {t-2, t-1}. If every copy of a tile a shape needs is already visible, no
    opponent can hold that shape, so that wait is impossible ("no-chance").
    The more shapes ruled out by the wall, the safer the tile.

    Honors have no sequence waits, so they are unaffected (1.0). The 0.3 floor
    reflects pair/single waits, which adjacent walls cannot rule out. This is
    pure tile-counting, so it holds regardless of any win rule.
    """
    if not is_suited(tid):
        return 1.0
    suit = suit_of(tid)
    rank = tid % 9  # 0..8 within suit
    shapes = ((-1, 1), (1, 2), (-2, -1))  # closed, two-sided low-held, two-sided high-held
    total = 0
    possible = 0
    for a, b in shapes:
        ra, rb = rank + a, rank + b
        if not (0 <= ra <= 8 and 0 <= rb <= 8):
            continue
        total += 1
        ta, tb = suit * 9 + ra, suit * 9 + rb
        if int(visible[ta]) < 4 and int(visible[tb]) < 4:
            possible += 1
    if total == 0:
        return 1.0
    return 0.3 + 0.7 * (possible / total)


def tile_danger_scores(
    state: GameState,
    models: list[OpponentModel],
) -> dict[int, TileDanger]:
    """
    Compute danger scores for all 34 tile types.

    score: P(at least one opponent wins from your discard)
    expected_cost: E[total payment in base units] (shooter-pays-all, 3 losers)
    """
    visible = state.visible_tiles()
    result: dict[int, TileDanger] = {}

    for tid in range(NTILES):
        exhausted = int(visible[tid]) >= 4

        if exhausted or not models:
            result[tid] = TileDanger(
                tile_id=tid, score=0.0, exhausted=exhausted, expected_cost=0.0
            )
            continue

        # Wall read: shapes that complete this tile may be impossible if the
        # tiles needed to form them are all visible. Down-weights danger globally.
        wall_factor = wall_wait_factor(tid, visible)

        # P(at least one opponent wins) = 1 − P(none win)
        p_none_win = 1.0
        total_cost = 0.0
        for model in models:
            p_d = _p_dangerous(tid, model) * wall_factor
            p_win = model.waiting_prob * p_d
            p_none_win *= (1.0 - p_win)
            # Shooter-pays-all: pay all three losers simultaneously
            total_cost += p_win * _base_payment(model.est_tai) * 3.0

        result[tid] = TileDanger(
            tile_id=tid,
            score=round(1.0 - p_none_win, 4),
            exhausted=False,
            expected_cost=round(total_cost, 4),
        )

    return result


def expected_shooting_cost(
    tid: int,
    state: GameState,
    models: list[OpponentModel],
) -> float:
    """Expected payment cost (in base units) for discarding one specific tile."""
    return tile_danger_scores(state, models)[tid].expected_cost
