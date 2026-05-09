"""Tile danger scoring for Singapore Mahjong discard assessment."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cracked.tiles import NTILES
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
    Probability that tile tid completes this opponent's hand, given they are tenpai.

    Calibration:
      safe_tiles  → 0.00  (discarded by them, or exhausted)
      dangerous_tiles → 0.55  (in their flush suit or adjacent to their melds)
      unknown         → 0.15  (baseline risk for tiles we know nothing about)
    """
    if tid in model.safe_tiles:
        return 0.0
    if tid in model.dangerous_tiles:
        return 0.55
    return 0.15


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

        # P(at least one opponent wins) = 1 − P(none win)
        p_none_win = 1.0
        total_cost = 0.0
        for model in models:
            p_d = _p_dangerous(tid, model)
            p_win = model.tenpai_prob * p_d
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
