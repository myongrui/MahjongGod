"""Heuristic discard optimizer for Singapore Mahjong."""

from __future__ import annotations

from dataclasses import dataclass

from cracked.game_state import GameState
from cracked.shanten import best_discards
from cracked.opponent_model import OpponentModel, model_all_opponents
from cracked.danger import tile_danger_scores


@dataclass
class DiscardRecommendation:
    """Full evaluation of one candidate discard."""
    tile_id: int
    shanten_after: int
    weighted_acceptance: int    # tiles in unknown pool that improve hand if drawn
    shooting_cost: float        # E[payment in base units] if this tile is discarded
    danger_score: float         # P(at least one opponent wins from this discard)
    utility: float              # combined score — higher is better
    acceptance: dict[int, int]  # tile_id → remaining count that improve hand


def adaptive_alpha(
    state: GameState,
    models: list[OpponentModel],
    best_shanten: int,
) -> float:
    """
    Compute α: weight on offensive vs defensive play (0=pure defence, 1=pure offence).

    Drivers that lower α (push toward safety):
      - Opponents with high tenpai probability
      - Wall running thin
    Drivers that raise α (push toward offence):
      - Hand is close to tenpai1
    """
    hand_factor = {-1: 0.85, 0: 0.75, 1: 0.60}.get(best_shanten, 0.45)
    max_tenpai = max((m.tenpai_prob for m in models), default=0.0)
    opp_factor = 1.0 - max_tenpai * 0.5
    wall_factor = min(state.wall_tiles_remaining / 80.0, 1.0)
    alpha = hand_factor * opp_factor * wall_factor
    return max(0.15, min(0.90, alpha))


def recommend_discard(state: GameState) -> list[DiscardRecommendation]:
    """
    Rank every possible discard from a 14-tile hand by utility.

    Utility = α × offensive_score + (1−α) × defensive_score
      offensive_score: normalized weighted_acceptance (0–1)
      defensive_score: 1 − normalized shooting_cost (0–1, higher = safer)

    Primary sort key is shanten_after (lower is always better).
    Utility resolves ties within the same shanten level.

    Raises ValueError if the hand does not have exactly 14 concealed tiles
    (adjusted for exposed melds).
    """
    hand = state.my_hand
    n_melds = len(hand.melds)
    expected = 14 - 3 * n_melds
    if hand.total_concealed != expected:
        raise ValueError(
            f"Expected {expected} concealed tiles, got {hand.total_concealed}. "
            "Run 'cracked draw TILE' first."
        )

    unknown = state.unknown_tiles()
    shanten_results = best_discards(hand.concealed, unknown, n_melds)
    if not shanten_results:
        return []

    models = model_all_opponents(state)
    danger = tile_danger_scores(state, models)

    best_shanten_val = shanten_results[0]["shanten_after"]
    alpha = adaptive_alpha(state, models, best_shanten_val)

    acceptances = [r["weighted_acceptance"] for r in shanten_results]
    costs = [danger[r["tile_id"]].expected_cost for r in shanten_results]
    max_acc = max(acceptances) if acceptances else 1
    max_cost = max(costs) if costs else 0.001

    recommendations: list[DiscardRecommendation] = []
    for r in shanten_results:
        tid = r["tile_id"]
        d = danger[tid]
        norm_acc = r["weighted_acceptance"] / max(max_acc, 1)
        norm_safety = 1.0 - (d.expected_cost / max(max_cost, 0.001))
        utility = alpha * norm_acc + (1.0 - alpha) * norm_safety

        recommendations.append(DiscardRecommendation(
            tile_id=tid,
            shanten_after=r["shanten_after"],
            weighted_acceptance=r["weighted_acceptance"],
            shooting_cost=round(d.expected_cost, 3),
            danger_score=round(d.score, 3),
            utility=round(utility, 4),
            acceptance=r["acceptance"],
        ))

    recommendations.sort(key=lambda x: (x.shanten_after, -x.utility))
    return recommendations
