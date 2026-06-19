"""Heuristic discard optimizer for Singapore Mahjong."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cracked.tiles import NTILES, WIND_START, WIND_END, DRAGON_START, DRAGON_END, suit_of, Wind
from cracked.game_state import GameState
from cracked.tiles_away import best_discards
from cracked.opponent_model import OpponentModel, model_all_opponents
from cracked.danger import tile_danger_scores
from cracked.scoring import STARTING_CHIPS


@dataclass
class DiscardRecommendation:
    """Full evaluation of one candidate discard."""
    tile_id: int
    tiles_away_after: int
    weighted_acceptance: int    # tiles in unknown pool that improve hand if drawn
    shooting_cost: float        # E[payment in base units] if this tile is discarded
    danger_score: float         # P(at least one opponent wins from this discard)
    tai_potential: float        # estimated tai ceiling of the kept hand structure
    utility: float              # combined score — higher is better
    acceptance: dict[int, int]  # tile_id → remaining count that improve hand


# How much tai-potential weight vs acceptance weight at each tiles_away level.
# Closer to waiting → hand structure is more locked in → tai matters more.
_TAI_WEIGHT = {0: 0.45, 1: 0.30, 2: 0.20}


def hand_tai_potential(
    concealed: np.ndarray,
    melds: list,
    seat_wind: int,
    prevailing_wind: int,
) -> float:
    """
    Estimate the tai ceiling of an incomplete 13-tile hand structure.

    Scoring components (these stack where compatible):
      - Full flush (清一色):  4 tai
      - Half flush (混一色):  2 tai
      - All-pong (对对和):    2 tai  (stacks with flush)
      - Seven pairs (七对):   3 tai  (exclusive path)
      - Concealed honor tiles: partial credit toward dragon/wind pongs
      - Ping hu / chou ping hu penalty: −1 for mixed-suit, no-value sequence hands

    All-pong and flush can stack (e.g. all-pong circles = 4+2 = 6 tai).
    Seven pairs is an exclusive path and is taken when it beats flush+pong.
    """
    suit_counts = [int(concealed[s * 9:(s + 1) * 9].sum()) for s in range(3)]
    honor_count = int(concealed[WIND_START:].sum())
    suited_total = sum(suit_counts)

    if suited_total + honor_count == 0 and not melds:
        return 0.0

    # ── Exposed meld analysis ─────────────────────────────────────────
    meld_tai = 0.0
    all_pong_melds = True
    meld_suited_suits: set[int] = set()
    meld_has_honor = False

    for m in melds:
        t = m.tiles[0]
        if DRAGON_START <= t < DRAGON_END:
            meld_tai += 1.0
            meld_has_honor = True
        elif WIND_START <= t < WIND_END:
            meld_has_honor = True
            if t == prevailing_wind:
                meld_tai += 1.0
            if t == seat_wind:
                meld_tai += 1.0
        else:
            meld_suited_suits.add(suit_of(t))
        if m.type.value not in ("pong", "kong"):
            all_pong_melds = False

    # ── Flush potential ───────────────────────────────────────────────
    # Melds must all be the same suit as the dominant concealed suit.
    dom_suit = max(range(3), key=lambda s: suit_counts[s])
    dom_count = suit_counts[dom_suit]
    meld_suit_ok = not meld_suited_suits or meld_suited_suits == {dom_suit}

    flush_tai = 0.0
    if suited_total > 0 and meld_suit_ok:
        suit_purity = dom_count / suited_total
        has_any_honor = honor_count > 0 or meld_has_honor
        if suit_purity == 1.0:
            # All suited tiles are one colour → full or half flush
            flush_tai = 2.0 if has_any_honor else 4.0
        elif suit_purity >= 0.8:
            # Developing flush — one or two off-suit tiles still in hand
            flush_tai = 1.5 if has_any_honor else 3.0

    # ── All-pong potential ────────────────────────────────────────────
    pairs = sum(1 for i in range(NTILES) if concealed[i] >= 2)
    needed_sets = 4 - len(melds)   # concealed groups still required
    pong_tai = 0.0
    if all_pong_melds and pairs >= max(needed_sets - 1, 1):
        pong_tai = 2.0

    # ── Seven pairs potential ─────────────────────────────────────────
    seven_pairs_tai = 3.0 if (not melds and pairs >= 4) else 0.0

    # ── Concealed honor tile potential ────────────────────────────────
    # Pairs give full credit (likely going for the pong); singles give partial.
    honor_tai = 0.0
    for t in range(DRAGON_START, DRAGON_END):
        cnt = int(concealed[t])
        if cnt >= 2:
            honor_tai += 1.0
        elif cnt == 1:
            honor_tai += 0.3
    for t in range(WIND_START, WIND_END):
        cnt = int(concealed[t])
        val = (1.0 if t == prevailing_wind else 0.0) + (1.0 if t == seat_wind else 0.0)
        if val > 0:
            if cnt >= 2:
                honor_tai += val
            elif cnt == 1:
                honor_tai += val * 0.3

    # ── Ping hu / chou ping hu penalty ───────────────────────────────
    # A mixed-suit, all-sequence hand with no value tiles has a tai ceiling
    # of 0 from structure. Flag it so the optimizer prefers higher-value paths.
    no_structural = flush_tai == 0.0 and pong_tai == 0.0 and seven_pairs_tai == 0.0
    no_honor_value = meld_tai == 0.0 and honor_tai < 0.5
    ping_hu_penalty = -1.0 if (no_structural and no_honor_value) else 0.0

    # Flush and all-pong stack; seven pairs is an exclusive alternative path.
    structural = max(flush_tai + pong_tai, seven_pairs_tai)
    return max(0.0, meld_tai + structural + honor_tai + ping_hu_penalty)


def adaptive_alpha(
    state: GameState,
    models: list[OpponentModel],
    best_tiles_away: int,
    chip_lead: float = 0.0,
) -> float:
    """
    Compute α: weight on offensive vs defensive play (0=pure defence, 1=pure offence).

    Drivers that lower α (push toward safety):
      - Opponents with high waiting probability
      - A high-value (big-tai) waiting opponent — exponential, shooter-pays-all cost
      - Wall running thin
      - A large chip lead worth protecting
    Drivers that raise α (push toward offence):
      - Hand is close to waiting
      - Being behind on chips (need variance)
      - Sitting East (a win keeps the dealership)

    chip_lead: (my chips − average opponent chips) / starting stack, clamped to
    [-0.5, +0.5]. Defaults to 0 (no placement signal) for callers without chips.
    """
    hand_factor = {-1: 0.85, 0: 0.75, 1: 0.60}.get(best_tiles_away, 0.45)
    max_waiting = max((m.waiting_prob for m in models), default=0.0)
    opp_factor = 1.0 - max_waiting * 0.5

    # Threat-weighted defense: the most dangerous opponent's exponential payment
    # (2^(tai-1)) collapses α — one big-tai deal-in dwarfs many small wins.
    max_threat = max(
        (m.waiting_prob * (2.0 ** (m.est_tai - 1.0)) for m in models), default=0.0
    )
    threat_factor = 1.0 / (1.0 + 0.15 * max_threat)

    wall_factor = min(state.wall_tiles_remaining / 80.0, 1.0)

    # Placement: protect a lead (lower α), gamble when behind (raise α).
    # A max lead halves the offensive weight; a max deficit raises it by 50%.
    placement_factor = 1.0 - max(-0.5, min(0.5, chip_lead)) * 1.0
    # East keeps the deal on a win, so values winning slightly more.
    dealer_factor = 1.05 if state.my_seat == int(Wind.EAST) else 1.0

    alpha = (hand_factor * opp_factor * threat_factor * wall_factor
             * placement_factor * dealer_factor)
    return max(0.15, min(0.90, alpha))


def _chip_lead(my_chips: int | None, opponent_chips: list[int] | None) -> float:
    """Normalised chip lead vs the average opponent, or 0.0 if chips unknown."""
    if my_chips is None or not opponent_chips:
        return 0.0
    avg_opp = sum(opponent_chips) / len(opponent_chips)
    return (my_chips - avg_opp) / STARTING_CHIPS


def recommend_discard(
    state: GameState,
    my_chips: int | None = None,
    opponent_chips: list[int] | None = None,
) -> list[DiscardRecommendation]:
    """
    Rank every possible discard from a 14-tile hand by utility.

    Utility = α × offensive_score + (1−α) × defensive_score
      offensive_score: blend of normalized acceptance and tai potential
                       (tai weight scales with tiles_away — heavier near waiting)
      defensive_score: 1 − normalized shooting_cost

    Primary sort key is tiles_away_after (lower is always better).
    Utility resolves ties within the same tiles_away level.

    my_chips / opponent_chips (optional): current chip standings. When provided,
    α becomes placement-aware — protect a lead, gamble when behind.

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
    tiles_away_results = best_discards(hand.concealed, unknown, n_melds)
    if not tiles_away_results:
        return []

    models = model_all_opponents(state)
    danger = tile_danger_scores(state, models)

    best_tiles_away_val = tiles_away_results[0]["tiles_away_after"]
    chip_lead = _chip_lead(my_chips, opponent_chips)
    alpha = adaptive_alpha(state, models, best_tiles_away_val, chip_lead)

    # Tai weight: as tiles_away drops the hand structure becomes more locked in,
    # so tai potential deserves more influence over acceptance count.
    tai_w = _TAI_WEIGHT.get(best_tiles_away_val, 0.15)
    acc_w = 1.0 - tai_w

    # Compute tai potential for each candidate (post-discard 13-tile hand).
    tai_pots = []
    for r in tiles_away_results:
        mod = hand.concealed.copy()
        mod[r["tile_id"]] -= 1
        tai_pots.append(
            hand_tai_potential(mod, hand.melds, hand.seat_wind, state.prevailing_wind)
        )

    acceptances = [r["weighted_acceptance"] for r in tiles_away_results]
    costs = [danger[r["tile_id"]].expected_cost for r in tiles_away_results]
    max_acc  = max(acceptances) if acceptances else 1
    max_cost = max(costs) if costs else 0.001
    max_tai  = max(tai_pots) if tai_pots else 1.0

    recommendations: list[DiscardRecommendation] = []
    for r, tai_pot in zip(tiles_away_results, tai_pots):
        tid = r["tile_id"]
        d = danger[tid]
        norm_acc    = r["weighted_acceptance"] / max(max_acc, 1)
        norm_safety = 1.0 - (d.expected_cost / max(max_cost, 0.001))
        norm_tai    = tai_pot / max(max_tai, 0.1)

        offensive = acc_w * norm_acc + tai_w * norm_tai
        utility = alpha * offensive + (1.0 - alpha) * norm_safety

        recommendations.append(DiscardRecommendation(
            tile_id=tid,
            tiles_away_after=r["tiles_away_after"],
            weighted_acceptance=r["weighted_acceptance"],
            shooting_cost=round(d.expected_cost, 3),
            danger_score=round(d.score, 3),
            tai_potential=round(tai_pot, 2),
            utility=round(utility, 4),
            acceptance=r["acceptance"],
        ))

    recommendations.sort(key=lambda x: (x.tiles_away_after, -x.utility))
    return recommendations
