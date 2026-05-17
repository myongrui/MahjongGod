"""Feature extraction: GameState → fixed-size numpy vectors."""

from __future__ import annotations

import numpy as np

from cracked.tiles import NTILES, Wind, WIND_START, WIND_END, DRAGON_START, DRAGON_END
from cracked.game_state import GameState
from cracked.opponent_model import model_all_opponents
from cracked.shanten import shanten
from cracked.optimizer import hand_tai_potential, adaptive_alpha

_WIND_ORDER = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]

# -----------------------------------------------------------------------
# Layout overview
# -----------------------------------------------------------------------
# State block (89 dims):
#   [0:34]    concealed tile counts / 4
#   [34:68]   unknown tile counts / 4
#   [68]      n_melds / 4
#   [69]      n_flowers / 8
#   [70]      n_animals / 4
#   [71]      current shanten, (s+1)/9  (-1→0, 7→0.89)
#   [72:76]   my seat wind one-hot
#   [76:80]   prevailing wind one-hot
#   [80]      turn / 40, clamped
#   [81]      wall_remaining / 136
#   [82]      tai_potential / 10         (optimizer hand-value estimate)
#   [83]      flush_purity               (dominant-suit ratio of suited tiles)
#   [84]      pair_count / 7
#   [85]      all_pong_signal            (1.0 if hand trends toward all-pong)
#   [86]      seven_pairs_signal         (1.0 if 4+ pairs and no melds)
#   [87]      honor_tile_count / 14
#   [88]      adaptive_alpha             (offense/defense balance, 0.15–0.90)
#
# Opponent block (47 dims per opponent, 141 total):
#   [+0:4]    seat wind one-hot
#   [+4]      n_melds / 4
#   [+5:39]   discard tile counts / 4
#   [+39]     tenpai_prob
#   [+40:44]  suit_bias one-hot [bamboo, chars, circles, none]
#   [+44]     honor_bias
#   [+45]     dragon_danger_count / 3
#   [+46]     wind_danger_count / 4
#
# N_STATE_FEATURES = 89 + 141 = 230  (used by policy/value nets)
#
# Candidate-discard block (35 dims) appended by extract_features:
#   [+0:34]   candidate tile one-hot
#   [+34]     shanten after discard, (s+1)/9
#
# N_FEATURES = 230 + 35 = 265  (used by supervised DangerNet)

_STATE_BLOCK_SIZE = 89
_OPP_BLOCK_SIZE = 47

N_STATE_FEATURES = _STATE_BLOCK_SIZE + 3 * _OPP_BLOCK_SIZE   # 230
N_FEATURES = N_STATE_FEATURES + 35                            # 265


def _fill_state_block(feat: np.ndarray, state: GameState, models=None) -> None:
    """Write the 89-dim state block into feat[0:89]."""
    hand = state.my_hand
    n_melds = len(hand.melds)
    concealed = hand.concealed

    feat[0:34] = concealed / 4.0
    feat[34:68] = state.unknown_tiles() / 4.0
    feat[68] = n_melds / 4.0
    feat[69] = len(hand.flowers) / 8.0
    feat[70] = len(hand.animals) / 4.0
    s = shanten(concealed, n_melds)
    feat[71] = (s + 1) / 9.0
    for i, w in enumerate(_WIND_ORDER):
        feat[72 + i] = 1.0 if state.my_seat == w else 0.0
    for i, w in enumerate(_WIND_ORDER):
        feat[76 + i] = 1.0 if state.prevailing_wind == w else 0.0
    feat[80] = min(state.turn_number / 40.0, 1.0)
    feat[81] = state.wall_tiles_remaining / 136.0

    # -- Optimizer-derived hand structure features --
    tai_pot = hand_tai_potential(
        concealed, hand.melds, hand.seat_wind, state.prevailing_wind
    )
    feat[82] = tai_pot / 10.0

    suit_counts = [int(concealed[ss * 9:(ss + 1) * 9].sum()) for ss in range(3)]
    suited_total = sum(suit_counts)
    feat[83] = max(suit_counts) / max(suited_total, 1) if suited_total > 0 else 0.0

    pairs = sum(1 for i in range(NTILES) if concealed[i] >= 2)
    feat[84] = pairs / 7.0

    needed = 4 - n_melds
    all_pong_melds = (
        all(m.type.value in ("pong", "kong") for m in hand.melds)
        if hand.melds else True
    )
    feat[85] = 1.0 if (all_pong_melds and pairs >= max(needed - 1, 1)) else 0.0
    feat[86] = 1.0 if (not hand.melds and pairs >= 4) else 0.0
    feat[87] = int(concealed[WIND_START:].sum()) / 14.0

    if models is not None:
        feat[88] = adaptive_alpha(state, models, s)


def _fill_opponent_blocks(
    feat: np.ndarray,
    base: int,
    state: GameState,
    models=None,
) -> None:
    """Write the 141-dim (3 × 47) opponent block."""
    if models is None:
        models = model_all_opponents(state)
    for opp_idx, (opp, model) in enumerate(zip(state.opponents, models)):
        b = base + opp_idx * _OPP_BLOCK_SIZE
        for i, w in enumerate(_WIND_ORDER):
            feat[b + i] = 1.0 if opp.seat == w else 0.0
        feat[b + 4] = len(opp.melds) / 4.0
        discard_arr = np.zeros(NTILES, dtype=np.float32)
        for t in opp.discards:
            discard_arr[t] += 1
        feat[b + 5: b + 39] = discard_arr / 4.0
        feat[b + 39] = model.tenpai_prob
        suit_idx = model.suit_bias if model.suit_bias is not None else 3
        feat[b + 40 + suit_idx] = 1.0
        feat[b + 44] = 1.0 if model.honor_bias else 0.0
        feat[b + 45] = (
            sum(1 for t in model.dangerous_tiles if DRAGON_START <= t < DRAGON_END)
            / 3.0
        )
        feat[b + 46] = (
            sum(1 for t in model.dangerous_tiles if WIND_START <= t < WIND_END)
            / 4.0
        )


def extract_state_features(state: GameState) -> np.ndarray:
    """
    Encode the observable game state into a float32 vector of length
    N_STATE_FEATURES (230).

    Used by the policy/value nets which evaluate the whole state at once
    rather than one candidate discard at a time.
    """
    feat = np.zeros(N_STATE_FEATURES, dtype=np.float32)
    models = model_all_opponents(state)
    _fill_state_block(feat, state, models)
    _fill_opponent_blocks(feat, base=_STATE_BLOCK_SIZE, state=state, models=models)
    return feat


def extract_features(state: GameState, candidate_discard: int) -> np.ndarray:
    """
    Encode (state, candidate_discard) into a float32 vector of length
    N_FEATURES (265).

    The first 89 dims are the state block (same as extract_state_features).
    The next 35 dims encode the candidate tile and shanten-after.
    The final 141 dims encode the three opponents.
    """
    feat = np.zeros(N_FEATURES, dtype=np.float32)
    models = model_all_opponents(state)
    _fill_state_block(feat, state, models)

    # Candidate discard block at [89:124]
    cand_base = _STATE_BLOCK_SIZE
    feat[cand_base + candidate_discard] = 1.0
    trial = state.my_hand.concealed.copy()
    trial[candidate_discard] -= 1
    s_after = shanten(trial, len(state.my_hand.melds))
    feat[cand_base + 34] = (s_after + 1) / 9.0

    # Opponent blocks at [124:265]
    opp_base = cand_base + 35
    _fill_opponent_blocks(feat, base=opp_base, state=state, models=models)
    return feat
