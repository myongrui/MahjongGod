"""Feature extraction: GameState → fixed-size numpy vectors."""

from __future__ import annotations

import numpy as np

from cracked.tiles import NTILES, Wind
from cracked.game_state import GameState
from cracked.opponent_model import model_all_opponents
from cracked.shanten import shanten

_WIND_ORDER = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]

# -----------------------------------------------------------------------
# Layout overview
# -----------------------------------------------------------------------
# State block (82 dims) — encodes game context without a candidate discard:
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
#
# Opponent block (45 dims per opponent, 135 total) follows the state block:
#   [+0:4]    seat wind one-hot
#   [+4]      n_melds / 4
#   [+5:39]   discard tile counts / 4
#   [+39]     tenpai_prob
#   [+40:44]  suit_bias one-hot [bamboo, chars, circles, none]
#   [+44]     honor_bias
#
# N_STATE_FEATURES = 82 + 135 = 217  (used by policy/value nets)
#
# Candidate-discard block (35 dims) appended by extract_features:
#   [+0:34]   candidate tile one-hot
#   [+34]     shanten after discard, (s+1)/9
#
# N_FEATURES = 217 + 35 = 252  (used by supervised DangerNet)

N_STATE_FEATURES = 217
N_FEATURES = 252


def _fill_state_block(feat: np.ndarray, state: GameState) -> None:
    """Write the 82-dim state block into feat[0:82]."""
    hand = state.my_hand
    n_melds = len(hand.melds)
    feat[0:34] = hand.concealed / 4.0
    feat[34:68] = state.unknown_tiles() / 4.0
    feat[68] = n_melds / 4.0
    feat[69] = len(hand.flowers) / 8.0
    feat[70] = len(hand.animals) / 4.0
    s = shanten(hand.concealed, n_melds)
    feat[71] = (s + 1) / 9.0
    for i, w in enumerate(_WIND_ORDER):
        feat[72 + i] = 1.0 if state.my_seat == w else 0.0
    for i, w in enumerate(_WIND_ORDER):
        feat[76 + i] = 1.0 if state.prevailing_wind == w else 0.0
    feat[80] = min(state.turn_number / 40.0, 1.0)
    feat[81] = state.wall_tiles_remaining / 136.0


def _fill_opponent_blocks(feat: np.ndarray, base: int, state: GameState) -> None:
    """Write the 135-dim (3 × 45) opponent block into feat[base:base+135]."""
    models = model_all_opponents(state)
    for opp_idx, (opp, model) in enumerate(zip(state.opponents, models)):
        b = base + opp_idx * 45
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


def extract_state_features(state: GameState) -> np.ndarray:
    """
    Encode the observable game state into a float32 vector of length N_STATE_FEATURES (217).

    Used by the policy/value nets which evaluate the whole state at once
    rather than one candidate discard at a time.
    """
    feat = np.zeros(N_STATE_FEATURES, dtype=np.float32)
    _fill_state_block(feat, state)
    _fill_opponent_blocks(feat, base=82, state=state)
    return feat


def extract_features(state: GameState, candidate_discard: int) -> np.ndarray:
    """
    Encode (state, candidate_discard) into a float32 vector of length N_FEATURES (252).

    The first 217 dims are identical to extract_state_features(state).
    The last 35 dims encode the candidate tile and shanten-after.
    """
    feat = np.zeros(N_FEATURES, dtype=np.float32)
    _fill_state_block(feat, state)

    # Candidate discard block at [82:117]
    feat[82 + candidate_discard] = 1.0
    trial = state.my_hand.concealed.copy()
    trial[candidate_discard] -= 1
    s_after = shanten(trial, len(state.my_hand.melds))
    feat[116] = (s_after + 1) / 9.0

    # Opponent blocks at [117:252]
    _fill_opponent_blocks(feat, base=117, state=state)
    return feat
