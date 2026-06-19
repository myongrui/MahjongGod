"""
Tiles-away calculator for Singapore Mahjong.

Tiles-away number = minimum tile swaps to reach a complete hand.
  -1 = hand is already complete (waiting + 1)
   0 = waiting (one tile away from winning)
   n = n tiles away from waiting

Three winning forms are checked and the minimum tiles_away taken:
  1. Standard form: 4 groups (sequences/triplets) + 1 pair
  2. Seven pairs: 7 distinct pairs
  3. Thirteen orphans: one of each terminal/honor + one duplicate
"""

from __future__ import annotations

import numpy as np

from cracked.tiles import (
    NTILES, BAMBOO_START, BAMBOO_END, CHAR_START, CHAR_END,
    CIRCLE_START, CIRCLE_END, WIND_START, DRAGON_END,
    new_hand_array,
)

# Terminals and honors used in Thirteen Orphans
_ORPHAN_TILES = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]  # 13 tiles


# ---------------------------------------------------------------------------
# Standard form tiles_away
# ---------------------------------------------------------------------------

def _suit_tiles_away(tiles: np.ndarray) -> tuple[int, int]:
    """
    Recursively find the best (sets, partial) count for one suit's tiles.
    sets = complete groups (sequences or triplets)
    partial = partial groups (pairs or two-sided waits)

    Returns (max_sets, max_partial_given_max_sets).
    Uses recursive backtracking with memoization via a tuple key.
    """
    key = tuple(tiles)
    if key in _suit_cache:
        return _suit_cache[key]

    best_m, best_p = _suit_recurse(tiles.copy(), 0, 0)
    _suit_cache[key] = (best_m, best_p)
    return best_m, best_p

_suit_cache: dict[tuple, tuple[int, int]] = {}


def _suit_recurse(tiles: np.ndarray, sets: int, partial: int) -> tuple[int, int]:
    best = (sets, partial)

    # Find the first tile present
    for i in range(len(tiles)):
        if tiles[i] == 0:
            continue

        # Try to form a triplet
        if tiles[i] >= 3:
            tiles[i] -= 3
            r = _suit_recurse(tiles, sets + 1, partial)
            tiles[i] += 3
            if r > best:
                best = r

        # Try to form a sequence (only for suited tiles with two following tiles)
        if i + 2 < len(tiles) and tiles[i + 1] > 0 and tiles[i + 2] > 0:
            tiles[i] -= 1
            tiles[i + 1] -= 1
            tiles[i + 2] -= 1
            r = _suit_recurse(tiles, sets + 1, partial)
            tiles[i] += 1
            tiles[i + 1] += 1
            tiles[i + 2] += 1
            if r > best:
                best = r

        # Try to form a pair (partial)
        if tiles[i] >= 2:
            tiles[i] -= 2
            r = _suit_recurse(tiles, sets, partial + 1)
            tiles[i] += 2
            if r > best:
                best = r

        # Try to form a closed wait (partial: i, i+2)
        if i + 2 < len(tiles) and tiles[i + 2] > 0:
            tiles[i] -= 1
            tiles[i + 2] -= 1
            r = _suit_recurse(tiles, sets, partial + 1)
            tiles[i] += 1
            tiles[i + 2] += 1
            if r > best:
                best = r

        # Try to form a sequential partial (i, i+1)
        if i + 1 < len(tiles) and tiles[i + 1] > 0:
            tiles[i] -= 1
            tiles[i + 1] -= 1
            r = _suit_recurse(tiles, sets, partial + 1)
            tiles[i] += 1
            tiles[i + 1] += 1
            if r > best:
                best = r

        # Once we've processed tile i, don't revisit it (move forward)
        break

    return best


def tiles_away_standard(hand: np.ndarray, exposed_melds: int = 0) -> int:
    """
    Tiles-away for standard form (4 groups + 1 pair), accounting for exposed melds.
    exposed_melds: number of already-completed exposed sets (pong/chow/kong).
    """
    groups_needed = 4 - exposed_melds

    best = 8 - 2 * exposed_melds  # worst case

    # Split hand into suits and honors
    bamboo = hand[BAMBOO_START:BAMBOO_END].copy()
    chars  = hand[CHAR_START:CHAR_END].copy()
    circles = hand[CIRCLE_START:CIRCLE_END].copy()
    honors = hand[WIND_START:DRAGON_END].copy()

    # Honors can only form triplets (no sequences), so handle separately.
    # A lone honor (count=1) contributes nothing — it needs 2 more copies to form
    # a pong, so it is not a 1-draw partial. Only pairs (count=2) are partials.
    honor_sets = 0
    honor_partial = 0
    for i in range(len(honors)):
        if honors[i] >= 3:
            honor_sets += 1
        elif honors[i] == 2:
            honor_partial += 1

    # For each suit, compute best (sets, partial)
    suit_results = []
    for suit_tiles in (bamboo, chars, circles):
        m, p = _suit_tiles_away(suit_tiles)
        suit_results.append((m, p))

    # Now try assigning the "pair" to each possible source and compute tiles_away
    # Try pair from each suit
    for pair_suit_idx, suit_tiles in enumerate((bamboo, chars, circles)):
        for pair_tid in range(len(suit_tiles)):
            if suit_tiles[pair_tid] < 2:
                continue
            # Use this tile as the pair
            suit_tiles[pair_tid] -= 2
            pm, pp = _suit_tiles_away(suit_tiles)
            suit_tiles[pair_tid] += 2

            total_m = pm + honor_sets
            total_p = pp + honor_partial
            for i, (m, p) in enumerate(suit_results):
                if i != pair_suit_idx:
                    total_m += m
                    total_p += p

            s = _calc_tiles_away(total_m, total_p, groups_needed, has_pair=True)
            best = min(best, s)

    # Try pair from honors
    for i in range(len(honors)):
        if honors[i] < 2:
            continue
        honors[i] -= 2
        hm = 0
        hp = 0
        for j in range(len(honors)):
            if honors[j] >= 3:
                hm += 1
            elif honors[j] >= 2:
                hp += 1
        honors[i] += 2

        total_m = hm
        total_p = hp
        for m, p in suit_results:
            total_m += m
            total_p += p

        s = _calc_tiles_away(total_m, total_p, groups_needed, has_pair=True)
        best = min(best, s)

    # Also try without a designated pair (might be better when far from waiting)
    total_m = honor_sets
    total_p = honor_partial
    for m, p in suit_results:
        total_m += m
        total_p += p
    s = _calc_tiles_away(total_m, total_p, groups_needed, has_pair=False)
    best = min(best, s)

    return best


def _calc_tiles_away(sets: int, partial: int, groups_needed: int, has_pair: bool) -> int:
    """Compute tiles_away from group counts."""
    # Cap sets at groups_needed
    sets = min(sets, groups_needed)
    # Total blocks (sets + partial) cannot exceed groups_needed + (1 if has_pair else 0)
    # because extra partials don't help
    max_blocks = groups_needed + (1 if has_pair else 0)
    partial = min(partial, max_blocks - sets)

    return (groups_needed - sets) * 2 - partial - (1 if has_pair else 0)


# ---------------------------------------------------------------------------
# Seven Pairs tiles_away
# ---------------------------------------------------------------------------

def tiles_away_seven_pairs(hand: np.ndarray) -> int:
    """Tiles-away for seven pairs hand. Requires 7 pairs of distinct tiles."""
    pairs = int(np.sum(hand >= 2))
    # Need 7 pairs; each pair we already have reduces tiles_away by 1
    # Start at 6 (need 6 more pairs after the first)
    return 6 - pairs


# ---------------------------------------------------------------------------
# Thirteen Orphans tiles_away
# ---------------------------------------------------------------------------

def tiles_away_thirteen_orphans(hand: np.ndarray) -> int:
    """Tiles-away for thirteen orphans (十三幺)."""
    unique_orphans = sum(1 for t in _ORPHAN_TILES if hand[t] > 0)
    has_pair = any(hand[t] >= 2 for t in _ORPHAN_TILES)
    return 13 - unique_orphans - (1 if has_pair else 0)


# ---------------------------------------------------------------------------
# Combined tiles_away + acceptance count
# ---------------------------------------------------------------------------

def tiles_away(hand: np.ndarray, exposed_melds: int = 0) -> int:
    """
    Minimum tiles_away across all three winning forms.
    hand: 34-element array of tile counts (concealed tiles only).
    exposed_melds: number of already-exposed complete sets.
    """
    s = tiles_away_standard(hand, exposed_melds)
    # Seven pairs and thirteen orphans only apply with no exposed melds
    if exposed_melds == 0:
        s = min(s, tiles_away_seven_pairs(hand))
        s = min(s, tiles_away_thirteen_orphans(hand))
    return s


def acceptance_count(hand: np.ndarray, unknown_tiles: np.ndarray, exposed_melds: int = 0) -> dict[int, int]:
    """
    Compute the acceptance set (uke-ire): tiles that would reduce tiles_away if drawn.

    hand: 34-element concealed hand array (13 tiles — after a discard).
    unknown_tiles: 34-element array of tiles not yet visible (in wall or opponent hands).
    exposed_melds: number of exposed complete sets.

    Returns dict mapping tile_id -> count remaining in unknown tiles.
    """
    current = tiles_away(hand, exposed_melds)
    if current == -1:
        return {}
    result = {}
    for tid in range(NTILES):
        if unknown_tiles[tid] <= 0:
            continue
        hand[tid] += 1
        if tiles_away(hand, exposed_melds) < current:
            result[tid] = int(unknown_tiles[tid])
        hand[tid] -= 1
    return result


def best_discards(hand14: np.ndarray, unknown_tiles: np.ndarray, exposed_melds: int = 0) -> list[dict]:
    """
    For a 14-tile hand, evaluate every possible discard and return results sorted
    by (tiles_away_after ASC, acceptance_count DESC).

    Returns a list of dicts with keys:
      tile_id, tiles_away_after, acceptance, weighted_acceptance
    """
    results = []
    for tid in range(NTILES):
        if hand14[tid] == 0:
            continue
        hand14[tid] -= 1
        s = tiles_away(hand14, exposed_melds)
        acc = acceptance_count(hand14, unknown_tiles, exposed_melds) if s >= 0 else {}
        weighted = sum(acc.values())
        results.append({
            "tile_id": tid,
            "tiles_away_after": s,
            "acceptance": acc,
            "weighted_acceptance": weighted,
        })
        hand14[tid] += 1

    results.sort(key=lambda x: (x["tiles_away_after"], -x["weighted_acceptance"]))
    return results
