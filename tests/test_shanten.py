"""
Shanten tests using known hand configurations.
All hands are 13 tiles (concealed) unless noted.
"""

import pytest
import numpy as np
from cracked.tiles import tiles_from_names, full_wall, tile_id, new_hand_array
from cracked.shanten import (
    shanten, shanten_standard, shanten_seven_pairs,
    shanten_thirteen_orphans, acceptance_count, best_discards,
)


def hand(*names: str) -> np.ndarray:
    return tiles_from_names(list(names))


# ---------------------------------------------------------------------------
# Complete hands (shanten = -1)
# ---------------------------------------------------------------------------

def test_complete_hand_all_sequences():
    h = hand("b1","b2","b3", "c4","c5","c6", "d7","d8","d9", "ew","ew","ew", "rd","rd")
    assert shanten(h) == -1

def test_complete_hand_all_pongs():
    h = hand("b1","b1","b1", "c2","c2","c2", "d3","d3","d3", "ew","ew","ew", "rd","rd")
    assert shanten(h) == -1

def test_complete_seven_pairs():
    h = hand("b1","b1", "b2","b2", "b3","b3", "c1","c1", "c2","c2", "c3","c3", "d1","d1")
    assert shanten(h) == -1
    assert shanten_seven_pairs(h) == -1

def test_complete_thirteen_orphans():
    h = hand("b1","b9", "c1","c9", "d1","d9", "ew","sw","ww","nw", "rd","gd","wd","b1")
    assert shanten(h) == -1
    assert shanten_thirteen_orphans(h) == -1


# ---------------------------------------------------------------------------
# Tenpai hands (shanten = 0)
# ---------------------------------------------------------------------------

def test_tenpai_single_wait():
    # Waiting on b3 to complete b1-b2-b3
    h = hand("b1","b2", "c1","c2","c3", "d1","d2","d3", "ew","ew","ew", "rd","rd")
    assert shanten(h) == 0

def test_tenpai_seven_pairs():
    # 6 pairs + 1 single = tenpai for 7th pair
    h = hand("b1","b1", "b2","b2", "b3","b3", "c1","c1", "c2","c2", "c3","c3", "d1")
    assert shanten_seven_pairs(h) == 0
    assert shanten(h) == 0

def test_tenpai_thirteen_orphans_no_pair():
    # All 13 orphan types, no pair
    h = hand("b1","b9","c1","c9","d1","d9","ew","sw","ww","nw","rd","gd","wd")
    assert shanten_thirteen_orphans(h) == 0
    assert shanten(h) == 0

def test_tenpai_thirteen_orphans_with_pair():
    # 12 unique orphans + 1 pair among them = tenpai (waiting on the missing orphan)
    h = hand("b1","b1","b9","c1","c9","d1","d9","ew","sw","ww","nw","rd","gd")
    assert shanten_thirteen_orphans(h) == 0


# ---------------------------------------------------------------------------
# Various shanten distances
# ---------------------------------------------------------------------------

def test_shanten_1():
    # One tile away from tenpai
    h = hand("b1","b2", "b4","b5", "c1","c2","c3", "d1","d2","d3", "ew","ew","ew")
    assert shanten(h) == 1

def test_shanten_2():
    # Relatively scattered hand
    h = hand("b1","b3","b5","b7","b9","c1","c3","c5","c7","c9","d1","d3","d5")
    s = shanten(h)
    assert s >= 2

def test_shanten_with_exposed_meld():
    # 1 exposed pong reduces groups_needed to 3
    h = hand("b1","b2", "c1","c2","c3", "d1","d2","d3", "ew","ew")  # 10 concealed
    s = shanten_standard(h, exposed_melds=1)
    assert s == 0  # tenpai waiting on b3

def test_all_isolated_tiles_shanten():
    # 13 completely isolated tiles — worst case
    h = hand("b1","b3","b5","b7","b9","c2","c4","c6","c8","d1","d3","d5","d7")
    s = shanten(h)
    assert s >= 4


# ---------------------------------------------------------------------------
# Acceptance count
# ---------------------------------------------------------------------------

def test_acceptance_tenpai_hand():
    # Tenpai waiting on b3 or b6 (two-sided wait on b4-b5)
    h = hand("b4","b5", "c1","c2","c3", "d1","d2","d3", "ew","ew","ew", "rd","rd")
    wall = full_wall()
    # Remove the tiles in our hand from the wall
    for tid in range(34):
        wall[tid] = max(0, 4 - int(h[tid]))
    acc = acceptance_count(h, wall)
    # Should accept b3 and b6
    assert tile_id("b3") in acc
    assert tile_id("b6") in acc

def test_acceptance_empty_for_complete_hand():
    # Complete hand has shanten -1; no tile reduces it further
    h = hand("b1","b2","b3", "c4","c5","c6", "d7","d8","d9", "ew","ew","ew", "rd","rd")
    wall = full_wall()
    acc = acceptance_count(h, wall)
    assert len(acc) == 0


# ---------------------------------------------------------------------------
# best_discards
# ---------------------------------------------------------------------------

def test_best_discards_prefers_lower_shanten():
    # 14-tile hand: 4 complete groups + 2 isolated honors (rd, gd)
    # Discarding rd or gd leaves 4 groups + 1 isolated tile = tenpai (shanten=0)
    # Discarding any suited tile breaks a group and gives shanten=1
    h = hand("b1","b2","b3", "c1","c2","c3", "d1","d2","d3", "ew","ew","ew", "rd","gd")
    wall = full_wall()
    for tid in range(34):
        wall[tid] = max(0, 4 - int(h[tid]))
    results = best_discards(h, wall)
    assert results[0]["shanten_after"] == 0
    assert results[0]["tile_id"] in (tile_id("rd"), tile_id("gd"))

def test_best_discards_honors_over_suited_when_isolated():
    # If we have an isolated honor and a partial sequence, prefer discarding the honor
    h = hand("b1","b2","b3", "c1","c2","c3", "d1","d2","d3", "ew","ew","ew", "nw","rd")
    wall = full_wall()
    for tid in range(34):
        wall[tid] = max(0, 4 - int(h[tid]))
    results = best_discards(h, wall)
    top_tile_ids = {r["tile_id"] for r in results[:3]}
    # Either nw or rd should be among the top discards (isolated honors)
    assert tile_id("nw") in top_tile_ids or tile_id("rd") in top_tile_ids
