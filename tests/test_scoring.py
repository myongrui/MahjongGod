"""
Scoring tests for Singapore Mahjong tai (台) engine.

All hands are complete winning hands (tiles_away = -1).
"""

import pytest
from cracked.tiles import tile_id, Wind, Dragon
from cracked.hand import HandState, Meld, MeldType
from cracked.scoring import (
    HouseRules, WinContext, TaiResult, calculate_tai,
    DEFAULT_RULES,
)

# Shared rules for tests — common Singapore ruleset
RULES = HouseRules(tai_cap=5, min_tai=3)
EAST_CTX = WinContext(winning_tile=tile_id("b1"), prevailing_wind=Wind.EAST)


def hand_from(*names, seat=Wind.EAST, flowers=None, animals=None, melds=None) -> HandState:
    h = HandState.from_tile_names(list(names), seat_wind=seat)
    if flowers:
        h.flowers = list(flowers)
    if animals:
        h.animals = list(animals)
    if melds:
        for m in melds:
            h.add_meld(m)
    return h


def pong_meld(name: str, concealed=False) -> Meld:
    tid = tile_id(name)
    return Meld(MeldType.PONG, (tid, tid, tid), concealed=concealed)


def chow_meld(a: str, b: str, c: str) -> Meld:
    return Meld(MeldType.CHOW, (tile_id(a), tile_id(b), tile_id(c)))


# ---------------------------------------------------------------------------
# Basic win-context tai
# ---------------------------------------------------------------------------

def test_self_draw_is_not_a_tai_but_scores_men_qing():
    # Self-draw is a payout multiplier, not a tai. A fully-concealed self-draw
    # scores 門清 instead.
    h = hand_from("b1","b2","b3","b4","b5","b6","b7","b8","b9","b1","b2","b3","b4","b4")
    ctx = WinContext(winning_tile=tile_id("b4"), is_self_draw=True, prevailing_wind=Wind.EAST)
    result = calculate_tai(h, ctx, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Self-draw (自摸)" not in names
    assert "Fully concealed self-draw (門清)" in names

def test_last_tile_adds_one_tai():
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd")
    ctx = WinContext(winning_tile=tile_id("b3"), is_last_tile=True, prevailing_wind=Wind.EAST)
    result = calculate_tai(h, ctx, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Last tile — 海底撈月" in names


# ---------------------------------------------------------------------------
# Dragon pongs
# ---------------------------------------------------------------------------

def test_dragon_pong_one_tai():
    # 1 dragon pong (RD) = 1 tai
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","rd","rd","rd","ew","ew")
    result = calculate_tai(h, EAST_CTX, RULES)
    bd = dict(result.breakdown)
    assert bd.get("RD dragon pong") == 1

def test_two_dragon_pongs():
    h = hand_from("b1","b2","b3","c1","c2","c3","rd","rd","rd","gd","gd","gd","ew","ew")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "RD dragon pong" in names
    assert "GD dragon pong" in names

def test_small_three_dragons():
    # 2 dragon pongs + 1 dragon pair = small three dragons (小三元)
    h = hand_from("b1","b2","b3","rd","rd","rd","gd","gd","gd","wd","wd","b4","b5","b6")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Small three dragons (小三元)" in names


# ---------------------------------------------------------------------------
# Wind pongs
# ---------------------------------------------------------------------------

def test_seat_wind_pong():
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd",
                  seat=Wind.EAST)
    ctx = WinContext(winning_tile=tile_id("b3"), prevailing_wind=Wind.SOUTH)
    result = calculate_tai(h, ctx, RULES)
    names = {name for name, _ in result.breakdown}
    assert "East wind pong" in names

def test_prevailing_wind_pong():
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd",
                  seat=Wind.SOUTH)
    ctx = WinContext(winning_tile=tile_id("b3"), prevailing_wind=Wind.EAST)
    result = calculate_tai(h, ctx, RULES)
    names = {name for name, _ in result.breakdown}
    assert "East wind pong" in names

def test_double_wind_pong():
    # Seat = prevailing = East → 2 tai for the pong
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd",
                  seat=Wind.EAST)
    ctx = WinContext(winning_tile=tile_id("b3"), prevailing_wind=Wind.EAST)
    result = calculate_tai(h, ctx, RULES)
    bd = dict(result.breakdown)
    assert bd.get("East wind pong (double wind)") == 2

def test_non_seat_non_prevailing_wind_pong_gives_no_tai():
    # West wind pong when seat=East, prevailing=South → 0 tai from that pong
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ww","ww","ww","rd","rd",
                  seat=Wind.EAST)
    ctx = WinContext(winning_tile=tile_id("b3"), prevailing_wind=Wind.SOUTH)
    result = calculate_tai(h, ctx, RULES)
    names = {name for name, _ in result.breakdown}
    assert "West wind pong" not in names
    assert "West wind pong (double wind)" not in names


# ---------------------------------------------------------------------------
# All pongs
# ---------------------------------------------------------------------------

def test_all_pongs():
    h = hand_from("b1","b1","b1","c2","c2","c2","d3","d3","d3","ew","ew","ew","rd","rd")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "All pongs (對對胡)" in names

def test_mixed_hand_no_all_pongs():
    # Has a sequence → not all pongs
    h = hand_from("b1","b2","b3","c2","c2","c2","d3","d3","d3","ew","ew","ew","rd","rd")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "All pongs (對對胡)" not in names


# ---------------------------------------------------------------------------
# Flush hands
# ---------------------------------------------------------------------------

def test_full_flush():
    # All bamboo
    h = hand_from("b1","b2","b3","b4","b5","b6","b7","b8","b9","b1","b2","b3","b4","b4")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Full flush (清一色)" in names
    assert "Half flush (混一色)" not in names

def test_half_flush():
    # One suit + honor tiles
    h = hand_from("b1","b2","b3","b4","b5","b6","b7","b8","b9","ew","ew","ew","rd","rd")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Half flush (混一色)" in names
    assert "Full flush (清一色)" not in names

def test_multi_suit_no_flush():
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Full flush (清一色)" not in names
    assert "Half flush (混一色)" not in names


# ---------------------------------------------------------------------------
# Seven pairs
# ---------------------------------------------------------------------------

def test_seven_pairs_base():
    # Non-runnable pairs (no four can form sets) → unambiguously seven pairs.
    h = hand_from("b1","b1","b4","b4","b7","b7","c2","c2","c5","c5","c8","c8","d9","d9")
    result = calculate_tai(h, EAST_CTX, HouseRules(tai_cap=5, min_tai=3, seven_pairs_base=3))
    names = {name for name, _ in result.breakdown}
    assert "Seven pairs (七對子)" in names
    bd = dict(result.breakdown)
    assert bd["Seven pairs (七對子)"] == 3

def test_seven_pairs_with_flush():
    # All bamboo, but ranks 6/7/9 can't form any set → only seven pairs applies.
    h = hand_from("b1","b1","b2","b2","b3","b3","b4","b4","b6","b6","b7","b7","b9","b9")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Seven pairs (七對子)" in names
    assert "Full flush (清一色)" in names


# ---------------------------------------------------------------------------
# Limit hands
# ---------------------------------------------------------------------------

def test_big_three_dragons_is_limit():
    h = hand_from("b1","b2","b3","rd","rd","rd","gd","gd","gd","wd","wd","wd","ew","ew")
    result = calculate_tai(h, EAST_CTX, RULES)
    assert result.total >= RULES.tai_cap
    names = {name for name, _ in result.breakdown}
    assert "Big three dragons (大三元)" in names

def test_thirteen_orphans_is_limit():
    h = hand_from("b1","b9","c1","c9","d1","d9","ew","sw","ww","nw","rd","gd","wd","b1")
    result = calculate_tai(h, EAST_CTX, RULES)
    assert result.total >= RULES.tai_cap
    names = {name for name, _ in result.breakdown}
    assert "Thirteen orphans (十三幺)" in names

def test_all_honors_is_limit():
    h = hand_from("ew","ew","ew","sw","sw","sw","ww","ww","ww","rd","rd","rd","gd","gd")
    result = calculate_tai(h, EAST_CTX, RULES)
    assert result.total >= RULES.tai_cap
    names = {name for name, _ in result.breakdown}
    assert "All honors (字一色)" in names

def test_tai_capped_at_cap():
    # Full flush (4) + all pongs (2) + dragon pong (1) > cap of 5
    h = hand_from("b1","b1","b1","b2","b2","b2","b3","b3","b3","b4","b4","b4","b5","b5")
    result = calculate_tai(h, EAST_CTX, RULES)
    assert result.total == RULES.tai_cap
    assert result.capped


# ---------------------------------------------------------------------------
# Flower / season scoring
# ---------------------------------------------------------------------------

def test_seat_flower_adds_tai():
    from cracked.tiles import FLOWER_SPRING
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd",
                  seat=Wind.EAST, flowers=[FLOWER_SPRING])
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Seat flower" in names

def test_non_matching_flower_gives_no_tai():
    from cracked.tiles import FLOWER_SUMMER  # Summer matches South, not East
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd",
                  seat=Wind.EAST, flowers=[FLOWER_SUMMER])
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Flower/season" not in names
    assert "Seat flower" not in names
    assert "Seat season" not in names

def test_matching_season_adds_tai():
    from cracked.tiles import SEASON_PLUM  # Plum matches East seat
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd",
                  seat=Wind.EAST, flowers=[SEASON_PLUM])
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "Seat season" in names

def test_animals_always_give_one_tai_each():
    from cracked.tiles import ANIMAL_CAT, ANIMAL_MOUSE
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd",
                  seat=Wind.EAST, animals=[ANIMAL_CAT, ANIMAL_MOUSE])
    result = calculate_tai(h, EAST_CTX, RULES)
    animal_entries = [(n, v) for n, v in result.breakdown if n == "Animal"]
    assert len(animal_entries) == 2
    assert all(v == 1 for _, v in animal_entries)

def test_all_four_animals_give_four_tai():
    from cracked.tiles import ANIMAL_CAT, ANIMAL_MOUSE, ANIMAL_COCKEREL, ANIMAL_WORM
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","rd",
                  seat=Wind.EAST, animals=[ANIMAL_CAT, ANIMAL_MOUSE, ANIMAL_COCKEREL, ANIMAL_WORM])
    result = calculate_tai(h, EAST_CTX, RULES)
    animal_tai = sum(v for n, v in result.breakdown if n == "Animal")
    assert animal_tai == 4


# ---------------------------------------------------------------------------
# Exposed melds
# ---------------------------------------------------------------------------

def test_exposed_meld_contributes_to_scoring():
    # Exposed dragon pong (RD) + concealed hand forming remaining groups
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew",
                  melds=[pong_meld("rd")])
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "RD dragon pong" in names

def test_all_pongs_with_exposed_pong():
    h = hand_from("b1","b1","b1","c2","c2","c2","d3","d3","d3","rd","rd",
                  melds=[pong_meld("ew")])
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {name for name, _ in result.breakdown}
    assert "All pongs (對對胡)" in names


# ---------------------------------------------------------------------------
# Validity check
# ---------------------------------------------------------------------------

def test_min_tai_check():
    rules = HouseRules(tai_cap=5, min_tai=3)
    # All-sequence shape but won on a closed wait by discard with no bonus tiles:
    # the 平胡 4 tai does not count, so the hand is 0 tai → invalid win.
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","b4","b5","b6","b8","b8")
    ctx = WinContext(winning_tile=tile_id("b5"), prevailing_wind=Wind.EAST)  # b5 = middle of b4b5b6
    result = calculate_tai(h, ctx, rules)
    assert not result.is_valid_win(rules)


# ---------------------------------------------------------------------------
# PDF-alignment fixes (Batch 1)
# ---------------------------------------------------------------------------

def test_men_qing_only_when_no_exposed_melds():
    # Self-draw with an exposed pong is NOT 門清.
    h = hand_from("b1","b1","b1","c2","c2","c2","d3","d3","d3","rd","rd",
                  melds=[pong_meld("ew")])
    ctx = WinContext(winning_tile=tile_id("rd"), is_self_draw=True, prevailing_wind=Wind.EAST)
    names = {n for n, _ in calculate_tai(h, ctx, RULES).breakdown}
    assert "Fully concealed self-draw (門清)" not in names

def test_small_three_dragons_scores_three():
    # Mixed-suit (no flush) 小三元: 2 dragon pongs (2) + pair bonus (1) = 3.
    h = hand_from("b1","b2","b3","rd","rd","rd","gd","gd","gd","wd","wd","c4","c5","c6")
    bd = dict(calculate_tai(h, EAST_CTX, RULES).breakdown)
    assert bd["Small three dragons (小三元)"] == 1
    assert calculate_tai(h, EAST_CTX, RULES).total == 3

def test_small_four_winds_hand_bonus_is_two():
    # 3 wind pongs + wind pair; the 小四喜 hand bonus itself is 2 (not the limit).
    h = hand_from("ew","ew","ew","sw","sw","sw","ww","ww","ww","nw","nw","b1","b2","b3",
                  seat=Wind.SOUTH)
    ctx = WinContext(winning_tile=tile_id("b3"), prevailing_wind=Wind.EAST)
    bd = dict(calculate_tai(h, ctx, RULES).breakdown)
    assert bd["Small four winds (小四喜)"] == 2

def test_bonus_tiles_count_within_cap():
    from cracked.tiles import ANIMAL_CAT, ANIMAL_MOUSE
    # Full flush (4) + 2 animals would be 6, but bonus counts within the cap → 5.
    h = hand_from("b1","b2","b3","b4","b5","b6","b7","b8","b9","b1","b2","b3","b4","b4",
                  animals=[ANIMAL_CAT, ANIMAL_MOUSE])
    result = calculate_tai(h, EAST_CTX, RULES)
    assert result.total == 5

def test_all_four_animals_total_five():
    from cracked.tiles import ANIMAL_CAT, ANIMAL_MOUSE, ANIMAL_COCKEREL, ANIMAL_WORM
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","b4","b5","b6","b7","b7",
                  animals=[ANIMAL_CAT, ANIMAL_MOUSE, ANIMAL_COCKEREL, ANIMAL_WORM])
    result = calculate_tai(h, EAST_CTX, HouseRules(tai_cap=10, min_tai=1))
    # four single points + one all-four bonus = 5
    animal_tai = sum(v for n, v in result.breakdown if n in ("Animal", "All four animals"))
    assert animal_tai == 5

def test_complete_flower_group_bonus():
    from cracked.tiles import (FLOWER_SPRING, FLOWER_SUMMER, FLOWER_AUTUMN, FLOWER_WINTER)
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","b4","b5","b6","b7","b7",
                  seat=Wind.EAST,
                  flowers=[FLOWER_SPRING, FLOWER_SUMMER, FLOWER_AUTUMN, FLOWER_WINTER])
    names = {n for n, _ in calculate_tai(h, EAST_CTX, HouseRules(tai_cap=10, min_tai=1)).breakdown}
    assert "Complete flower group (一台花)" in names
    assert "Seat flower" in names  # the matching-tile point still counts separately


# ---------------------------------------------------------------------------
# Sequence hand + missing hands (Batch 2)
# ---------------------------------------------------------------------------

def test_sequence_hand_two_sided_discard():
    # All sequences, suited pair, two-sided wait on a discard → 平胡 = 4.
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","b4","b5","b6","d5","d5")
    ctx = WinContext(winning_tile=tile_id("b1"), prevailing_wind=Wind.EAST)  # low end of b1b2b3
    result = calculate_tai(h, ctx, RULES)
    bd = dict(result.breakdown)
    assert bd.get("Sequence hand (平胡)") == 4
    assert result.total == 4

def test_lesser_sequence_hand_with_bonus():
    from cracked.tiles import ANIMAL_CAT
    # Same shape but a bonus tile was drawn → 小平胡 (1), plus the animal.
    h = hand_from("b1","b2","b3","c1","c2","c3","d1","d2","d3","b4","b5","b6","d5","d5",
                  animals=[ANIMAL_CAT])
    ctx = WinContext(winning_tile=tile_id("b1"), prevailing_wind=Wind.EAST)
    names = {n for n, _ in calculate_tai(h, ctx, HouseRules(tai_cap=5, min_tai=1)).breakdown}
    assert "Lesser sequence hand (小平胡)" in names
    assert "Sequence hand (平胡)" not in names

def test_mixed_terminals_scores_four():
    h = hand_from("b1","b1","b1","b9","b9","b9","c1","c1","c1","nw","nw","nw","d9","d9",
                  seat=Wind.EAST)
    ctx = WinContext(winning_tile=tile_id("d9"), prevailing_wind=Wind.EAST)
    result = calculate_tai(h, ctx, RULES)
    names = {n for n, _ in result.breakdown}
    assert "Mixed terminals (混么九)" in names
    assert result.total == 4  # All-Pongs (2) + Mixed terminals (2)

def test_pure_terminals_is_limit():
    h = hand_from("b1","b1","b1","b9","b9","b9","c1","c1","c1","d9","d9","d9","c9","c9")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {n for n, _ in result.breakdown}
    assert "Pure terminals (清么九)" in names
    assert result.total == RULES.tai_cap

def test_pure_green_scores_four():
    # bamboo 2,3,4 + green dragon + 8 pair — all green tiles.
    h = hand_from("b2","b2","b2","b3","b3","b3","b4","b4","b4","gd","gd","gd","b8","b8")
    result = calculate_tai(h, EAST_CTX, RULES)
    names = {n for n, _ in result.breakdown}
    assert "Pure green (绿一色)" in names

def test_nine_gates_is_limit():
    h = hand_from("b1","b1","b1","b2","b3","b4","b5","b5","b6","b7","b8","b9","b9","b9")
    ctx = WinContext(winning_tile=tile_id("b5"), prevailing_wind=Wind.EAST)
    result = calculate_tai(h, ctx, RULES)
    names = {n for n, _ in result.breakdown}
    assert "Nine gates (九连宝灯)" in names
    assert result.total == RULES.tai_cap

def test_hidden_treasure_is_limit():
    # Four concealed triplets + pair, self-drawn.
    h = hand_from("b1","b1","b1","c2","c2","c2","d3","d3","d3","b5","b5","b5","rd","rd")
    ctx = WinContext(winning_tile=tile_id("b5"), is_self_draw=True, prevailing_wind=Wind.EAST)
    result = calculate_tai(h, ctx, RULES)
    names = {n for n, _ in result.breakdown}
    assert "Hidden treasure (四暗刻)" in names
    assert result.total == RULES.tai_cap

def test_hidden_treasure_not_on_discard():
    # Same shape but won on a discard → not 四暗刻 (just a triplets hand).
    h = hand_from("b1","b1","b1","c2","c2","c2","d3","d3","d3","b5","b5","b5","rd","rd")
    ctx = WinContext(winning_tile=tile_id("b5"), is_self_draw=False, prevailing_wind=Wind.EAST)
    names = {n for n, _ in calculate_tai(h, ctx, RULES).breakdown}
    assert "Hidden treasure (四暗刻)" not in names


# ---------------------------------------------------------------------------
# Timing limit hands (天胡 / 地胡 / 人胡)
# ---------------------------------------------------------------------------

def test_heavenly_hand_is_limit():
    h = hand_from("b1","b1","b1","c2","c2","c2","d3","d3","d3","ew","ew","ew","rd","rd")
    ctx = WinContext(winning_tile=tile_id("rd"), is_self_draw=True, is_heavenly=True,
                     prevailing_wind=Wind.EAST)
    names = {n for n, _ in calculate_tai(h, ctx, RULES).breakdown}
    assert "Heavenly hand (天胡)" in names
    assert calculate_tai(h, ctx, RULES).total == RULES.tai_cap

def test_earthly_hand_is_limit():
    h = hand_from("b1","b1","b1","c2","c2","c2","d3","d3","d3","ew","ew","ew","rd","rd")
    ctx = WinContext(winning_tile=tile_id("rd"), is_earthly=True, prevailing_wind=Wind.EAST)
    names = {n for n, _ in calculate_tai(h, ctx, RULES).breakdown}
    assert "Earthly hand (地胡)" in names

def test_humanly_hand_is_limit():
    h = hand_from("b1","b1","b1","c2","c2","c2","d3","d3","d3","ew","ew","ew","rd","rd")
    ctx = WinContext(winning_tile=tile_id("rd"), is_humanly=True, prevailing_wind=Wind.EAST)
    names = {n for n, _ in calculate_tai(h, ctx, RULES).breakdown}
    assert "Humanly hand (人胡)" in names
