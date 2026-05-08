"""
Tile encoding for Singapore Mahjong.

34 tile types encoded as integers 0-33:
  0-8:   Bamboo 1-9    (b1..b9)
  9-17:  Characters 1-9 (c1..c9)
  18-26: Circles 1-9   (d1..d9)
  27-30: Winds          East, South, West, North
  31-33: Dragons        Red (中), Green (發), White (白)

Flowers (1-4), Seasons (1-4), and Animals (cat, mouse, cockerel, worm)
are outside this scheme — they are set aside immediately on draw and
tracked separately in HandState.
"""

from __future__ import annotations

import numpy as np
from enum import IntEnum

NTILES = 34  # unique tile types in the main pool
COPIES = 4   # copies of each tile type in a full set

# Suit boundaries
BAMBOO_START, BAMBOO_END = 0, 9
CHAR_START, CHAR_END = 9, 18
CIRCLE_START, CIRCLE_END = 18, 27
WIND_START, WIND_END = 27, 31
DRAGON_START, DRAGON_END = 31, 34

SUITED_END = 27  # tiles 0-26 belong to a numbered suit


class Wind(IntEnum):
    EAST = 27
    SOUTH = 28
    WEST = 29
    NORTH = 30


class Dragon(IntEnum):
    RED = 31    # 中
    GREEN = 32  # 發
    WHITE = 33  # 白


# Bonus tile IDs (outside the 0-33 range — stored separately)
FLOWER_SPRING = 34
FLOWER_SUMMER = 35
FLOWER_AUTUMN = 36
FLOWER_WINTER = 37
SEASON_PLUM = 38
SEASON_ORCHID = 39
SEASON_CHRYSANTHEMUM = 40
SEASON_BAMBOO_PLANT = 41
ANIMAL_CAT = 42
ANIMAL_MOUSE = 43
ANIMAL_COCKEREL = 44
ANIMAL_WORM = 45

# Flowers matched by seat wind (East→Spring, South→Summer, West→Autumn, North→Winter)
SEAT_FLOWER: dict[int, int] = {
    Wind.EAST: FLOWER_SPRING,
    Wind.SOUTH: FLOWER_SUMMER,
    Wind.WEST: FLOWER_AUTUMN,
    Wind.NORTH: FLOWER_WINTER,
}
SEAT_SEASON: dict[int, int] = {
    Wind.EAST: SEASON_PLUM,
    Wind.SOUTH: SEASON_ORCHID,
    Wind.WEST: SEASON_CHRYSANTHEMUM,
    Wind.NORTH: SEASON_BAMBOO_PLANT,
}

# Display names indexed by tile ID 0-33
_SUIT_CHARS = {
    range(BAMBOO_START, BAMBOO_END): ("b", "Bamboo"),
    range(CHAR_START, CHAR_END): ("c", "Char"),
    range(CIRCLE_START, CIRCLE_END): ("d", "Circle"),
}
_WIND_NAMES = ["EW", "SW", "WW", "NW"]
_DRAGON_NAMES = ["RD", "GD", "WD"]


def tile_name(tid: int) -> str:
    """Short display name for a tile, e.g. 'b3', 'c7', 'd1', 'EW', 'RD'."""
    if 0 <= tid < SUITED_END:
        suit_char = "bcd"[tid // 9]
        rank = tid % 9 + 1
        return f"{suit_char}{rank}"
    if WIND_START <= tid < WIND_END:
        return _WIND_NAMES[tid - WIND_START]
    if DRAGON_START <= tid < DRAGON_END:
        return _DRAGON_NAMES[tid - DRAGON_START]
    raise ValueError(f"Invalid tile ID: {tid}")


def tile_id(name: str) -> int:
    """Parse a short tile name back to its ID. e.g. 'b3' -> 2, 'EW' -> 27."""
    name = name.strip().lower()
    suit_map = {"b": 0, "c": 9, "d": 18}
    wind_map = {"ew": 27, "sw": 28, "ww": 29, "nw": 30}
    dragon_map = {"rd": 31, "gd": 32, "wd": 33}

    if name in wind_map:
        return wind_map[name]
    if name in dragon_map:
        return dragon_map[name]
    if len(name) == 2 and name[0] in suit_map and name[1].isdigit():
        rank = int(name[1])
        if 1 <= rank <= 9:
            return suit_map[name[0]] + rank - 1
    raise ValueError(f"Unknown tile name: '{name}'")


# ---------------------------------------------------------------------------
# Bonus tile helpers (flowers 34-37, seasons 38-41, animals 42-45)
# ---------------------------------------------------------------------------

_BONUS_NAME_MAP: dict[str, int] = {
    "f1": FLOWER_SPRING,   "spring": FLOWER_SPRING,
    "f2": FLOWER_SUMMER,   "summer": FLOWER_SUMMER,
    "f3": FLOWER_AUTUMN,   "autumn": FLOWER_AUTUMN,
    "f4": FLOWER_WINTER,   "winter": FLOWER_WINTER,
    "s1": SEASON_PLUM,     "plum": SEASON_PLUM,
    "s2": SEASON_ORCHID,   "orchid": SEASON_ORCHID,
    "s3": SEASON_CHRYSANTHEMUM, "chrysanthemum": SEASON_CHRYSANTHEMUM,
    "s4": SEASON_BAMBOO_PLANT,  "bamboo_plant": SEASON_BAMBOO_PLANT,
    "a1": ANIMAL_CAT,      "cat": ANIMAL_CAT,
    "a2": ANIMAL_MOUSE,    "mouse": ANIMAL_MOUSE,
    "a3": ANIMAL_COCKEREL, "cockerel": ANIMAL_COCKEREL,
    "a4": ANIMAL_WORM,     "worm": ANIMAL_WORM,
}

_BONUS_ID_MAP: dict[int, str] = {
    FLOWER_SPRING: "f1(Spring)", FLOWER_SUMMER: "f2(Summer)",
    FLOWER_AUTUMN: "f3(Autumn)", FLOWER_WINTER: "f4(Winter)",
    SEASON_PLUM: "s1(Plum)", SEASON_ORCHID: "s2(Orchid)",
    SEASON_CHRYSANTHEMUM: "s3(Chrysan.)", SEASON_BAMBOO_PLANT: "s4(Bamboo)",
    ANIMAL_CAT: "Cat", ANIMAL_MOUSE: "Mouse",
    ANIMAL_COCKEREL: "Cockerel", ANIMAL_WORM: "Worm",
}


def bonus_tile_id(name: str) -> int:
    """Parse a bonus tile name to its ID. e.g. 'f1', 'spring', 'cat', 'a2'."""
    key = name.strip().lower()
    if key in _BONUS_NAME_MAP:
        return _BONUS_NAME_MAP[key]
    raise ValueError(f"Unknown bonus tile: '{name}'. Use f1-f4, s1-s4, a1-a4 or names like 'spring', 'cat'.")


def bonus_tile_name(bid: int) -> str:
    """Short display name for a bonus tile ID."""
    if bid in _BONUS_ID_MAP:
        return _BONUS_ID_MAP[bid]
    raise ValueError(f"Invalid bonus tile ID: {bid}")


def is_bonus_tile(bid: int) -> bool:
    return 34 <= bid <= 45


def is_animal(bid: int) -> bool:
    return ANIMAL_CAT <= bid <= ANIMAL_WORM


def is_suited(tid: int) -> bool:
    return 0 <= tid < SUITED_END


def is_honor(tid: int) -> bool:
    return WIND_START <= tid < DRAGON_END


def is_terminal(tid: int) -> bool:
    """1 or 9 of a suited tile."""
    return is_suited(tid) and tid % 9 in (0, 8)


def suit_of(tid: int) -> int:
    """0=Bamboo, 1=Characters, 2=Circles. Raises for honors."""
    if not is_suited(tid):
        raise ValueError(f"Tile {tid} has no suit")
    return tid // 9


def rank_of(tid: int) -> int:
    """1-indexed rank within suit. Raises for honors."""
    if not is_suited(tid):
        raise ValueError(f"Tile {tid} has no rank")
    return tid % 9 + 1


def new_hand_array() -> np.ndarray:
    """Return a zeroed 34-element int8 array representing an empty hand."""
    return np.zeros(NTILES, dtype=np.int8)


def full_wall() -> np.ndarray:
    """34-element array with 4 copies of each tile (136 tiles total)."""
    return np.full(NTILES, COPIES, dtype=np.int8)


def tiles_from_names(names: list[str]) -> np.ndarray:
    """Convert a list of tile name strings to a hand array."""
    arr = new_hand_array()
    for n in names:
        arr[tile_id(n)] += 1
    return arr


def names_from_array(arr: np.ndarray) -> list[str]:
    """Expand a hand array to a sorted list of tile names."""
    result = []
    for tid in range(NTILES):
        result.extend([tile_name(tid)] * int(arr[tid]))
    return result
