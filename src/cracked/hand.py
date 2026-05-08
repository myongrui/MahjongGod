"""Hand state representation for Singapore Mahjong."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from cracked.tiles import NTILES, new_hand_array, tile_name, tile_id, tiles_from_names


class MeldType(Enum):
    CHOW = "chow"    # sequence of 3 suited tiles
    PONG = "pong"    # triplet
    KONG = "kong"    # quad (exposed or concealed)


@dataclass
class Meld:
    type: MeldType
    tiles: tuple[int, ...]   # tile IDs in the meld (3 for chow/pong, 4 for kong)
    concealed: bool = False  # True only for concealed kongs
    source_player: int | None = None  # seat index of the player whose discard completed this meld (None = self-draw)

    def __post_init__(self):
        self.tiles = tuple(sorted(self.tiles))

    def __repr__(self) -> str:
        names = " ".join(tile_name(t) for t in self.tiles)
        tag = "CONC" if self.concealed else "EXP"
        return f"{self.type.value.upper()}[{names}]({tag})"


@dataclass
class HandState:
    """
    Complete state of one player's hand.

    concealed: 34-element int8 array — tile counts for tiles not yet exposed.
    melds: list of exposed (or concealed-kong) melds.
    flowers: bonus flower/season tile IDs held (set aside on draw).
    animals: bonus animal tile IDs held (set aside on draw).
    seat_wind: seat wind tile ID (Wind.EAST etc.)
    """

    concealed: np.ndarray = field(default_factory=new_hand_array)
    melds: list[Meld] = field(default_factory=list)
    flowers: list[int] = field(default_factory=list)
    animals: list[int] = field(default_factory=list)
    seat_wind: int = 27  # default East

    @property
    def total_concealed(self) -> int:
        return int(self.concealed.sum())

    @property
    def meld_count(self) -> int:
        """Number of exposed meld sets (concealed kongs count as exposed for hand-size purposes)."""
        return len(self.melds)

    @property
    def expected_concealed(self) -> int:
        """How many concealed tiles a complete hand needs given current melds."""
        return 14 - 3 * len(self.melds)

    def add_tile(self, tid: int) -> None:
        if not (0 <= tid < NTILES):
            raise ValueError(f"Invalid tile ID: {tid}")
        self.concealed[tid] += 1

    def remove_tile(self, tid: int) -> None:
        if self.concealed[tid] <= 0:
            raise ValueError(f"Tile {tile_name(tid)} not in concealed hand")
        self.concealed[tid] -= 1

    def concealed_after_discard(self, tid: int) -> np.ndarray:
        """Return a new array representing the concealed hand after discarding tile tid."""
        if self.concealed[tid] <= 0:
            raise ValueError(f"Tile {tile_name(tid)} not in concealed hand")
        arr = self.concealed.copy()
        arr[tid] -= 1
        return arr

    def add_meld(self, meld: Meld) -> None:
        self.melds.append(meld)

    def concealed_tiles_list(self) -> list[int]:
        """Expand concealed array to a sorted list of tile IDs."""
        result = []
        for tid in range(NTILES):
            result.extend([tid] * int(self.concealed[tid]))
        return result

    @classmethod
    def from_tile_names(cls, names: list[str], seat_wind: int = 27, **kwargs) -> HandState:
        """Convenience constructor from a list of tile name strings."""
        arr = tiles_from_names(names)
        return cls(concealed=arr, seat_wind=seat_wind, **kwargs)

    def copy(self) -> HandState:
        return HandState(
            concealed=self.concealed.copy(),
            melds=list(self.melds),
            flowers=list(self.flowers),
            animals=list(self.animals),
            seat_wind=self.seat_wind,
        )

    def __repr__(self) -> str:
        tiles = " ".join(tile_name(t) for t in self.concealed_tiles_list())
        melds = " | ".join(repr(m) for m in self.melds)
        parts = [f"Concealed: [{tiles}]"]
        if melds:
            parts.append(f"Melds: {melds}")
        if self.flowers:
            parts.append(f"Flowers: {self.flowers}")
        if self.animals:
            parts.append(f"Animals: {self.animals}")
        return " / ".join(parts)
