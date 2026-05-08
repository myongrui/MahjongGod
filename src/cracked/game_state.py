"""Observable game state for Singapore Mahjong."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cracked.tiles import NTILES, full_wall, new_hand_array, Wind
from cracked.hand import HandState, Meld, MeldType


def _default_state_file() -> Path:
    return Path(os.environ.get("CRACKED_STATE_FILE", ".cracked_game.json"))


@dataclass
class PlayerView:
    """Observable state of one opponent."""
    seat: int                                        # Wind constant (27-30)
    discards: list[int] = field(default_factory=list)  # tile IDs in order discarded
    melds: list[Meld] = field(default_factory=list)
    flowers: list[int] = field(default_factory=list)   # bonus tile IDs (34-45)


@dataclass
class GameState:
    my_hand: HandState
    my_seat: int
    prevailing_wind: int
    opponents: list[PlayerView]          # all 3 opponents, identified by seat
    wall_tiles_remaining: int = 136
    turn_number: int = 0

    def visible_tiles(self) -> np.ndarray:
        """All tiles we can account for: our hand + all discards + all exposed melds."""
        arr = self.my_hand.concealed.copy()
        for m in self.my_hand.melds:
            for t in m.tiles:
                arr[t] += 1
        for opp in self.opponents:
            for t in opp.discards:
                arr[t] += 1
            for m in opp.melds:
                for t in m.tiles:
                    arr[t] += 1
        return arr

    def unknown_tiles(self) -> np.ndarray:
        """Tiles not visible to us (wall + opponents' concealed hands)."""
        return np.clip(full_wall() - self.visible_tiles(), 0, 4)

    def opponent_by_seat(self, seat: int) -> PlayerView:
        for opp in self.opponents:
            if opp.seat == seat:
                return opp
        raise ValueError(f"No opponent with seat {seat}")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "my_hand": _hand_to_dict(self.my_hand),
            "my_seat": self.my_seat,
            "prevailing_wind": self.prevailing_wind,
            "opponents": [_player_to_dict(o) for o in self.opponents],
            "wall_tiles_remaining": self.wall_tiles_remaining,
            "turn_number": self.turn_number,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GameState:
        return cls(
            my_hand=_hand_from_dict(d["my_hand"]),
            my_seat=d["my_seat"],
            prevailing_wind=d["prevailing_wind"],
            opponents=[_player_from_dict(o) for o in d["opponents"]],
            wall_tiles_remaining=d["wall_tiles_remaining"],
            turn_number=d["turn_number"],
        )


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------

def _meld_to_dict(m: Meld) -> dict:
    return {
        "type": m.type.value,
        "tiles": list(m.tiles),
        "concealed": m.concealed,
        "source_player": m.source_player,
    }


def _meld_from_dict(d: dict) -> Meld:
    return Meld(
        type=MeldType(d["type"]),
        tiles=tuple(d["tiles"]),
        concealed=d["concealed"],
        source_player=d.get("source_player"),
    )


def _hand_to_dict(h: HandState) -> dict:
    return {
        "concealed": h.concealed.tolist(),
        "melds": [_meld_to_dict(m) for m in h.melds],
        "flowers": h.flowers,
        "animals": h.animals,
        "seat_wind": h.seat_wind,
    }


def _hand_from_dict(d: dict) -> HandState:
    return HandState(
        concealed=np.array(d["concealed"], dtype=np.int8),
        melds=[_meld_from_dict(m) for m in d["melds"]],
        flowers=d.get("flowers", []),
        animals=d.get("animals", []),
        seat_wind=d["seat_wind"],
    )


def _player_to_dict(p: PlayerView) -> dict:
    return {
        "seat": p.seat,
        "discards": p.discards,
        "melds": [_meld_to_dict(m) for m in p.melds],
        "flowers": p.flowers,
    }


def _player_from_dict(d: dict) -> PlayerView:
    return PlayerView(
        seat=d["seat"],
        discards=d["discards"],
        melds=[_meld_from_dict(m) for m in d["melds"]],
        flowers=d.get("flowers", []),
    )


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def save_state(state: GameState, path: Path | None = None) -> None:
    p = path or _default_state_file()
    p.write_text(json.dumps(state.to_dict(), indent=2))


def load_state(path: Path | None = None) -> GameState:
    p = path or _default_state_file()
    if not p.exists():
        raise FileNotFoundError(f"No active game found at '{p}'. Run 'cracked new-game' first.")
    return GameState.from_dict(json.loads(p.read_text()))
