"""Seat policies for the game engine.

A policy makes the *decisions* for one seat — what to discard and whether to
claim a discard (pong/kong/chow) — while the engine owns all mechanics (wall,
draw, ron resolution, claim priority and execution, chip payment, turn order).

Three concrete policies plug into engine seats:
  - HumanPolicy:     defers the discard to external input (the TUI); never claims.
  - HeuristicPolicy: the fixed-weight risk-aware bot (deduces opponent danger,
                     balances winning vs. shooting) backed by recommend_discard.
  - ModelPolicy:     the RL model (lives in cracked.training.policy_model; needs
                     torch, so it is not imported here).

Ron (winning off a discard) is always taken by the engine and is not a policy
decision.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from cracked.tiles import NTILES
from cracked.game_state import GameState
from cracked.tiles_away import tiles_away
from cracked.optimizer import recommend_discard

_EYE = np.eye(NTILES, dtype=np.int8)


class Policy(Protocol):
    """Decision interface the engine drives for one seat.

    `view` is the seat's observable game state (engine.player_view_for(seat));
    its `my_hand` is that seat's current hand. At any discard decision the seat
    holds a valid 14 - 3*n_melds concealed count.
    """

    def choose_discard(self, view: GameState) -> Optional[int]:
        """Tile id to discard, or None to signal 'await external input'."""
        ...

    def wants_pong(self, view: GameState, tile: int) -> bool:
        ...

    def wants_kong(self, view: GameState, tile: int) -> bool:
        ...

    def choose_chow(
        self, view: GameState, tile: int, options: list[tuple[int, int, int]]
    ) -> Optional[tuple[int, int, int]]:
        ...


class HumanPolicy:
    """Human seat: discard comes from external input; never auto-claims."""

    def choose_discard(self, view: GameState) -> Optional[int]:
        return None

    def wants_pong(self, view: GameState, tile: int) -> bool:
        return False

    def wants_kong(self, view: GameState, tile: int) -> bool:
        return False

    def choose_chow(
        self, view: GameState, tile: int, options: list[tuple[int, int, int]]
    ) -> Optional[tuple[int, int, int]]:
        return None


class HeuristicPolicy:
    """Fixed-weight risk-aware bot.

    Discards via the danger/opponent-model optimizer; claims maintain or improve
    tiles away (kong allows one extra step since the replacement compensates;
    chow requires strict improvement).
    """

    def choose_discard(self, view: GameState) -> Optional[int]:
        results = recommend_discard(view)
        if not results:
            concealed = view.my_hand.concealed
            return int(np.argmax(concealed > 0))
        return results[0].tile_id

    def wants_pong(self, view: GameState, tile: int) -> bool:
        concealed = view.my_hand.concealed
        n_melds = len(view.my_hand.melds)
        if concealed[tile] < 2:
            return False
        current_s = tiles_away(concealed, n_melds)
        if current_s == -1:
            return False
        test = concealed.copy()
        test[tile] -= 2
        best_s = min(
            (tiles_away(test - _EYE[t], n_melds + 1) for t in range(NTILES) if test[t] > 0),
            default=current_s,
        )
        return best_s <= current_s

    def wants_kong(self, view: GameState, tile: int) -> bool:
        concealed = view.my_hand.concealed
        n_melds = len(view.my_hand.melds)
        if concealed[tile] < 3:
            return False
        current_s = tiles_away(concealed, n_melds)
        if current_s == -1:
            return False
        test = concealed.copy()
        test[tile] -= 3
        best_s = min(
            (tiles_away(test - _EYE[t], n_melds + 1) for t in range(NTILES) if test[t] > 0),
            default=current_s,
        )
        return best_s <= current_s + 1

    def choose_chow(
        self, view: GameState, tile: int, options: list[tuple[int, int, int]]
    ) -> Optional[tuple[int, int, int]]:
        if not options:
            return None
        concealed = view.my_hand.concealed
        n_melds = len(view.my_hand.melds)
        current_s = tiles_away(concealed, n_melds)
        best_option: Optional[tuple[int, int, int]] = None
        best_s = current_s  # chow requires strict improvement
        for chow_tiles in options:
            test = concealed.copy()
            for t in chow_tiles:
                if t != tile:
                    test[t] -= 1
            min_s = min(
                (tiles_away(test - _EYE[t], n_melds + 1) for t in range(NTILES) if test[t] > 0),
                default=current_s,
            )
            if min_s < best_s:
                best_s = min_s
                best_option = chow_tiles
        return best_option
