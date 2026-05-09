"""Turn-by-turn game engine for Singapore Mahjong.

Synchronous state machine that owns the full ground truth of a game
(all 4 hands, wall, discard piles). Drives both spectator and interactive
TUI modes — the UI calls step() on a timer and submit_discard() on human input.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from cracked.tiles import NTILES, Wind, tile_name
from cracked.hand import HandState
from cracked.game_state import GameState, PlayerView
from cracked.shanten import shanten
from cracked.simulator import SimHand, _heuristic_discard
from cracked.optimizer import recommend_discard, DiscardRecommendation

_WIND_ORDER: list[int] = [int(Wind.EAST), int(Wind.SOUTH), int(Wind.WEST), int(Wind.NORTH)]


class EventType(Enum):
    DEAL            = "deal"
    DRAW            = "draw"
    DISCARD         = "discard"
    WIN_SELF_DRAW   = "win_self_draw"
    WIN_DISCARD     = "win_discard"
    WALL_EXHAUSTED  = "wall_exhausted"
    AWAIT_DISCARD   = "await_discard"


@dataclass
class GameEvent:
    type:   EventType
    seat:   int               # Wind constant of the acting player (-1 = no player)
    tile:   Optional[int] = None
    detail: dict = field(default_factory=dict)


@dataclass
class PlayerState:
    seat:     int
    hand:     HandState
    discards: list[int] = field(default_factory=list)
    is_human: bool = False


class GameEngine:
    """
    Turn-by-turn 4-player game engine.

    Spectator mode: human_seats=None  → step() always processes AI turns.
    Interactive:    human_seats={seat} → step() returns AWAIT_DISCARD when
                    it is the human's turn; caller must call submit_discard().
    """

    MAX_ROUNDS = 40

    def __init__(
        self,
        human_seats: Optional[set[int]] = None,
        prevailing_wind: int = int(Wind.EAST),
        seed: Optional[int] = None,
    ) -> None:
        self._human_seats: set[int] = human_seats or set()
        self.prevailing_wind = prevailing_wind
        self._rng = random.Random(seed)

        self.players: dict[int, PlayerState] = {}
        self._wall: list[int] = []
        self._wall_idx: int = 0
        self.turn_number: int = 0
        self._seat_idx: int = 0          # index into _WIND_ORDER
        self._phase: str = "not_started"
        self.winner: Optional[int] = None
        self._awaiting_discard: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_seat(self) -> int:
        return _WIND_ORDER[self._seat_idx]

    @property
    def wall_remaining(self) -> int:
        return max(0, len(self._wall) - self._wall_idx)

    @property
    def is_finished(self) -> bool:
        return self._phase == "finished"

    @property
    def awaiting_human_discard(self) -> bool:
        return self._awaiting_discard

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deal(self) -> list[GameEvent]:
        """Shuffle wall and deal 13 tiles to each player. Resets full state."""
        flat: list[int] = [tid for tid in range(NTILES) for _ in range(4)]
        self._rng.shuffle(flat)
        self._wall = flat
        self._wall_idx = 0

        self.players = {}
        for seat in _WIND_ORDER:
            hand = HandState(seat_wind=seat)
            for _ in range(13):
                hand.add_tile(self._wall[self._wall_idx])
                self._wall_idx += 1
            self.players[seat] = PlayerState(
                seat=seat,
                hand=hand,
                is_human=(seat in self._human_seats),
            )

        self._phase = "playing"
        self._seat_idx = 0
        self.turn_number = 0
        self._awaiting_discard = False
        self.winner = None

        return [GameEvent(EventType.DEAL, seat=-1,
                          detail={"wall_remaining": self.wall_remaining})]

    def step(self) -> list[GameEvent]:
        """
        Advance one turn for the current player.

        - AI players:   returns [DRAW, DISCARD] or [DRAW, WIN_SELF_DRAW]
        - Human player: returns [DRAW, AWAIT_DISCARD]; call submit_discard() next
        - Wall empty:   returns [WALL_EXHAUSTED]
        - Game finished: returns []
        """
        if self._phase == "finished":
            return []
        if self._awaiting_discard:
            return [GameEvent(EventType.AWAIT_DISCARD, seat=self.current_seat)]
        if self._wall_idx >= len(self._wall):
            self._phase = "finished"
            return [GameEvent(EventType.WALL_EXHAUSTED, seat=-1)]

        seat = self.current_seat
        player = self.players[seat]
        events: list[GameEvent] = []

        # Draw
        drawn = self._wall[self._wall_idx]
        self._wall_idx += 1
        self.turn_number += 1
        player.hand.add_tile(drawn)
        events.append(GameEvent(EventType.DRAW, seat=seat, tile=drawn,
                                detail={"wall_remaining": self.wall_remaining}))

        # Self-draw win check
        if shanten(player.hand.concealed, len(player.hand.melds)) == -1:
            self._phase = "finished"
            self.winner = seat
            events.append(GameEvent(EventType.WIN_SELF_DRAW, seat=seat, tile=drawn))
            return events

        # Human seat: pause and wait for discard input
        if player.is_human:
            self._awaiting_discard = True
            events.append(GameEvent(EventType.AWAIT_DISCARD, seat=seat, tile=drawn))
            return events

        # AI seat: choose and perform discard
        events.extend(self._execute_discard(seat))
        return events

    def submit_discard(self, tile_id: int) -> list[GameEvent]:
        """Human player submits their discard choice. Returns DISCARD + ron events."""
        if not self._awaiting_discard:
            raise ValueError("Not awaiting a human discard")
        seat = self.current_seat
        if self.players[seat].hand.concealed[tile_id] <= 0:
            raise ValueError(f"{tile_name(tile_id)} is not in your hand")
        self._awaiting_discard = False
        return self._execute_discard(seat, forced=tile_id)

    def get_recommendations(self) -> list[DiscardRecommendation]:
        """Run the heuristic optimizer from the human player's perspective."""
        if not self._awaiting_discard:
            raise ValueError("Not awaiting a human discard")
        return recommend_discard(self.player_view_for(self.current_seat))

    def player_view_for(self, seat: int) -> GameState:
        """Build a GameState representing one player's observable game state."""
        player = self.players[seat]
        opponents = [
            PlayerView(seat=w, discards=list(self.players[w].discards),
                       melds=list(self.players[w].hand.melds))
            for w in _WIND_ORDER if w != seat
        ]
        return GameState(
            my_hand=player.hand.copy(),
            my_seat=seat,
            prevailing_wind=self.prevailing_wind,
            opponents=opponents,
            wall_tiles_remaining=self.wall_remaining,
            turn_number=self.turn_number,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_discard(self, seat: int, forced: Optional[int] = None) -> list[GameEvent]:
        """Remove a tile from seat's hand, record it, check for ron wins."""
        player = self.players[seat]
        events: list[GameEvent] = []

        if forced is not None:
            tid = forced
        else:
            sim = SimHand(player.hand.concealed.copy(), len(player.hand.melds), seat)
            tid = _heuristic_discard(sim)

        player.hand.remove_tile(tid)
        player.discards.append(tid)
        events.append(GameEvent(EventType.DISCARD, seat=seat, tile=tid))

        # Ron check: any other player wins from this discard?
        for claimer_seat in _WIND_ORDER:
            if claimer_seat == seat:
                continue
            claimer = self.players[claimer_seat]
            claimer.hand.concealed[tid] += 1
            wins = shanten(claimer.hand.concealed, len(claimer.hand.melds)) == -1
            claimer.hand.concealed[tid] -= 1
            if wins:
                self._phase = "finished"
                self.winner = claimer_seat
                events.append(GameEvent(EventType.WIN_DISCARD, seat=claimer_seat, tile=tid,
                                        detail={"shooter": seat}))
                return events

        self._seat_idx = (self._seat_idx + 1) % 4
        return events
