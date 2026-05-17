"""
Multi-hand Singapore Mahjong match manager.

Handles wind rotation, chip persistence, and round tracking across hands.

Wind rotation rules:
  - East wins: no rotation, East deals again.
  - Wall exhausted (draw): no rotation, East deals again.
  - Non-East wins: rotate.
      Dealership passes clockwise; each player's wind label steps E→N→W→S→E:
      South → East (new dealer)
      East  → North
      North → West
      West  → South
  - After 4 rotations in a table-wind round (all 4 players served as East):
    table wind advances East → South → West → North.
  - Match ends after n_rounds table-wind rounds (default 4 = East + South + West + North).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cracked.tiles import Wind
from cracked.engine import GameEngine, GameEvent
from cracked.scoring import STARTING_CHIPS

_E = int(Wind.EAST)
_S = int(Wind.SOUTH)
_W = int(Wind.WEST)
_N = int(Wind.NORTH)

WIND_NAMES = {_E: "East", _S: "South", _W: "West", _N: "North"}
_WIND_CYCLE = [_E, _S, _W, _N]

# After a rotation, each player's wind steps E→N→W→S→E (dealer passes clockwise).
_NEXT_WIND = {_E: _N, _S: _E, _W: _S, _N: _W}


@dataclass
class HandResult:
    hand_number: int
    table_wind: int
    winner_wind: Optional[int]   # None = draw (wall exhausted)
    rotated: bool
    chips_after: dict[int, int]  # wind-constant → chip count after this hand


class GameMatch:
    """
    Manages a full Singapore Mahjong game: multiple hands, wind rotation,
    chip persistence.

    Usage:
        match = GameMatch(n_rounds=2, human_initial_wind=Wind.EAST)
        events = match.start_hand()
        # ... drive match.engine via step() / submit_discard() ...
        result = match.finish_hand()   # call when engine.is_finished
        if not match.is_complete:
            events = match.start_hand()  # next hand
    """

    def __init__(
        self,
        n_rounds: int = 4,
        human_initial_wind: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_rounds = n_rounds
        self._seed = seed

        # Chips keyed by current wind constant.
        # After a rotation the chip balance moves with the physical player.
        self.chips: dict[int, int] = {
            _E: STARTING_CHIPS, _S: STARTING_CHIPS,
            _W: STARTING_CHIPS, _N: STARTING_CHIPS,
        }

        # player_at[wind] = player number (1–4); rotates with winds so P1 is
        # always the same physical person regardless of their current wind label.
        self.player_at: dict[int, int] = {_E: 1, _S: 2, _W: 3, _N: 4}

        self.table_wind: int = _E
        self._rotations_in_round: int = 0
        self._total_hands: int = 0
        self._rounds_completed: int = 0

        # Human player's current wind label (changes on each rotation).
        self._human_wind: Optional[int] = (
            int(human_initial_wind) if human_initial_wind is not None else None
        )

        self.engine: Optional[GameEngine] = None
        self.is_complete: bool = False
        self.history: list[HandResult] = []

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def hand_number(self) -> int:
        """1-based hand index for the current (or next) hand."""
        return self._total_hands + 1

    @property
    def human_wind(self) -> Optional[int]:
        """Current wind constant for the human player, or None (all AI)."""
        return self._human_wind

    @property
    def round_label(self) -> str:
        """e.g. 'East Round' or 'South Round'."""
        return f"{WIND_NAMES[self.table_wind]} Round"

    # ------------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------------

    def start_hand(self) -> list[GameEvent]:
        """Create a fresh GameEngine for the current hand and deal tiles."""
        human_seats = {self._human_wind} if self._human_wind is not None else None
        seed = (self._seed + self._total_hands) if self._seed is not None else None
        self.engine = GameEngine(
            human_seats=human_seats,
            prevailing_wind=self.table_wind,
            seed=seed,
        )
        events = self.engine.deal()
        # Inject persistent chip balances after deal() to override its reset.
        self.engine.chips = dict(self.chips)
        return events

    def finish_hand(self) -> HandResult:
        """
        Sync chips from the engine, apply wind rotation if needed, and
        advance match state. Call this once engine.is_finished is True.
        """
        assert self.engine is not None, "No active hand"

        # Pull final chip state from engine.
        self.chips = dict(self.engine.chips)
        winner = self.engine.winner  # Wind constant of winner, or None

        kong_declared = self.engine.kong_declared
        rotated = (winner is not None and winner != _E) or (winner is None and kong_declared)

        self._total_hands += 1

        result = HandResult(
            hand_number=self._total_hands,
            table_wind=self.table_wind,
            winner_wind=winner,
            rotated=rotated,
            chips_after=dict(self.chips),
        )
        self.history.append(result)

        if rotated:
            self._do_rotate()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_rotate(self) -> None:
        """Dealer seat passes clockwise: South→East, East→North, North→West, West→South."""
        self.chips = {
            _E: self.chips[_S],
            _N: self.chips[_E],
            _W: self.chips[_N],
            _S: self.chips[_W],
        }
        self.player_at = {
            _E: self.player_at[_S],
            _N: self.player_at[_E],
            _W: self.player_at[_N],
            _S: self.player_at[_W],
        }
        if self._human_wind is not None:
            self._human_wind = _NEXT_WIND[self._human_wind]

        self._rotations_in_round += 1
        if self._rotations_in_round >= 4:
            self._rotations_in_round = 0
            self._advance_table_wind()

    def _advance_table_wind(self) -> None:
        self._rounds_completed += 1
        if self._rounds_completed >= self.n_rounds:
            self.is_complete = True
        else:
            idx = _WIND_CYCLE.index(self.table_wind)
            self.table_wind = _WIND_CYCLE[(idx + 1) % len(_WIND_CYCLE)]
