"""Turn-by-turn game engine for Singapore Mahjong.

Synchronous state machine owning the full ground truth of a game (all 4 hands,
wall, discard piles). Drives both spectator and interactive TUI modes.

Claim mechanics:
  - Pong/Kong: any player (except discarder), clockwise priority.
  - Chow: left player only (discarder + 1 in turn order).
  - Ron always takes priority over all claims.
  - Human players skip claim opportunities (no UI yet).

Bonus tiles (flowers 34-37, seasons 38-41, animals 42-45) are included in the
wall. When drawn, they are set aside and a replacement tile is drawn immediately.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from cracked.tiles import NTILES, Wind, tile_name, is_bonus_tile, is_animal
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView
from cracked.shanten import shanten
from cracked.scoring import calculate_tai, WinContext, chip_payment, STARTING_CHIPS
from cracked.simulator import SimHand, _heuristic_discard
from cracked.optimizer import recommend_discard, DiscardRecommendation

_WIND_ORDER: list[int] = [int(Wind.EAST), int(Wind.SOUTH), int(Wind.WEST), int(Wind.NORTH)]
_BONUS_TILES: list[int] = list(range(34, 46))   # one of each (12 total)


class EventType(Enum):
    DEAL            = "deal"
    DRAW            = "draw"
    BONUS           = "bonus"          # bonus tile set aside; replacement auto-drawn
    DISCARD         = "discard"
    MELD            = "meld"           # pong / chow / kong from a discard
    WIN_SELF_DRAW   = "win_self_draw"
    WIN_DISCARD     = "win_discard"
    WALL_EXHAUSTED  = "wall_exhausted"
    AWAIT_DISCARD   = "await_discard"


@dataclass
class GameEvent:
    type:   EventType
    seat:   int               # Wind constant of acting player (-1 = no player)
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
        self.chips: dict[int, int] = {}
        self._wall: list[int] = []
        self._wall_idx: int = 0
        self.turn_number: int = 0
        self._seat_idx: int = 0
        self._phase: str = "not_started"
        self.winner: Optional[int] = None
        self.kong_declared: bool = False
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
        """Shuffle wall (136 standard + 12 bonus tiles) and deal 13 to each player."""
        flat: list[int] = [tid for tid in range(NTILES) for _ in range(4)] + _BONUS_TILES
        self._rng.shuffle(flat)
        self._wall = flat
        self._wall_idx = 0

        self.players = {}
        for seat in _WIND_ORDER:
            hand = HandState(seat_wind=seat)
            dealt = 0
            while dealt < 13 and self._wall_idx < len(self._wall):
                tid = self._wall[self._wall_idx]
                self._wall_idx += 1
                if is_bonus_tile(tid):
                    if is_animal(tid):
                        hand.animals.append(tid)
                    else:
                        hand.flowers.append(tid)
                else:
                    hand.add_tile(tid)
                    dealt += 1
            self.players[seat] = PlayerState(
                seat=seat, hand=hand, is_human=(seat in self._human_seats)
            )

        self.chips = {seat: STARTING_CHIPS for seat in _WIND_ORDER}
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

        - AI players:   returns [BONUS*] + [DRAW] + [DISCARD] (+ any claim chain)
        - Human player: returns [BONUS*] + [DRAW, AWAIT_DISCARD]
        - Wall empty:   returns [WALL_EXHAUSTED]
        - Game finished: returns []
        """
        if self._phase == "finished":
            return []
        if self._awaiting_discard:
            return [GameEvent(EventType.AWAIT_DISCARD, seat=self.current_seat)]

        seat = self.current_seat
        player = self.players[seat]
        events: list[GameEvent] = []

        draw_events, exhausted = self._draw_tile(seat)
        events.extend(draw_events)
        if exhausted:
            self._phase = "finished"
            return events

        drawn = next(e.tile for e in reversed(draw_events) if e.type == EventType.DRAW)

        if shanten(player.hand.concealed, len(player.hand.melds)) == -1:
            ctx = WinContext(winning_tile=drawn, is_self_draw=True,
                             is_last_tile=self.wall_remaining <= 15)
            tai_result = calculate_tai(player.hand, ctx)
            if tai_result.is_valid_win():
                _, zimo_pay = chip_payment(tai_result.total)
                for other in _WIND_ORDER:
                    if other != seat:
                        self.chips[other] -= zimo_pay
                        self.chips[seat] += zimo_pay
                self._phase = "finished"
                self.winner = seat
                events.append(GameEvent(EventType.WIN_SELF_DRAW, seat=seat, tile=drawn,
                                        detail={"tai": tai_result.total, "zimo_pay": zimo_pay}))
                return events

        if player.is_human:
            self._awaiting_discard = True
            events.append(GameEvent(EventType.AWAIT_DISCARD, seat=seat, tile=drawn))
            return events

        events.extend(self._execute_discard(seat))
        return events

    def submit_discard(self, tile_id: int) -> list[GameEvent]:
        """Human player submits their discard choice. Returns events including any AI claims."""
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
    # Internal: drawing
    # ------------------------------------------------------------------

    def _draw_tile(self, seat: int) -> tuple[list[GameEvent], bool]:
        """Draw tile(s) for seat, handling bonus tiles with automatic replacement.

        Returns (events, wall_exhausted). Bonus events precede the DRAW event.
        """
        events: list[GameEvent] = []
        player = self.players[seat]

        while self.wall_remaining > 15:
            tid = self._wall[self._wall_idx]
            self._wall_idx += 1

            if is_bonus_tile(tid):
                if is_animal(tid):
                    player.hand.animals.append(tid)
                else:
                    player.hand.flowers.append(tid)
                events.append(GameEvent(EventType.BONUS, seat=seat, tile=tid,
                                        detail={"wall_remaining": self.wall_remaining}))
                continue  # draw replacement from live wall only

            player.hand.add_tile(tid)
            self.turn_number += 1
            events.append(GameEvent(EventType.DRAW, seat=seat, tile=tid,
                                    detail={"wall_remaining": self.wall_remaining}))
            return events, False

        events.append(GameEvent(EventType.WALL_EXHAUSTED, seat=-1))
        return events, True

    # ------------------------------------------------------------------
    # Internal: discarding and claim chain
    # ------------------------------------------------------------------

    def _execute_discard(self, seat: int, forced: Optional[int] = None) -> list[GameEvent]:
        """Remove a tile from seat's hand, check for ron, then check for claims."""
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

        # Ron check
        for claimer_seat in _WIND_ORDER:
            if claimer_seat == seat:
                continue
            claimer = self.players[claimer_seat]
            claimer.hand.concealed[tid] += 1
            tai_result = None
            if shanten(claimer.hand.concealed, len(claimer.hand.melds)) == -1:
                ctx = WinContext(winning_tile=tid, is_self_draw=False,
                                 is_last_tile=self.wall_remaining <= 15)
                tai_result = calculate_tai(claimer.hand, ctx)
            claimer.hand.concealed[tid] -= 1
            if tai_result is not None and tai_result.is_valid_win():
                shooter_pay, _ = chip_payment(tai_result.total)
                self.chips[seat] -= shooter_pay
                self.chips[claimer_seat] += shooter_pay
                self._phase = "finished"
                self.winner = claimer_seat
                events.append(GameEvent(EventType.WIN_DISCARD, seat=claimer_seat, tile=tid,
                                        detail={"shooter": seat, "tai": tai_result.total,
                                                "shooter_pay": shooter_pay}))
                return events

        # Pong / kong / chow claims
        claim_events = self._check_claims(tid, seat)
        if claim_events:
            events.extend(claim_events)
            return events

        self._seat_idx = (self._seat_idx + 1) % 4
        return events

    def _check_claims(self, discard_tid: int, discarder_seat: int) -> list[GameEvent]:
        """Return claim events if any AI player wants to pong/kong/chow, else []."""
        discarder_idx = _WIND_ORDER.index(discarder_seat)

        # Pong / Kong: clockwise priority from discarder
        for offset in range(1, 4):
            claimer_seat = _WIND_ORDER[(discarder_idx + offset) % 4]
            claimer = self.players[claimer_seat]
            if claimer.is_human:
                continue
            count = int(claimer.hand.concealed[discard_tid])
            if count >= 3 and self._ai_wants_kong(claimer_seat, discard_tid):
                return self._do_kong(claimer_seat, discarder_seat, discard_tid)
            if count >= 2 and self._ai_wants_pong(claimer_seat, discard_tid):
                return self._do_pong(claimer_seat, discarder_seat, discard_tid)

        # Chow: left player only
        left_seat = _WIND_ORDER[(discarder_idx + 1) % 4]
        if not self.players[left_seat].is_human:
            chow = self._pick_best_chow(left_seat, discard_tid)
            if chow:
                return self._do_chow(left_seat, discarder_seat, discard_tid, chow)

        return []

    # ------------------------------------------------------------------
    # Internal: AI claim decisions
    # ------------------------------------------------------------------

    def _ai_wants_pong(self, seat: int, tile: int) -> bool:
        """True if ponging and then making the best discard maintains or improves shanten."""
        player = self.players[seat]
        current_s = shanten(player.hand.concealed, len(player.hand.melds))
        if current_s == -1:
            return False
        test = player.hand.concealed.copy()
        test[tile] -= 2
        test_melds = len(player.hand.melds) + 1
        best_s = min(
            (shanten(test - np.eye(NTILES, dtype=np.int8)[t], test_melds)
             for t in range(NTILES) if test[t] > 0),
            default=current_s,
        )
        return best_s <= current_s

    def _ai_wants_kong(self, seat: int, tile: int) -> bool:
        """True if the hand structure is not significantly worsened by konging
        (replacement draw compensates for one extra step)."""
        player = self.players[seat]
        current_s = shanten(player.hand.concealed, len(player.hand.melds))
        test = player.hand.concealed.copy()
        test[tile] -= 3
        test_melds = len(player.hand.melds) + 1
        best_s = min(
            (shanten(test - np.eye(NTILES, dtype=np.int8)[t], test_melds)
             for t in range(NTILES) if test[t] > 0),
            default=current_s,
        )
        return best_s <= current_s + 1

    def _find_chow_options(self, hand: HandState, tile: int) -> list[tuple[int, int, int]]:
        """All valid chow combinations for claiming the given suited tile."""
        if tile >= 27:
            return []
        suit_start = (tile // 9) * 9
        rank = tile % 9
        options = []
        for low in (rank - 2, rank - 1, rank):
            if low < 0 or low + 2 > 8:
                continue
            t1, t2, t3 = suit_start + low, suit_start + low + 1, suit_start + low + 2
            if all(hand.concealed[t] > 0 for t in (t1, t2, t3) if t != tile):
                options.append((t1, t2, t3))
        return options

    def _pick_best_chow(self, seat: int, tile: int) -> Optional[tuple[int, int, int]]:
        """Return the chow option that strictly improves shanten, or None."""
        player = self.players[seat]
        options = self._find_chow_options(player.hand, tile)
        if not options:
            return None
        current_s = shanten(player.hand.concealed, len(player.hand.melds))
        best_option: Optional[tuple[int, int, int]] = None
        best_s = current_s  # chow requires strict improvement
        eye = np.eye(NTILES, dtype=np.int8)
        for chow_tiles in options:
            test = player.hand.concealed.copy()
            for t in chow_tiles:
                if t != tile:
                    test[t] -= 1
            test_melds = len(player.hand.melds) + 1
            min_s = min(
                (shanten(test - eye[t], test_melds) for t in range(NTILES) if test[t] > 0),
                default=current_s,
            )
            if min_s < best_s:
                best_s = min_s
                best_option = chow_tiles
        return best_option

    # ------------------------------------------------------------------
    # Internal: executing claims
    # ------------------------------------------------------------------

    def _do_pong(self, claimer_seat: int, discarder_seat: int, tile: int) -> list[GameEvent]:
        claimer = self.players[claimer_seat]
        claimer.hand.concealed[tile] -= 2
        claimer.hand.melds.append(
            Meld(MeldType.PONG, (tile, tile, tile), concealed=False,
                 source_player=discarder_seat)
        )
        self._seat_idx = _WIND_ORDER.index(claimer_seat)
        events = [GameEvent(EventType.MELD, seat=claimer_seat, tile=tile,
                            detail={"meld_type": "pong", "from": discarder_seat,
                                    "tiles": [tile, tile, tile]})]
        events.extend(self._execute_discard(claimer_seat))
        return events

    def _do_kong(self, claimer_seat: int, discarder_seat: int, tile: int) -> list[GameEvent]:
        self.kong_declared = True
        claimer = self.players[claimer_seat]
        claimer.hand.concealed[tile] -= 3
        claimer.hand.melds.append(
            Meld(MeldType.KONG, (tile, tile, tile, tile), concealed=False,
                 source_player=discarder_seat)
        )
        self._seat_idx = _WIND_ORDER.index(claimer_seat)
        events = [GameEvent(EventType.MELD, seat=claimer_seat, tile=tile,
                            detail={"meld_type": "kong", "from": discarder_seat,
                                    "tiles": [tile, tile, tile, tile]})]

        draw_events, exhausted = self._draw_tile(claimer_seat)
        events.extend(draw_events)
        if exhausted:
            self._phase = "finished"
            return events

        if shanten(claimer.hand.concealed, len(claimer.hand.melds)) == -1:
            drawn = next(e.tile for e in reversed(draw_events) if e.type == EventType.DRAW)
            ctx = WinContext(winning_tile=drawn, is_self_draw=True, is_replacement=True,
                             is_last_tile=self.wall_remaining <= 15)
            tai_result = calculate_tai(claimer.hand, ctx)
            if tai_result.is_valid_win():
                _, zimo_pay = chip_payment(tai_result.total)
                for other in _WIND_ORDER:
                    if other != claimer_seat:
                        self.chips[other] -= zimo_pay
                        self.chips[claimer_seat] += zimo_pay
                self._phase = "finished"
                self.winner = claimer_seat
                events.append(GameEvent(EventType.WIN_SELF_DRAW, seat=claimer_seat, tile=drawn,
                                        detail={"tai": tai_result.total, "zimo_pay": zimo_pay}))
                return events

        events.extend(self._execute_discard(claimer_seat))
        return events

    def _do_chow(self, claimer_seat: int, discarder_seat: int, tile: int,
                 chow_tiles: tuple[int, int, int]) -> list[GameEvent]:
        claimer = self.players[claimer_seat]
        for t in chow_tiles:
            if t != tile:
                claimer.hand.concealed[t] -= 1
        claimer.hand.melds.append(
            Meld(MeldType.CHOW, chow_tiles, concealed=False, source_player=discarder_seat)
        )
        self._seat_idx = _WIND_ORDER.index(claimer_seat)
        events = [GameEvent(EventType.MELD, seat=claimer_seat, tile=tile,
                            detail={"meld_type": "chow", "from": discarder_seat,
                                    "tiles": list(chow_tiles)})]
        events.extend(self._execute_discard(claimer_seat))
        return events
