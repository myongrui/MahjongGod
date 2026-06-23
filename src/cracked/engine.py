"""Turn-by-turn game engine for Singapore Mahjong.

Synchronous state machine owning the full ground truth of a game (all 4 hands,
wall, discard piles). Drives both spectator and interactive TUI modes.

Claim mechanics:
  - Pong/Kong: any player (except discarder), clockwise priority.
  - Chow: left player only (discarder + 1 in turn order).
  - Discard win always takes priority over all claims.
  - Human players skip claim opportunities (no UI yet).

Bonus tiles (flowers 34-37, seasons 38-41, animals 42-45) are included in the
wall. When drawn, they are set aside and a replacement tile is drawn immediately.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from cracked.tiles import NTILES, Wind, tile_name, is_bonus_tile, is_animal
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView
from cracked.tiles_away import tiles_away
from cracked.scoring import calculate_tai, WinContext, chip_payment, STARTING_CHIPS
from cracked.optimizer import recommend_discard, DiscardRecommendation
from cracked.policy import Policy, HumanPolicy, HeuristicPolicy

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
    AWAIT_CLAIM     = "await_claim"    # human may pong/kong/chow this discard


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
        policies: Optional[dict[int, Policy]] = None,
    ) -> None:
        self._human_seats: set[int] = human_seats or set()
        self.prevailing_wind = prevailing_wind
        self._rng = random.Random(seed)

        # One policy per seat. Explicit policies win; otherwise human seats get a
        # HumanPolicy (await external input) and the rest the risk-aware bot.
        explicit = policies or {}
        self.policies: dict[int, Policy] = {}
        for seat in _WIND_ORDER:
            if seat in explicit:
                self.policies[seat] = explicit[seat]
            elif seat in self._human_seats:
                self.policies[seat] = HumanPolicy()
            else:
                self.policies[seat] = HeuristicPolicy()

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
        self._awaiting_claim: bool = False
        self._pending_claim: Optional[dict] = None

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

    @property
    def awaiting_human_claim(self) -> bool:
        return self._awaiting_claim

    @property
    def pending_claim_kinds(self) -> Optional[dict]:
        """When awaiting a human claim, the available kinds: {'pong': True, 'kong': True,
        'chow': [(t1,t2,t3), ...]} (only the keys that apply)."""
        if not self._awaiting_claim or self._pending_claim is None:
            return None
        return self._pending_claim["dps"][self._pending_claim["index"]][1]

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
            self.players[seat] = PlayerState(seat=seat, hand=hand)

        self.chips = {seat: STARTING_CHIPS for seat in _WIND_ORDER}
        self._phase = "playing"
        self._seat_idx = 0
        self.turn_number = 0
        self._awaiting_discard = False
        self._awaiting_claim = False
        self._pending_claim = None
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

        if tiles_away(player.hand.concealed, len(player.hand.melds)) == -1:
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

        choice = self.policies[seat].choose_discard(self.player_view_for(seat))
        if choice is None:
            self._awaiting_discard = True
            events.append(GameEvent(EventType.AWAIT_DISCARD, seat=seat, tile=drawn))
            return events

        events.extend(self._execute_discard(seat, forced=choice))
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
        """Remove a tile from seat's hand, check for a discard win, then check for claims."""
        player = self.players[seat]
        events: list[GameEvent] = []

        if forced is not None:
            tid = forced
        else:
            # Reached after a claim. AI seats return a tile; a human seat returns None,
            # in which case we pause for their discard (e.g. after the human pongs/chows).
            choice = self.policies[seat].choose_discard(self.player_view_for(seat))
            if choice is None:
                self._awaiting_discard = True
                return [GameEvent(EventType.AWAIT_DISCARD, seat=seat)]
            tid = choice

        player.hand.remove_tile(tid)
        player.discards.append(tid)
        events.append(GameEvent(EventType.DISCARD, seat=seat, tile=tid))

        # Discard-win check
        for claimer_seat in _WIND_ORDER:
            if claimer_seat == seat:
                continue
            claimer = self.players[claimer_seat]
            claimer.hand.concealed[tid] += 1
            tai_result = None
            if tiles_away(claimer.hand.concealed, len(claimer.hand.melds)) == -1:
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

        # Pong / kong / chow claims (may pause for a human claimer)
        events.extend(self._resolve_claims(tid, seat))
        return events

    def _claim_decision_points(self, tid: int, discarder_seat: int) -> list:
        """Ordered claim opportunities in priority order: pong/kong (clockwise from the
        discarder) before chow (left player only). Each point is [seat, kinds], where kinds
        holds the claim types that seat may take here."""
        di = _WIND_ORDER.index(discarder_seat)
        dps = []
        for offset in range(1, 4):
            seat = _WIND_ORDER[(di + offset) % 4]
            count = int(self.players[seat].hand.concealed[tid])
            kinds: dict = {}
            if count >= 3:
                kinds["kong"] = True
            if count >= 2:
                kinds["pong"] = True
            if kinds:
                dps.append([seat, kinds])
        left_seat = _WIND_ORDER[(di + 1) % 4]
        options = self._find_chow_options(self.players[left_seat].hand, tid)
        if options:
            dps.append([left_seat, {"chow": options}])
        return dps

    def _resolve_claims(self, tid: int, discarder_seat: int) -> list[GameEvent]:
        """Begin claim resolution for a discard. Returns claim events, an AWAIT_CLAIM pause
        (human's turn to decide), or [] after advancing the turn when nobody claims."""
        self._pending_claim = {"tid": tid, "discarder": discarder_seat,
                               "dps": self._claim_decision_points(tid, discarder_seat),
                               "index": 0}
        return self._advance_claims()

    def _advance_claims(self) -> list[GameEvent]:
        """Walk the pending claim opportunities in priority order. AI seats decide
        immediately; a human seat pauses with an AWAIT_CLAIM event. When all decline, the
        turn passes to the next player."""
        pc = self._pending_claim
        tid, discarder, dps = pc["tid"], pc["discarder"], pc["dps"]
        i = pc["index"]
        while i < len(dps):
            seat, kinds = dps[i]
            if seat in self._human_seats:
                pc["index"] = i
                self._awaiting_claim = True
                return [GameEvent(EventType.AWAIT_CLAIM, seat=seat, tile=tid,
                                  detail={"kinds": kinds})]
            view = self.player_view_for(seat)
            policy = self.policies[seat]
            if "kong" in kinds and policy.wants_kong(view, tid):
                return self._claim_done(self._do_kong(seat, discarder, tid))
            if "pong" in kinds and policy.wants_pong(view, tid):
                return self._claim_done(self._do_pong(seat, discarder, tid))
            if "chow" in kinds:
                chow = policy.choose_chow(view, tid, kinds["chow"])
                if chow:
                    return self._claim_done(self._do_chow(seat, discarder, tid, chow))
            i += 1
        # nobody claimed → next player's turn
        self._pending_claim = None
        self._awaiting_claim = False
        self._seat_idx = (self._seat_idx + 1) % 4
        return []

    def _claim_done(self, events: list[GameEvent]) -> list[GameEvent]:
        self._pending_claim = None
        self._awaiting_claim = False
        return events

    def submit_claim(self, kind: str, chow: Optional[tuple] = None) -> list[GameEvent]:
        """Human takes the offered claim: kind in {'pong','kong','chow'} (chow needs the
        chosen (t1,t2,t3) tuple). Returns the resulting events (then AWAIT_DISCARD)."""
        if not self._awaiting_claim:
            raise ValueError("Not awaiting a human claim")
        pc = self._pending_claim
        seat = pc["dps"][pc["index"]][0]
        tid, discarder = pc["tid"], pc["discarder"]
        self._pending_claim = None
        self._awaiting_claim = False
        if kind == "kong":
            return self._do_kong(seat, discarder, tid)
        if kind == "pong":
            return self._do_pong(seat, discarder, tid)
        if kind == "chow":
            return self._do_chow(seat, discarder, tid, tuple(chow))
        raise ValueError(f"unknown claim kind: {kind}")

    def pass_claim(self) -> list[GameEvent]:
        """Human declines the offered claim; resume with lower-priority claimers."""
        if not self._awaiting_claim:
            raise ValueError("Not awaiting a human claim")
        self._awaiting_claim = False
        self._pending_claim["index"] += 1
        return self._advance_claims()

    # ------------------------------------------------------------------
    # Internal: claim option enumeration (decisions live in seat policies)
    # ------------------------------------------------------------------

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

        if tiles_away(claimer.hand.concealed, len(claimer.hand.melds)) == -1:
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
