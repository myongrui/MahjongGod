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
from cracked.scoring import calculate_tai, WinContext, chip_payment, STARTING_CHIPS, DEFAULT_RULES
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
    AWAIT_WIN       = "await_win"      # human may declare a win (hu) or decline


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
        self._awaiting_win: bool = False
        self._pending_win: Optional[dict] = None
        # Sacred/missed-discard prohibition: tiles each seat may not win/pong from a
        # discard until their next turn (回头牌 / 过水牌). Cleared at the seat's turn.
        self._prohibited: dict[int, set[int]] = {w: set() for w in _WIND_ORDER}

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
    def awaiting_human_win(self) -> bool:
        return self._awaiting_win

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
        self._awaiting_win = False
        self._pending_win = None
        self._prohibited = {w: set() for w in _WIND_ORDER}
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
        if self._awaiting_win or self._awaiting_claim:
            return []                       # blocked on a pending human hu / claim decision

        seat = self.current_seat
        self._prohibited[seat].clear()      # a new turn lifts this seat's sacred/missed bans
        player = self.players[seat]
        events: list[GameEvent] = []

        draw_events, exhausted = self._draw_tile(seat)
        events.extend(draw_events)
        if exhausted:
            self._phase = "finished"
            return events
        if self.is_finished:                      # eight-flower instant win during draw
            return events

        drawn = next(e.tile for e in reversed(draw_events) if e.type == EventType.DRAW)

        if tiles_away(player.hand.concealed, len(player.hand.melds)) == -1:
            ctx = self._self_draw_ctx(seat, drawn)
            if calculate_tai(player.hand, ctx).is_valid_win():
                if seat in self._human_seats:               # let the human choose hu / decline
                    self._awaiting_win = True
                    self._pending_win = {"kind": "self_draw", "seat": seat, "tid": drawn}
                    events.append(GameEvent(EventType.AWAIT_WIN, seat=seat, tile=drawn,
                                            detail={"self_draw": True}))
                    return events
                events.extend(self._declare_self_draw_win(seat, drawn))
                return events

        # Own-turn kongs (concealed / promotion) before discarding.
        events.extend(self._offer_self_kongs(seat))
        if self.is_finished:
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

    def _self_draw_ctx(self, seat: int, drawn: int) -> WinContext:
        is_heavenly = (seat == int(Wind.EAST) and self.turn_number == 1
                       and all(not p.discards for p in self.players.values())
                       and self._no_melds_anywhere())
        return WinContext(winning_tile=drawn, is_self_draw=True,
                          is_last_tile=self.wall_remaining <= 15,
                          is_heavenly=is_heavenly, prevailing_wind=self.prevailing_wind)

    def _declare_self_draw_win(self, seat: int, drawn: int) -> list[GameEvent]:
        """Settle a self-draw win: score, everyone pays, finish the hand."""
        tai_result = calculate_tai(self.players[seat].hand, self._self_draw_ctx(seat, drawn))
        _, zimo_pay = chip_payment(tai_result.total)
        for other in _WIND_ORDER:
            if other != seat:
                self.chips[other] -= zimo_pay
                self.chips[seat] += zimo_pay
        self._phase = "finished"
        self.winner = seat
        return [GameEvent(EventType.WIN_SELF_DRAW, seat=seat, tile=drawn,
                          detail={"tai": tai_result.total, "zimo_pay": zimo_pay})]

    def submit_win(self) -> list[GameEvent]:
        """Human declares the offered win (hu)."""
        if not self._awaiting_win:
            raise ValueError("Not awaiting a human win")
        pw = self._pending_win
        self._awaiting_win = False
        self._pending_win = None
        if pw["kind"] == "ron":
            return self._declare_discard_win(pw["seat"], pw["discarder"], pw["tid"])
        return self._declare_self_draw_win(pw["seat"], pw["tid"])

    def decline_win(self) -> list[GameEvent]:
        """Human declines the offered win and play continues."""
        if not self._awaiting_win:
            raise ValueError("Not awaiting a human win")
        pw = self._pending_win
        self._awaiting_win = False
        self._pending_win = None
        if pw["kind"] == "ron":
            self._prohibited[pw["seat"]].add(pw["tid"])   # 过水牌: passed up a winning tile
            # Check lower-priority ron winners, then fall through to claims.
            win_events, stop = self._resolve_discard_win(pw["tid"], pw["discarder"],
                                                         start_offset=pw["offset"] + 1)
            if stop:
                return win_events
            return win_events + self._resolve_claims(pw["tid"], pw["discarder"])
        # Declined a self-draw: continue the turn (kongs, then discard).
        seat = pw["seat"]
        events = self._offer_self_kongs(seat)
        if self.is_finished:
            return events
        choice = self.policies[seat].choose_discard(self.player_view_for(seat))
        if choice is None:
            self._awaiting_discard = True
            events.append(GameEvent(EventType.AWAIT_DISCARD, seat=seat))
            return events
        events.extend(self._execute_discard(seat, forced=choice))
        return events

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
    # Internal: timing-hand detection (天胡 / 地胡 / 人胡)
    # ------------------------------------------------------------------

    def _no_melds_anywhere(self) -> bool:
        return all(not p.hand.melds for p in self.players.values())

    def _eight_flower_win(self, seat: int) -> GameEvent:
        """Resolve a 花胡 (eight-flower) instant limit win as a self-draw."""
        _, zimo_pay = chip_payment(DEFAULT_RULES.tai_cap)
        for other in _WIND_ORDER:
            if other != seat:
                self.chips[other] -= zimo_pay
                self.chips[seat] += zimo_pay
        self._phase = "finished"
        self.winner = seat
        return GameEvent(EventType.WIN_SELF_DRAW, seat=seat, tile=-1,
                         detail={"tai": DEFAULT_RULES.tai_cap, "zimo_pay": zimo_pay,
                                 "eight_flowers": True})

    def _discard_win_timing(self, discarder: int, claimer: int) -> tuple[bool, bool]:
        """(is_earthly, is_humanly) for a discard win, per the first-round rules."""
        if not self._no_melds_anywhere():
            return False, False
        total_discards = sum(len(p.discards) for p in self.players.values())
        east = int(Wind.EAST)
        # Earthly: a non-dealer wins on the dealer's very first discard.
        is_earthly = (discarder == east and total_discards == 1 and self.turn_number == 1)
        # Humanly: a non-dealer wins on a first-round discard before their own
        # first draw (approximated by: claimer has not discarded yet, round 1).
        is_humanly = (not is_earthly and claimer != east
                      and not self.players[claimer].discards
                      and self.turn_number <= 4)
        return is_earthly, is_humanly

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
                # Two complete flower groups (花胡): 8 flowers/seasons = instant limit win.
                if not is_animal(tid) and len(player.hand.flowers) == 8:
                    events.append(self._eight_flower_win(seat))
                    return events, False
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
        self._prohibited[seat].add(tid)     # 回头牌: can't win/pong this until your next turn
        events.append(GameEvent(EventType.DISCARD, seat=seat, tile=tid))

        # Discard-win check (head-bump priority). Pauses if a human may declare hu.
        win_events, stop = self._resolve_discard_win(tid, seat)
        events.extend(win_events)
        if stop:
            return events

        # Pong / kong / chow claims (may pause for a human claimer)
        events.extend(self._resolve_claims(tid, seat))
        return events

    def _can_ron(self, claimer_seat: int, discarder_seat: int, tid: int) -> bool:
        """True if claimer_seat has a valid winning hand on the discard tid."""
        if tid in self._prohibited[claimer_seat]:    # 回头牌 / 过水牌: barred this go-around
            return False
        claimer = self.players[claimer_seat]
        claimer.hand.concealed[tid] += 1
        valid = False
        if tiles_away(claimer.hand.concealed, len(claimer.hand.melds)) == -1:
            is_earthly, is_humanly = self._discard_win_timing(discarder_seat, claimer_seat)
            ctx = WinContext(winning_tile=tid, is_self_draw=False,
                             is_last_tile=self.wall_remaining <= 15,
                             is_earthly=is_earthly, is_humanly=is_humanly,
                             prevailing_wind=self.prevailing_wind)
            valid = calculate_tai(claimer.hand, ctx).is_valid_win()
        claimer.hand.concealed[tid] -= 1
        return valid

    def _declare_discard_win(self, claimer_seat: int, discarder_seat: int, tid: int) -> list[GameEvent]:
        """Settle a ron: score, pay the shooter, finish the hand."""
        claimer = self.players[claimer_seat]
        claimer.hand.concealed[tid] += 1
        is_earthly, is_humanly = self._discard_win_timing(discarder_seat, claimer_seat)
        ctx = WinContext(winning_tile=tid, is_self_draw=False,
                         is_last_tile=self.wall_remaining <= 15,
                         is_earthly=is_earthly, is_humanly=is_humanly,
                         prevailing_wind=self.prevailing_wind)
        tai_result = calculate_tai(claimer.hand, ctx)
        claimer.hand.concealed[tid] -= 1
        shooter_pay, _ = chip_payment(tai_result.total)
        self.chips[discarder_seat] -= shooter_pay
        self.chips[claimer_seat] += shooter_pay
        self._phase = "finished"
        self.winner = claimer_seat
        return [GameEvent(EventType.WIN_DISCARD, seat=claimer_seat, tile=tid,
                          detail={"shooter": discarder_seat, "tai": tai_result.total,
                                  "shooter_pay": shooter_pay})]

    def _resolve_discard_win(self, tid: int, discarder_seat: int,
                             start_offset: int = 1) -> tuple[list[GameEvent], bool]:
        """Walk potential ron winners in head-bump order. Returns (events, stop):
        stop=True when a win is declared OR we pause for a human's hu decision."""
        di = _WIND_ORDER.index(discarder_seat)
        for offset in range(start_offset, 4):
            claimer_seat = _WIND_ORDER[(di + offset) % 4]
            if not self._can_ron(claimer_seat, discarder_seat, tid):
                continue
            if claimer_seat in self._human_seats:
                self._awaiting_win = True
                self._pending_win = {"kind": "ron", "tid": tid, "seat": claimer_seat,
                                     "discarder": discarder_seat, "offset": offset}
                return [GameEvent(EventType.AWAIT_WIN, seat=claimer_seat, tile=tid,
                                  detail={"from": discarder_seat})], True
            return self._declare_discard_win(claimer_seat, discarder_seat, tid), True
        return [], False

    def _claim_decision_points(self, tid: int, discarder_seat: int) -> list:
        """Ordered claim opportunities in priority order: pong/kong (clockwise from the
        discarder) before chow (left player only). Each point is [seat, kinds], where kinds
        holds the claim types that seat may take here."""
        di = _WIND_ORDER.index(discarder_seat)
        dps = []
        for offset in range(1, 4):
            seat = _WIND_ORDER[(di + offset) % 4]
            if tid in self._prohibited[seat]:        # 回头牌 / 过水牌: no pong/kong of a barred tile
                continue
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
            if "pong" in kinds or "kong" in kinds:
                self._prohibited[seat].add(tid)      # 过水牌: passed a pong/kong opportunity
            i += 1
        # nobody claimed → next player's turn
        self._pending_claim = None
        self._awaiting_claim = False
        self._seat_idx = (self._seat_idx + 1) % 4
        return []

    def _claim_done(self, events: list[GameEvent]) -> list[GameEvent]:
        # The claim's own discard (inside _do_pong/_kong/_chow) already resolved its
        # state — either advancing the turn or pausing for a NESTED human claim/discard/
        # win. Only finalize the original claim here when nothing new is pending; never
        # clobber a fresh pause (which would orphan the human's prompt and let play run on).
        if not (self._awaiting_claim or self._awaiting_discard or self._awaiting_win):
            self._pending_claim = None
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
        pc = self._pending_claim
        seat, kinds = pc["dps"][pc["index"]]
        if "pong" in kinds or "kong" in kinds:
            self._prohibited[seat].add(pc["tid"])    # 过水牌: passed a pong/kong opportunity
        pc["index"] += 1
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

    def _kong_replacement(self, seat: int) -> tuple[list[GameEvent], bool]:
        """Draw a kong replacement tile and resolve any 嶺上開花 self-draw win.
        Returns (events, hand_over). hand_over is True if the hand ended here."""
        self.kong_declared = True
        events, exhausted = self._draw_tile(seat)
        if exhausted:
            self._phase = "finished"
            return events, True
        if self.is_finished:                      # eight-flower during replacement draw
            return events, True
        player = self.players[seat]
        if tiles_away(player.hand.concealed, len(player.hand.melds)) == -1:
            drawn = next(e.tile for e in reversed(events) if e.type == EventType.DRAW)
            ctx = WinContext(winning_tile=drawn, is_self_draw=True, is_replacement=True,
                             is_last_tile=self.wall_remaining <= 15,
                             prevailing_wind=self.prevailing_wind)
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
                                        detail={"tai": tai_result.total, "zimo_pay": zimo_pay,
                                                "replacement": True}))
                return events, True
        return events, False

    def _do_kong(self, claimer_seat: int, discarder_seat: int, tile: int) -> list[GameEvent]:
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
        replacement, over = self._kong_replacement(claimer_seat)
        events.extend(replacement)
        if over:
            return events
        events.extend(self._offer_self_kongs(claimer_seat))
        if self.is_finished or self._awaiting_discard:
            return events
        events.extend(self._execute_discard(claimer_seat))
        return events

    # ------------------------------------------------------------------
    # Internal: own-turn kongs (concealed 暗杠 / promoted 加杠) + robbing (搶槓)
    # ------------------------------------------------------------------

    def _offer_self_kongs(self, seat: int) -> list[GameEvent]:
        """Let `seat` declare concealed/promoted kong(s) before discarding. Each
        kong draws a replacement (可 嶺上開花); a promotion may be robbed (搶槓)."""
        events: list[GameEvent] = []
        chooser = getattr(self.policies[seat], "choose_self_kong", None)
        while chooser is not None:
            tid = chooser(self.player_view_for(seat))
            if tid is None:
                break
            player = self.players[seat]
            promotion = any(m.type == MeldType.PONG and m.tiles[0] == tid for m in player.hand.melds)
            if promotion:
                robbed = self._check_robbing_kong(seat, tid)
                if robbed is not None:
                    events.extend(robbed)
                    return events                 # hand ended — kong was robbed
                events.extend(self._do_promoted_kong(seat, tid))
            elif player.hand.concealed[tid] >= 4:
                events.extend(self._do_concealed_kong(seat, tid))
            else:
                break
            if self.is_finished:                  # 嶺上開花 on the replacement draw
                return events
        return events

    def _do_concealed_kong(self, seat: int, tid: int) -> list[GameEvent]:
        player = self.players[seat]
        player.hand.concealed[tid] -= 4
        player.hand.melds.append(
            Meld(MeldType.KONG, (tid, tid, tid, tid), concealed=True, source_player=None)
        )
        events = [GameEvent(EventType.MELD, seat=seat, tile=tid,
                            detail={"meld_type": "kong", "concealed": True,
                                    "tiles": [tid, tid, tid, tid]})]
        replacement, _ = self._kong_replacement(seat)
        events.extend(replacement)
        return events

    def _do_promoted_kong(self, seat: int, tid: int) -> list[GameEvent]:
        player = self.players[seat]
        for i, m in enumerate(player.hand.melds):
            if m.type == MeldType.PONG and m.tiles[0] == tid:
                player.hand.melds[i] = Meld(MeldType.KONG, (tid, tid, tid, tid),
                                            concealed=False, source_player=m.source_player)
                break
        player.hand.concealed[tid] -= 1
        events = [GameEvent(EventType.MELD, seat=seat, tile=tid,
                            detail={"meld_type": "kong", "promoted": True,
                                    "tiles": [tid, tid, tid, tid]})]
        replacement, _ = self._kong_replacement(seat)
        events.extend(replacement)
        return events

    def _check_robbing_kong(self, kong_seat: int, tid: int) -> Optional[list[GameEvent]]:
        """If another player can win on the tile being promoted (加杠), they rob it
        (搶槓). Returns the win events (hand finished) or None. Head-bump priority."""
        di = _WIND_ORDER.index(kong_seat)
        for offset in range(1, 4):
            claimer_seat = _WIND_ORDER[(di + offset) % 4]
            claimer = self.players[claimer_seat]
            claimer.hand.concealed[tid] += 1
            tai_result = None
            if tiles_away(claimer.hand.concealed, len(claimer.hand.melds)) == -1:
                ctx = WinContext(winning_tile=tid, is_self_draw=False, is_robbing_kong=True,
                                 is_last_tile=self.wall_remaining <= 15,
                                 prevailing_wind=self.prevailing_wind)
                tai_result = calculate_tai(claimer.hand, ctx)
            claimer.hand.concealed[tid] -= 1
            if tai_result is not None and tai_result.is_valid_win():
                shooter_pay, _ = chip_payment(tai_result.total)
                self.chips[kong_seat] -= shooter_pay
                self.chips[claimer_seat] += shooter_pay
                self._phase = "finished"
                self.winner = claimer_seat
                return [GameEvent(EventType.WIN_DISCARD, seat=claimer_seat, tile=tid,
                                  detail={"shooter": kong_seat, "tai": tai_result.total,
                                          "shooter_pay": shooter_pay, "robbing_kong": True})]
        return None

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
