"""
Ursina 3D game — a playable mahjong table driven by the live engine.

The live counterpart to `ursina_table.py` (the static 3D prototype): it pumps a real
`GameMatch`/`GameEngine` and renders every `GameEvent` in 3D, reusing the prototype's
tile model, lighting, and slide+collision discard physics.

    pip install -e ".[threed]"
    python -m cracked.ursina_game        # Menu → Spectate (watch the AI play)

Phase A (this file) is spectator only: four AI bots, no input. Interactive play (you at
East, click-to-discard, pong/kong/chow/hu prompts) is the next pass and needs an engine
claim-await extension.
"""
from __future__ import annotations

import math
import os
import random
import sys
from collections import deque
from pathlib import Path

try:
    from ursina import (Ursina, Entity, Text, Button, color, camera, Vec3,
                        time, application, window, mouse, BoxCollider)
    _HAVE_URSINA = True
except Exception:                       # ursina not installed → module still imports
    _HAVE_URSINA = False

from cracked.engine import EventType
from cracked.hand import MeldType
from cracked.match import GameMatch
from cracked.tiles import (Wind, tile_name, bonus_tile_name, tiles_from_names,
                           WIND_START, WIND_END, DRAGON_START, DRAGON_END,
                           SEAT_FLOWER, SEAT_SEASON)
from cracked import ursina_table as ut   # reuse: make_tile, setup_room, build_felt, ThrowManager, constants
from cracked import ursina_room as ur     # reuse: setup_cabin (dark wood cabin + retro shader)
from cracked import ursina_audio as ua    # tile draw / throw / collision sounds

WINDS = [int(Wind.EAST), int(Wind.SOUTH), int(Wind.WEST), int(Wind.NORTH)]
_NAME = {int(Wind.EAST): "E", int(Wind.SOUTH): "S", int(Wind.WEST): "W", int(Wind.NORTH): "N"}
_FULL = {int(Wind.EAST): "East", int(Wind.SOUTH): "South",
         int(Wind.WEST): "West", int(Wind.NORTH): "North"}
# MAHJONG_SANDBOX test hand for the human seat: 1-shanten and claim-rich, so every
# claim type can be exercised — RD triplet (kong), GD pair (pong), b3b4 & c6c7
# (chow from the left), and it completes to a win after the two sequences fill.
_SANDBOX_HAND = ["rd", "rd", "rd", "gd", "gd", "b3", "b4", "c6", "c7", "d1", "d2", "d3", "d5"]
# JetBrains Mono is the single UI font. Ursina resolves fonts by name within
# application.fonts_folder, so we point that at this package's fonts/ dir.
_HUD_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
_HUD_FONT = "JetBrainsMono-Regular.ttf"

_BTN_BG = color.rgba(0, 0, 0, 0.6)      # translucent black box for all buttons

BEAT = 0.45                              # seconds per animated event
DRAW_GAP = ut.STEP * 0.6                 # gap between the hand and the just-drawn tile


if _HAVE_URSINA:

    def _use_ui_font():
        """Point Ursina at this package's fonts/ dir and make JetBrains Mono the
        default for every Text and Button (idempotent; safe to call repeatedly)."""
        application.fonts_folder = Path(_HUD_FONT_DIR)
        Text.default_font = _HUD_FONT
        Text.default_monospace_font = _HUD_FONT

    class GameDriver(Entity):
        """Owns a GameMatch and pumps `engine.step()` on a beat, animating each event."""

        def __init__(self, mode: str = "spectator", seed: int = 7):
            super().__init__()
            self.mode = mode
            self._sandbox = bool(os.environ.get("MAHJONG_SANDBOX"))   # rig the human's hand for testing
            if mode == "interactive":
                self.match = GameMatch(n_rounds=1, human_initial_wind=int(Wind.EAST), seed=seed)
            else:
                self.match = GameMatch(n_rounds=1, seed=seed)        # all-AI spectator
            self.pile = ut.ThrowManager(auto=False)                  # discard slinger (no auto-toss)
            self.seats = {w: {"holder": None, "tiles": [], "melds": None} for w in WINDS}
            self._drawn: dict = {}                                   # seat -> just-drawn tile (shown apart)
            self.queue: deque = deque()
            self._beat = 0.0
            self._banner = None
            self._reveal_all = False                             # face-up every hand at hand-end
            self._next_btn = None                                # "Next hand" button at hand-end
            _use_ui_font()                                        # JetBrains Mono everywhere
            edge = window.aspect_ratio / 2                        # UI x-extent depends on window aspect
            self.hud = Text(text="", origin=(-.5, .5), position=(-edge + 0.02, .47), scale=0.9, font=_HUD_FONT)
            self.log = Text(text="", origin=(.5, .5), position=(edge - 0.02, 0.3), scale=0.7, font=_HUD_FONT)
            self._log: deque = deque(maxlen=24)                 # rolling action log (oldest drops off top)
            self.prompt = Text(text="", origin=(0, 0), y=-0.28, scale=1.3, color=color.yellow)
            self._claim_btns: list = []                         # active claim buttons (interactive)
            self._win_btns: list = []                           # active hu/skip buttons (interactive)
            self._claim_phase = None                            # None | "offer" | "select"
            self._claim_kinds = None                            # available claim kinds for the current discard
            self._claim_discard_tid = None
            self._claim_sel: list = []                          # selected hand tile entities
            self._claim_relevant: set = set()                   # hand tids armed for selection
            self._last_discard = None                           # board entity of the most recent discard
            self._pulse = 0.0                                   # drives the discard highlight throb
            self._discarded_once = False                        # hide the discard hint after the first discard
            self._seen_claim_hint = False                       # hide the claim hint after it's been seen once
            self._claim_marker = None                           # colour overlay on the claimable discard
            self._draw_skip = None                              # the "draw" tile that declines a claim
            self._start_hand()

        # -- seat geometry -------------------------------------------------------------
        def _front_wind(self) -> int:
            if self.mode == "interactive" and self.match.human_wind is not None:
                return self.match.human_wind
            return WINDS[0]

        def _screen_index(self, wind: int) -> int:
            return (WINDS.index(wind) - WINDS.index(self._front_wind())) % 4

        # -- hand / meld rendering (rebuilt from engine state) -------------------------
        def _rebuild_hand(self, wind: int):
            s = self.seats[wind]
            for e in s["tiles"]:
                ut.destroy_tree(e)                                   # tiles have child meshes — free the subtree
            s["tiles"] = []
            si = self._screen_index(wind)
            # seats go anti-clockwise so the turn order E→S→W→N runs counter-clockwise on
            # screen (bottom→left→top→right), as in real mahjong — hence -si, not +si
            holder_y = -si * 90 + 180
            if s["holder"] is None:
                s["holder"] = Entity(rotation_y=holder_y)
            else:
                s["holder"].rotation_y = holder_y
            hand = self.match.engine.players[wind].hand
            reveal = self._reveal_all or (wind == self._front_wind())  # all face-up at hand-end
            row = hand.concealed_tiles_list()
            drawn = self._drawn.get(wind)
            if drawn is not None and drawn in row:
                row = list(row)
                row.remove(drawn)                                    # shown apart, not in the row
            else:
                drawn = None
            # the +180 holder mirrors x, so lay the sorted hand reversed → reads low→high L→R
            row = list(reversed(row))
            x0 = -(len(row) - 1) / 2 * ut.STEP
            for i, tid in enumerate(row):
                self._place_hand_tile(s, tid, x0 + i * ut.STEP, reveal)
            if drawn is not None:                                    # set apart on the player's right
                self._place_hand_tile(s, drawn, x0 - ut.STEP - DRAW_GAP, reveal)
            self._rebuild_melds(wind, hand)
            self._refresh_obstacles()

        def _place_hand_tile(self, s, tid, lx, reveal):
            t = ut.make_tile(tid, reveal=reveal, parent=s["holder"],
                             bright=ut._pool(math.hypot(lx, ut.EDGE)))
            t.position = (lx, ut.TH / 2, ut.EDGE)
            t.tid = tid                                          # for click-to-discard
            t.base_y = ut.TH / 2                                 # rest height (hover lifts above this)
            if self._reveal_all:
                # Hand-end reveal: lay the tile flat, printed face up, so the camera
                # reads every seat's hand (standing tiles face their own owner, away
                # from the camera, so opponents would otherwise show only their backs).
                t.rotation_x = -90
                t.y = ut.THROW_REST_Y
            s["tiles"].append(t)

        def _refresh_obstacles(self):
            """Register every standing hand tile and flat meld tile as a static collider so
            sliding discards bump off them too — not only the resting discard pile."""
            boxes = []
            for w in WINDS:
                s = self.seats[w]
                if s["holder"] is None:
                    continue
                yaw = math.radians(s["holder"].rotation_y)
                for t in s["tiles"]:
                    p = t.world_position
                    boxes.append((p.x, p.z, yaw, ut.TW / 2, ut.TD / 2))    # standing footprint
                if s["melds"] is not None:
                    for t in s["melds"].children:
                        p = t.world_position
                        boxes.append((p.x, p.z, yaw, ut.TW / 2, ut.TH / 2))  # flat meld footprint
            self.pile.obstacles = boxes

        def _rebuild_melds(self, wind: int, hand):
            s = self.seats[wind]
            if s["melds"] is not None:
                ut.destroy_tree(s["melds"])                          # meld holder → tiles → child meshes
                s["melds"] = None
            flat = [tid for m in hand.melds for tid in m.tiles]
            flat += list(hand.flowers) + list(hand.animals)          # bonus tiles sit face-up too
            if not flat:
                return
            mh = Entity(parent=s["holder"], rotation_x=-90)          # lying flat, face-up
            s["melds"] = mh
            x0 = -(len(flat) - 1) / 2 * ut.STEP
            for i, tid in enumerate(flat):
                t = ut.make_tile(tid, reveal=True, parent=mh, bright=0.8)
                t.rotation_z = 0                            # spin the flat face so it reads upright to its owner
                # mh's -90° x-rotation maps local y → holder -z, so negate to sit the meld in
                # FRONT of its owner (just inside the hand) instead of across the table
                t.position = (x0 + i * ut.STEP, -(ut.EDGE - 1.2), ut.THROW_REST_Y)

        def _animate_discard(self, wind: int, tid: int):
            holder = self.seats[wind]["holder"]
            local = holder.right * random.uniform(-1.5, 1.5) + holder.forward * (ut.EDGE - 0.8)
            self._last_discard = self.pile.throw(tid, Vec3(local.x, ut.THROW_REST_Y, local.z))

        # -- hand lifecycle ------------------------------------------------------------
        def _start_hand(self):
            self.match.start_hand()                                  # deals; we read state directly
            self.queue.clear()
            self.pile.clear()
            self._set_banner(None)
            self._reveal_all = False                                 # back to hidden opponents
            if self._sandbox and self.match.human_wind is not None:  # rig a claim-rich test hand
                hand = self.match.engine.players[self.match.human_wind].hand
                hand.concealed[:] = tiles_from_names(_SANDBOX_HAND)
                self._add_log("[sandbox] rigged test hand")
            self._add_log(f"- {self.match.round_label}, hand {self.match.hand_number} -")
            for w in WINDS:
                self._rebuild_hand(w)
            self._refresh_hud()

        def _show_next_button(self):
            """Offer a button to advance to the next hand (shown at hand-end)."""
            if self._next_btn is not None:
                return
            self._next_btn = Button(parent=camera.ui, text="Next hand", color=_BTN_BG,
                                    text_color=color.white, scale=(0.26, 0.07), y=-0.42)
            self._next_btn.on_click = self._on_next_hand

        def _on_next_hand(self):
            if self._next_btn is not None:
                ut.destroy_tree(self._next_btn)
                self._next_btn = None
            self._next_hand()

        def _next_hand(self):
            self.match.finish_hand()
            if self.match.is_complete:
                self._set_banner("Match complete")
                self.mode = "_done"
                return
            self._start_hand()

        # -- event pump ----------------------------------------------------------------
        def update(self):
            if self.mode == "_done":
                return
            eng = self.match.engine
            self._pulse += time.dt
            if self.mode == "interactive":
                if eng.awaiting_human_discard:
                    self._hover_tiles()                              # lift the tile under the cursor
                elif self._claim_phase is not None:
                    self._claim_hover()
            self._beat += time.dt
            if self._beat < BEAT:
                return
            self._beat = 0.0
            if self.queue:                                           # drain queued events (animations + AWAIT_*)
                self._handle(self.queue.popleft())
                return
            if self.mode == "interactive" and (eng.awaiting_human_discard
                                               or eng.awaiting_human_claim
                                               or eng.awaiting_human_win):
                return                                               # waiting on the player's click
            if eng.is_finished:
                self._show_next_button()           # wait for the player to advance
                return
            self.queue.extend(eng.step())

        def _handle(self, ev):
            t = ev.type
            if t == EventType.DRAW:
                self._drawn[ev.seat] = ev.tile                       # set apart on rebuild
                ua.play("draw")
                self._rebuild_hand(ev.seat)
                self._refresh_hud()
            elif t == EventType.BONUS:
                self._drawn.pop(ev.seat, None)                       # no loose drawn tile
                self._add_log(f"{self._pname(ev.seat)} reveals {bonus_tile_name(ev.tile)}")
                self._rebuild_hand(ev.seat)
                self._refresh_hud()
            elif t == EventType.MELD:
                self._drawn.pop(ev.seat, None)
                ua.play("meld")
                self._add_log(f"{self._pname(ev.seat)} {ev.detail.get('meld_type', 'meld')}s {tile_name(ev.tile)}")
                if ev.detail.get("from") is not None:                # claimed off a discard, not a self-kong
                    self.pile.remove(self._last_discard)             # take the claimed tile off the table
                    self._last_discard = None
                self._rebuild_hand(ev.seat)
                self._refresh_hud()
            elif t == EventType.DISCARD:
                self._drawn.pop(ev.seat, None)                       # turn over
                self._add_log(f"{self._pname(ev.seat)} discards {tile_name(ev.tile)}")
                self._animate_discard(ev.seat, ev.tile)
                self._rebuild_hand(ev.seat)
                self._refresh_hud()
            elif t in (EventType.WIN_SELF_DRAW, EventType.WIN_DISCARD):
                ua.play("win", pitch_jitter=0)
                tai = ev.detail.get('tai', '?')
                pwin = self._pname(ev.seat)
                if t == EventType.WIN_SELF_DRAW:
                    zimo = ev.detail.get('zimo_pay', 0)
                    self._add_log(f"{pwin} self-draws - {tai} tai")
                    self._add_log(f"  {pwin} +{zimo * 3}c, others -{zimo}c each")
                else:
                    pay = ev.detail.get('shooter_pay', 0)
                    pshoot = self._pname(ev.detail.get('shooter'))
                    self._add_log(f"{pwin} wins on discard - {tai} tai")
                    self._add_log(f"  {pwin} +{pay}c, {pshoot} -{pay}c")
                self._set_banner(f"{pwin} wins! {tai} tai")
                self._reveal_all = True                              # show every hand face-up
                for w in WINDS:
                    self._rebuild_hand(w)
                self._refresh_hud()                                  # chips were paid this step
                self._show_next_button()
            elif t == EventType.WALL_EXHAUSTED:
                self._add_log("Wall exhausted - draw")
                self._set_banner("Wall exhausted - draw")
                self._reveal_all = True
                for w in WINDS:
                    self._rebuild_hand(w)
                self._show_next_button()
            elif t == EventType.AWAIT_DISCARD:
                if self.mode == "interactive":
                    self._begin_human_discard()
            elif t == EventType.AWAIT_CLAIM:
                if self.mode == "interactive":
                    self._begin_human_claim(ev)
            elif t == EventType.AWAIT_WIN:
                if self.mode == "interactive":
                    self._begin_human_win(ev)
            # DRAW logs nothing (routine); DEAL: nothing to do

        # -- human input (interactive mode) --------------------------------------------
        def _tile_collider(self, t):
            # collider taller than the tile so the hover-lift doesn't move it off the cursor
            # (which otherwise causes the raised tile to jitter up/down)
            return BoxCollider(t, size=Vec3(ut.TW, ut.TH + 1.2, ut.TD))

        def _begin_human_discard(self):
            self.prompt.text = "" if self._discarded_once else "Your turn - click a tile to discard"
            for t in self.seats[self._front_wind()]["tiles"]:
                t.collider = self._tile_collider(t)
                t.on_click = (lambda tid=t.tid: self._human_discard(tid))

        def _disarm_tiles(self):
            for t in self.seats[self._front_wind()]["tiles"]:
                t.collider = None                                # stops further clicks/hover
                t.y = t.base_y

        def _hover_tiles(self):
            hov = mouse.hovered_entity
            for t in self.seats[self._front_wind()]["tiles"]:
                t.y = t.base_y + (0.3 if t is hov else 0.0)

        def _human_discard(self, tid):
            eng = self.match.engine
            if not eng.awaiting_human_discard:
                return
            self._disarm_tiles()
            self.prompt.text = ""
            self._discarded_once = True
            self.queue.extend(eng.submit_discard(tid))

        # -- claims: highlight the discard, then pick hand tiles to form the meld -------
        def _begin_human_claim(self, ev):
            self._claim_kinds = ev.detail["kinds"]
            self._claim_discard_tid = ev.tile
            self._claim_sel = []
            self._claim_phase = "offer"
            if not self._seen_claim_hint:
                self.prompt.text = f"You can claim {tile_name(ev.tile)} - click it, or 'draw' to skip"
                self._seen_claim_hint = True
            else:
                self.prompt.text = ""
            d = self._last_discard
            if d is not None:                                    # arm + colour the glowing discard
                d.collider = BoxCollider(d, size=Vec3(ut.TW, ut.TH, ut.TD))
                d.on_click = self._claim_pick_discard
                self._claim_marker = Entity(parent=d, model="cube", color=color.gold,
                                            scale=(ut.TW * 1.12, ut.TH * 1.12, ut.TD * 1.12))
                self._claim_marker.alpha = 0.5
            self._make_draw_skip()

        def _make_draw_skip(self):
            """A blank translucent tile labelled 'draw' where a drawn tile would appear —
            click it to decline the claim (and just draw on your turn)."""
            front = self._front_wind()
            holder = self.seats[front]["holder"]
            n = len(self.seats[front]["tiles"])
            bx = -(n - 1) / 2 * ut.STEP - ut.STEP - DRAW_GAP     # the just-drawn-tile slot
            e = Entity(parent=holder, model="cube", scale=(ut.TW, ut.TH, ut.TD),
                       position=(bx, ut.TH / 2, ut.EDGE), color=color.white)
            e.alpha = 0.3
            e.collider = "box"
            e.on_click = self._human_pass
            lbl = Text("draw", parent=e, origin=(0, 0), world_scale=18,
                       y=-0.9, color=color.white)
            lbl.billboard = True
            self._draw_skip = e

        def _claim_pick_discard(self):
            if self._claim_phase != "offer":
                return
            self._claim_phase = "select"
            self.prompt.text = "Click your tiles to form the meld"
            self._claim_relevant = self._relevant_claim_tids()
            for t in self.seats[self._front_wind()]["tiles"]:
                if t.tid in self._claim_relevant:
                    t.collider = self._tile_collider(t)
                    t.on_click = (lambda e=t: self._claim_toggle(e))

        def _relevant_claim_tids(self) -> set:
            tids: set = set()
            k = self._claim_kinds
            if "pong" in k or "kong" in k:
                tids.add(self._claim_discard_tid)
            for opt in k.get("chow", []):
                tids.update(t for t in opt if t != self._claim_discard_tid)
            return tids

        def _claim_toggle(self, t):
            if self._claim_phase != "select":
                return
            if t in self._claim_sel:
                self._claim_sel.remove(t)
                t.y = t.base_y
            else:
                self._claim_sel.append(t)
                t.y = t.base_y + 0.3
            self._try_complete_claim()

        def _try_complete_claim(self):
            k, d = self._claim_kinds, self._claim_discard_tid
            sel = [e.tid for e in self._claim_sel]
            if len(sel) == 3 and all(x == d for x in sel) and "kong" in k:
                self._do_claim("kong")
            elif len(sel) == 2 and all(x == d for x in sel) and "pong" in k:
                if "kong" in k:
                    self._pong_confirm_button()                  # also a kong possible → let them choose
                else:
                    self._do_claim("pong")
            elif len(sel) == 2 and "chow" in k:
                want = sorted(sel + [d])
                for opt in k["chow"]:
                    if sorted(opt) == want:
                        self._do_claim("chow", opt)
                        return

        def _pong_confirm_button(self):
            if any(getattr(b, "_is_pong", False) for b in self._claim_btns):
                return
            b = Button(parent=camera.ui, text="Pong (or pick the 3rd for kong)",
                       color=_BTN_BG, text_color=color.white, scale=(0.34, 0.06), y=-0.4)
            b._is_pong = True
            b.on_click = lambda: self._do_claim("pong")
            self._claim_btns.append(b)

        def _do_claim(self, kind, chow=None):
            eng = self.match.engine
            if not eng.awaiting_human_claim:
                return
            self._end_claim_ui()
            self.queue.extend(eng.submit_claim(kind, chow))

        # -- win prompt (hu / decline) -------------------------------------------------
        def _begin_human_win(self, ev):
            if ev.detail.get("self_draw"):
                how = "self-draw"
            else:
                how = f"on P{self.match.player_at[ev.detail['from']]}'s discard"
            self.prompt.text = f"You can win ({how})!"
            hu = Button(parent=camera.ui, text="Hu!", color=_BTN_BG,
                        text_color=color.white, scale=(0.22, 0.07), x=-0.13, y=-0.4)
            hu.on_click = lambda: self._do_win(True)
            skip = Button(parent=camera.ui, text="Skip", color=_BTN_BG,
                          text_color=color.white, scale=(0.22, 0.07), x=0.13, y=-0.4)
            skip.on_click = lambda: self._do_win(False)
            self._win_btns = [hu, skip]

        def _do_win(self, take):
            eng = self.match.engine
            if not eng.awaiting_human_win:
                return
            for b in self._win_btns:
                ut.destroy_tree(b)
            self._win_btns = []
            self.prompt.text = ""
            self.queue.extend(eng.submit_win() if take else eng.decline_win())

        def _human_pass(self):
            eng = self.match.engine
            if not eng.awaiting_human_claim:
                return
            self._end_claim_ui()
            self.queue.extend(eng.pass_claim())

        def _end_claim_ui(self):
            for b in self._claim_btns:
                ut.destroy_tree(b)
            self._claim_btns = []
            if self._draw_skip is not None:
                ut.destroy_tree(self._draw_skip)
                self._draw_skip = None
            if self._claim_marker is not None:
                ut.destroy_tree(self._claim_marker)
                self._claim_marker = None
            for t in self.seats[self._front_wind()]["tiles"]:
                t.collider = None
                t.y = t.base_y
            if self._last_discard is not None:
                self._last_discard.collider = None
                self._last_discard.scale = 1
            self._claim_phase = None
            self._claim_sel = []
            self.prompt.text = ""

        def _claim_hover(self):
            if self._claim_phase == "offer" and self._last_discard is not None:
                try:
                    self._last_discard.scale = 1.0 + 0.18 * (0.5 + 0.5 * math.sin(self._pulse * 6))
                except Exception:
                    self._last_discard = None        # tile was recycled/destroyed — stop pulsing
            elif self._claim_phase == "select":
                hov = mouse.hovered_entity
                for t in self.seats[self._front_wind()]["tiles"]:
                    if t.tid in self._claim_relevant:
                        lifted = t in self._claim_sel
                        t.y = t.base_y + (0.3 if lifted else (0.12 if t is hov else 0.0))

        def _add_log(self, msg: str):
            self._log.append(msg)
            self.log.text = "\n".join(self._log)

        def _pname(self, seat: int) -> str:
            """Stable player label (P1-P4) for a seat wind."""
            return f"P{self.match.player_at[seat]}"

        # -- HUD -----------------------------------------------------------------------
        def _visible_tai(self, wind: int) -> tuple[int, list[str]]:
            """Tai visible to everyone from one player's exposed melds and bonus tiles:
            dragon pongs, seat/prevailing wind pongs, matching flowers/seasons, animals.
            Concealed tiles are not counted (they're hidden information)."""
            e = self.match.engine
            hand = e.players[wind].hand
            prevailing = e.prevailing_wind
            items: list[tuple[str, int]] = []
            for mld in hand.melds:
                if mld.type not in (MeldType.PONG, MeldType.KONG):
                    continue
                head = mld.tiles[0]
                if DRAGON_START <= head < DRAGON_END:
                    items.append((f"{tile_name(head)} pong", 1))
                elif WIND_START <= head < WIND_END:
                    tai = (1 if head == wind else 0) + (1 if head == prevailing else 0)
                    if tai:
                        items.append((f"{tile_name(head)} pong{' x2' if tai == 2 else ''}", tai))
            for f in hand.flowers:
                if f == SEAT_FLOWER.get(wind):
                    items.append(("flower", 1))
                elif f == SEAT_SEASON.get(wind):
                    items.append(("season", 1))
            items += [("animal", 1) for _ in hand.animals]
            return sum(t for _, t in items), [lbl for lbl, _ in items]

        def _refresh_hud(self):
            e, m = self.match.engine, self.match
            lines = [f"{m.round_label}, hand {m.hand_number}",
                     f"Table wind: {_FULL[m.table_wind]}",
                     "", "Players (chips / visible tai)"]
            wind_of = {m.player_at[w]: w for w in WINDS}     # P# stays put; its seat wind rotates
            for p in (1, 2, 3, 4):
                w = wind_of[p]
                mark = ">" if w == e.current_seat else " "
                tai, srcs = self._visible_tai(w)
                tai_str = f"  +{tai}t [{', '.join(srcs)}]" if tai else ""
                lines.append(f"{mark} P{p} ({_NAME[w]})  {e.chips[w]}c{tai_str}")
            lines += ["", f"Tiles left: {e.wall_remaining}"]
            self.hud.text = "\n".join(lines)

        def _set_banner(self, msg):
            if self._banner is not None:
                ut.destroy_tree(self._banner)                        # background=True adds a child entity
                self._banner = None
            if msg:
                self._banner = Text(text=msg, origin=(0, 0), scale=2.4,
                                    color=color.yellow, background=True)


    class Menu(Entity):
        """Title + Play (you vs AI) / Spectate / Quit."""

        def __init__(self, on_start):
            super().__init__(parent=camera.ui)
            self.title = Text(parent=self, text="crackedMahjong - 3D",
                              origin=(0, 0), y=0.30, scale=2.2)
            self.b_play = Button(parent=self, text="Play (you vs AI)", color=_BTN_BG,
                                 text_color=color.white, scale=(0.42, 0.085), y=0.08)
            self.b_spec = Button(parent=self, text="Spectate (watch AI)", color=_BTN_BG,
                                 text_color=color.white, scale=(0.42, 0.085), y=-0.03)
            self.b_quit = Button(parent=self, text="Quit", color=_BTN_BG,
                                 text_color=color.white, scale=(0.42, 0.085), y=-0.14)
            self.b_play.on_click = lambda: on_start("interactive")
            self.b_spec.on_click = lambda: on_start("spectator")
            self.b_quit.on_click = application.quit

        def hide(self):
            self.enabled = False


def main():
    if not _HAVE_URSINA:
        sys.exit('Ursina is not installed - run:  pip install -e ".[threed]"')

    app = Ursina(title="crackedMahjong - 3D")
    _use_ui_font()                           # JetBrains Mono for menu + all UI text
    ua.init()                                # build the tile sound bank
    ur.setup_cabin()                         # dark wood cabin + lights + camera + retro shader
    ut.build_felt()
    holder = {"menu": None, "driver": None}

    def start(mode):
        if holder["menu"]:
            holder["menu"].hide()
        # MAHJONG_SEED pins the deal for testing edge cases; otherwise random per game.
        env_seed = os.environ.get("MAHJONG_SEED")
        seed = int(env_seed) if env_seed else random.randrange(1 << 30)
        print(f"[crackedMahjong] game seed = {seed}")
        holder["driver"] = GameDriver(mode=mode, seed=seed)

    holder["menu"] = Menu(start)
    app.run()


if __name__ == "__main__":
    main()
