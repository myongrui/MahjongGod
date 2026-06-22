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
import random
import sys
from collections import deque

try:
    from ursina import (Ursina, Entity, Text, Button, color, camera, Vec3,
                        time, application, window)
    _HAVE_URSINA = True
except Exception:                       # ursina not installed → module still imports
    _HAVE_URSINA = False

from cracked.engine import EventType
from cracked.match import GameMatch
from cracked.tiles import Wind, tile_name, bonus_tile_name
from cracked import ursina_table as ut   # reuse: make_tile, setup_room, build_felt, ThrowManager, constants

WINDS = [int(Wind.EAST), int(Wind.SOUTH), int(Wind.WEST), int(Wind.NORTH)]
_NAME = {int(Wind.EAST): "E", int(Wind.SOUTH): "S", int(Wind.WEST): "W", int(Wind.NORTH): "N"}
_FULL = {int(Wind.EAST): "East", int(Wind.SOUTH): "South",
         int(Wind.WEST): "West", int(Wind.NORTH): "North"}

BEAT = 0.45                              # seconds per animated event
HAND_PAUSE = 2.2                         # pause after a win/draw before the next hand
DRAW_GAP = ut.STEP * 0.6                 # gap between the hand and the just-drawn tile


if _HAVE_URSINA:

    class GameDriver(Entity):
        """Owns a GameMatch and pumps `engine.step()` on a beat, animating each event."""

        def __init__(self, mode: str = "spectator", seed: int = 7):
            super().__init__()
            self.mode = mode
            if mode == "interactive":
                self.match = GameMatch(n_rounds=1, human_initial_wind=int(Wind.EAST), seed=seed)
            else:
                self.match = GameMatch(n_rounds=1, seed=seed)        # all-AI spectator
            self.pile = ut.ThrowManager(auto=False)                  # discard slinger (no auto-toss)
            self.seats = {w: {"holder": None, "tiles": [], "melds": None} for w in WINDS}
            self._drawn: dict = {}                                   # seat -> just-drawn tile (shown apart)
            self.queue: deque = deque()
            self._beat = 0.0
            self._pause = 0.0
            self._banner = None
            edge = window.aspect_ratio / 2                        # UI x-extent depends on window aspect
            self.hud = Text(text="", origin=(-.5, .5), position=(-edge + 0.02, .47), scale=0.9, font="VeraMono.ttf")
            self.log = Text(text="", origin=(.5, .5), position=(edge - 0.02, 0.3), scale=0.7, font="VeraMono.ttf")
            self._log: deque = deque(maxlen=24)                 # rolling action log (oldest drops off top)
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
            reveal = (wind == self._front_wind())                    # only the front seat face-up
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
            self.pile.throw(tid, Vec3(local.x, ut.THROW_REST_Y, local.z))

        # -- hand lifecycle ------------------------------------------------------------
        def _start_hand(self):
            self.match.start_hand()                                  # deals; we read state directly
            self.queue.clear()
            self.pile.clear()
            self._set_banner(None)
            self._add_log(f"— {self.match.round_label}, hand {self.match.hand_number} —")
            for w in WINDS:
                self._rebuild_hand(w)
            self._refresh_hud()

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
            if self.mode == "interactive" and self.match.engine.awaiting_human_discard:
                return                                               # wait for input (Phase B)
            if self._pause > 0:
                self._pause -= time.dt
                if self._pause <= 0:
                    self._next_hand()
                return
            self._beat += time.dt
            if self._beat < BEAT:
                return
            self._beat = 0.0
            if not self.queue:
                eng = self.match.engine
                if eng.is_finished:
                    self._pause = HAND_PAUSE
                    return
                self.queue.extend(eng.step())
                if not self.queue:
                    return
            self._handle(self.queue.popleft())

        def _handle(self, ev):
            t = ev.type
            if t == EventType.DRAW:
                self._drawn[ev.seat] = ev.tile                       # set apart on rebuild
                self._rebuild_hand(ev.seat)
                self._refresh_hud()
            elif t == EventType.BONUS:
                self._drawn.pop(ev.seat, None)                       # no loose drawn tile
                self._add_log(f"{_NAME[ev.seat]} reveals {bonus_tile_name(ev.tile)}")
                self._rebuild_hand(ev.seat)
                self._refresh_hud()
            elif t == EventType.MELD:
                self._drawn.pop(ev.seat, None)
                self._add_log(f"{_NAME[ev.seat]} {ev.detail.get('meld_type', 'meld')}s {tile_name(ev.tile)}")
                self._rebuild_hand(ev.seat)
                self._refresh_hud()
            elif t == EventType.DISCARD:
                self._drawn.pop(ev.seat, None)                       # turn over
                self._add_log(f"{_NAME[ev.seat]} discards {tile_name(ev.tile)}")
                self._animate_discard(ev.seat, ev.tile)
                self._rebuild_hand(ev.seat)
                self._refresh_hud()
            elif t in (EventType.WIN_SELF_DRAW, EventType.WIN_DISCARD):
                tai = ev.detail.get('tai', '?')
                how = "self-draws" if t == EventType.WIN_SELF_DRAW else "wins on discard"
                self._add_log(f"{_NAME[ev.seat]} {how} — {tai} tai")
                self._set_banner(f"{_NAME[ev.seat]} wins!   {tai} tai")
                self._refresh_hud()                                  # chips were paid this step
                self._pause = HAND_PAUSE
            elif t == EventType.WALL_EXHAUSTED:
                self._add_log("Wall exhausted — draw")
                self._set_banner("Wall exhausted — draw")
                self._pause = HAND_PAUSE
            # DRAW logs nothing (routine); DEAL / AWAIT_DISCARD: nothing to do

        def _add_log(self, msg: str):
            self._log.append(msg)
            self.log.text = "\n".join(self._log)

        # -- HUD -----------------------------------------------------------------------
        def _refresh_hud(self):
            e, m = self.match.engine, self.match
            lines = [f"Table wind: {_FULL[m.table_wind]}",
                     f"Current seat: {_FULL[e.current_seat]}",
                     "", "Chips"]
            for w in WINDS:
                mark = ">" if w == e.current_seat else " "
                lines.append(f"{mark} {_FULL[w]:<6}{e.chips[w]}")
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
        """Title + Spectate / Play / Quit. Play is greyed until interactive mode lands."""

        def __init__(self, on_start):
            super().__init__(parent=camera.ui)
            self.title = Text(parent=self, text="crackedMahjong — 3D",
                              origin=(0, 0), y=0.30, scale=2.2)
            self.b_spec = Button(parent=self, text="Spectate (watch AI)", color=color.azure,
                                 scale=(0.42, 0.085), y=0.08)
            self.b_play = Button(parent=self, text="Play — coming soon", color=color.gray,
                                 scale=(0.42, 0.085), y=-0.03)
            self.b_quit = Button(parent=self, text="Quit", color=color.dark_gray,
                                 scale=(0.42, 0.085), y=-0.14)
            self.b_spec.on_click = lambda: on_start("spectator")
            self.b_quit.on_click = application.quit                  # Play has no handler yet

        def hide(self):
            self.enabled = False


def main():
    if not _HAVE_URSINA:
        sys.exit('Ursina is not installed — run:  pip install -e ".[threed]"')

    app = Ursina(title="crackedMahjong — 3D")
    ut.setup_room()
    ut.build_felt()
    holder = {"menu": None, "driver": None}

    def start(mode):
        if holder["menu"]:
            holder["menu"].hide()
        holder["driver"] = GameDriver(mode=mode)

    holder["menu"] = Menu(start)
    app.run()


if __name__ == "__main__":
    main()
