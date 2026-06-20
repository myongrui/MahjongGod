"""
Standalone physics demo — our custom tile faces, rotating, colliding, bouncing.

A throwaway, engine-free tech demo: each tile uses the high-resolution custom
face artwork from `cracked.tui_tiles` (bamboo sticks, circle rings, and real CJK
symbols for characters/winds/dragons) painted onto a rigid body with velocity
and spin. Tiles bounce off the walls and off each other (equal-mass elastic
collisions). Rendered on a half-block pixel canvas; the face is sampled in the
tile's rotated local frame (downscaled to the display size), so it rotates with
the tile and the spin reads through the artwork.

This is NOT Textual's animate() (that only tweens A->B); it's a fixed-timestep
physics loop driven by a timer.

Run it with:

    python -m cracked.tui_physics_demo
"""
from __future__ import annotations

import math
import random
from typing import Optional

from rich.style import Style
from rich.text import Text

from textual.app import App, ComposeResult
from textual.widgets import Static

from cracked.tui_tiles import make_face, FW as _FW, FH as _FH, FELT as _FELT

# Display size of a tile in canvas pixels (smaller than the face bitmap, which
# is downscaled when sampled — keeps tiles small enough to bounce around).
_TW, _TH = 16.0, 21.0
_RADIUS = 7.5
_FPS = 20
_RESTITUTION = 0.995

# A spread of suits/ranks so the artwork (incl. real CJK symbols) varies.
_DEMO_TIDS = [4, 8, 13, 22, 27, 31]   # b5 b9 c5 d5 East RedDragon


class _Tile:
    __slots__ = ("x", "y", "vx", "vy", "ang", "av", "face")

    def __init__(self, x, y, vx, vy, face):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.ang = random.uniform(0, math.tau)
        self.av = random.uniform(-2.2, 2.2)
        self.face = face


class PhysicsDemoApp(App):
    TITLE = "crackedMahjong — physics demo (custom faces)"

    CSS = f"""
    Screen {{ background: {_FELT}; }}
    #canvas {{ width: 1fr; height: 1fr; }}
    #hint {{ dock: bottom; height: 1; color: $text-muted; text-align: center; }}
    """

    BINDINGS = [("q", "quit", "Quit"), ("escape", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Static(id="canvas")
        yield Static("physics demo — custom tile faces collide & bounce   ·   q / esc to quit", id="hint")

    def on_mount(self) -> None:
        self._pw = max(24, self.size.width)
        self._ph = max(24, (self.size.height - 1) * 2)
        self._tiles = self._spawn()
        self.set_interval(1 / _FPS, self._tick)

    def _spawn(self) -> list[_Tile]:
        tiles: list[_Tile] = []
        for tid in _DEMO_TIDS:
            x = random.uniform(_TW, self._pw - _TW)
            y = random.uniform(_TH, self._ph - _TH)
            a = random.uniform(0, math.tau)
            speed = random.uniform(12, 22)
            tiles.append(_Tile(x, y, math.cos(a) * speed, math.sin(a) * speed, make_face(tid)))
        return tiles

    # ------------------------------------------------------------------ physics
    def _tick(self) -> None:
        self._step(1 / _FPS)
        self.query_one("#canvas", Static).update(self._render())

    def _step(self, dt: float) -> None:
        ts = self._tiles
        for t in ts:
            t.x += t.vx * dt
            t.y += t.vy * dt
            t.ang += t.av * dt
            if t.x - _RADIUS < 0:
                t.x = _RADIUS; t.vx = abs(t.vx) * _RESTITUTION; t.av += random.uniform(-1, 1)
            elif t.x + _RADIUS > self._pw:
                t.x = self._pw - _RADIUS; t.vx = -abs(t.vx) * _RESTITUTION; t.av += random.uniform(-1, 1)
            if t.y - _RADIUS < 0:
                t.y = _RADIUS; t.vy = abs(t.vy) * _RESTITUTION; t.av += random.uniform(-1, 1)
            elif t.y + _RADIUS > self._ph:
                t.y = self._ph - _RADIUS; t.vy = -abs(t.vy) * _RESTITUTION; t.av += random.uniform(-1, 1)

        n = len(ts)
        for i in range(n):
            a = ts[i]
            for j in range(i + 1, n):
                b = ts[j]
                dx, dy = b.x - a.x, b.y - a.y
                dist = math.hypot(dx, dy)
                if 0 < dist < 2 * _RADIUS:
                    nx, ny = dx / dist, dy / dist
                    overlap = (2 * _RADIUS - dist) / 2
                    a.x -= nx * overlap; a.y -= ny * overlap
                    b.x += nx * overlap; b.y += ny * overlap
                    rvx, rvy = b.vx - a.vx, b.vy - a.vy
                    vn = rvx * nx + rvy * ny
                    if vn < 0:
                        a.vx += vn * nx; a.vy += vn * ny
                        b.vx -= vn * nx; b.vy -= vn * ny
                        a.av += random.uniform(-1.5, 1.5)
                        b.av += random.uniform(-1.5, 1.5)

    # ------------------------------------------------------------------ render
    def _render(self) -> Text:
        pw, ph = self._pw, self._ph
        buf: list[list[Optional[str]]] = [[None] * pw for _ in range(ph)]
        hw, hh = _TW / 2, _TH / 2
        sx, sy = (_FW - 1) / _TW, (_FH - 1) / _TH        # face downscale factors
        reach = int(math.hypot(hw, hh)) + 1
        for t in self._tiles:
            ca, sa = math.cos(-t.ang), math.sin(-t.ang)
            x0, x1 = max(0, int(t.x - reach)), min(pw - 1, int(t.x + reach))
            y0, y1 = max(0, int(t.y - reach)), min(ph - 1, int(t.y + reach))
            face = t.face
            for py in range(y0, y1 + 1):
                for px in range(x0, x1 + 1):
                    dx, dy = px - t.x, py - t.y
                    lx = dx * ca - dy * sa
                    ly = dx * sa + dy * ca
                    if -hw <= lx <= hw and -hh <= ly <= hh:
                        fx = int((lx + hw) * sx)
                        fy = int((ly + hh) * sy)
                        if 0 <= fx < _FW and 0 <= fy < _FH:
                            c = face[fy][fx]
                            if c is not None:
                                buf[py][px] = c

        text = Text()
        felt = _FELT
        for cy in range(ph // 2):
            top_row = buf[2 * cy]
            bot_row = buf[2 * cy + 1]
            for cx in range(pw):
                top = top_row[cx]
                bot = bot_row[cx]
                if top is None and bot is None:
                    text.append(" ")
                else:
                    text.append("▀", Style(color=top or felt, bgcolor=bot or felt))
            text.append("\n")
        return text


def main() -> None:
    PhysicsDemoApp().run()


if __name__ == "__main__":
    main()
