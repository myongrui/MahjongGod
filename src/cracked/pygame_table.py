"""
pygame-ce proof-of-concept — the mahjong table as real 2D graphics, faked 2.5D.

Validates the pipeline (engine deals a real hand → our `make_face` pixel art as
crisp nearest-scaled sprites → animated discard) and gives it a 3D-ish look the
terminal never could: a perspective felt (trapezoid + gradient lighting), tiles
with thickness + drop shadows so they read as physical objects, and depth
scaling (your hand large in front, opponents smaller toward the back).

Written async so the same code runs natively now and exports to the browser via
`pygbag` later. No game logic here — it reuses `cracked.engine` / `cracked.tui_tiles`.

Run it with:

    python -m cracked.pygame_table       # desktop
    # later, for web:  pygbag src/cracked/pygame_table.py
"""
from __future__ import annotations

import asyncio

import pygame

from cracked.tui_tiles import make_face, _blank_face, _IVORY, FW, FH, FELT
from cracked.engine import GameEngine
from cracked.tiles import Wind, tile_name

SCALE = 2                              # base integer scale → crisp pixel tiles
TW, TH = FW * SCALE, FH * SCALE        # 64 x 88 px per tile (at depth 1.0)
GAP = 8
WIN_W, WIN_H = 1024, 700

_GOLD = pygame.Color("#c8a23a")
_DEPTH = 9                             # tile thickness (extruded body)
_SIDE = (206, 190, 142)                # tile body (cream)
_SIDE_DARK = (150, 136, 92)            # shaded bottom of the body
_BACK_BLUE = "#2a6f9e"


def _back_face():
    f = _blank_face()
    for y in range(FH):
        for x in range(FW):
            if f[y][x] == _IVORY:
                f[y][x] = _BACK_BLUE
    return f


def _surface_from_face(face) -> pygame.Surface:
    surf = pygame.Surface((FW, FH), pygame.SRCALPHA)
    for y in range(FH):
        row = face[y]
        for x in range(FW):
            c = row[x]
            if c is not None:
                surf.set_at((x, y), pygame.Color(c))
    return surf


def _bevel(face: pygame.Surface) -> pygame.Surface:
    """Directional bevel (top/left highlight, bottom/right shade) → glossy 3D top."""
    w, h = face.get_size()
    ov = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.line(ov, (255, 255, 255, 115), (4, 2), (w - 5, 2), 3)
    pygame.draw.line(ov, (255, 255, 255, 75), (2, 4), (2, h - 5), 3)
    pygame.draw.line(ov, (0, 0, 0, 125), (4, h - 3), (w - 5, h - 3), 3)
    pygame.draw.line(ov, (0, 0, 0, 85), (w - 3, 4), (w - 3, h - 5), 3)
    face.blit(ov, (0, 0))
    return face


def _slab(raw: pygame.Surface, scale: float) -> pygame.Surface:
    """A raw 32x44 face → an extruded 3D tile slab (same look as the physics demo)."""
    tw, th = max(1, int(TW * scale)), max(1, int(TH * scale))
    face = _bevel(pygame.transform.scale(raw, (tw, th)))   # nearest → crisp
    depth = max(3, int(_DEPTH * scale))
    shp = max(5, int(11 * scale))
    surf = pygame.Surface((tw, th + depth + shp), pygame.SRCALPHA)
    # contact shadow at the base
    pygame.draw.ellipse(surf, (0, 0, 0, 50),
                        (int(tw * 0.06), th + depth - 2, int(tw * 0.88), shp))
    # extrude the body downward, shaded at the foot
    mask = pygame.mask.from_surface(face)
    body = mask.to_surface(setcolor=(*_SIDE, 255), unsetcolor=(0, 0, 0, 0))
    foot = mask.to_surface(setcolor=(*_SIDE_DARK, 255), unsetcolor=(0, 0, 0, 0))
    for d in range(depth, 0, -1):
        surf.blit(foot if d > depth - 3 else body, (0, d))
    surf.blit(face, (0, 0))
    return surf


def _ease_out_cubic(p: float) -> float:
    return 1 - (1 - p) ** 3


def _lerp(a, b, p):
    return a + (b - a) * p


class TablePoC:
    def __init__(self) -> None:
        eng = GameEngine(human_seats={int(Wind.EAST)}, seed=7)
        eng.deal()
        concealed = eng.players[int(Wind.EAST)].hand.concealed_tiles_list()
        self.hand = concealed[:13]
        self.drawn = concealed[13] if len(concealed) > 13 else concealed[-1]

        self._raw: dict[int, pygame.Surface] = {}
        self._raw_back = _surface_from_face(_back_face())
        self._cache: dict[tuple, pygame.Surface] = {}

        self._discards: list[tuple[int, int, int]] = []
        self._t = 0.0
        import random
        self._rng = random.Random(1)

        self._bg, self._well = self._build_bg()

        hand_w = 13 * (TW + GAP) - GAP
        self._hand_x0 = (WIN_W - hand_w) // 2 - 30
        self._hand_y = WIN_H - TH - 40
        self._drawn_home = (self._hand_x0 + hand_w + 46, self._hand_y)

    # ------------------------------------------------------------------ assets
    def _raw_face(self, tid: int) -> pygame.Surface:
        s = self._raw.get(tid)
        if s is None:
            s = self._raw[tid] = _surface_from_face(make_face(tid))
        return s

    def sprite(self, tid, scale: float = 1.0) -> pygame.Surface:
        key = (tid, round(scale, 2))
        s = self._cache.get(key)
        if s is None:
            raw = self._raw_back if tid == "back" else self._raw_face(tid)
            s = self._cache[key] = _slab(raw, scale)
        return s

    def _build_bg(self):
        bg = pygame.Surface((WIN_W, WIN_H))
        top, bot = (6, 34, 25), (13, 56, 40)            # vertical gradient (depth)
        for y in range(WIN_H):
            p = y / WIN_H
            bg.fill(tuple(int(_lerp(top[i], bot[i], p)) for i in range(3)), (0, y, WIN_W, 1))
        # perspective table top (trapezoid: narrower at back/top)
        tl, tr = (225, 64), (WIN_W - 225, 64)
        br, bl = (WIN_W - 70, WIN_H - 50), (70, WIN_H - 50)
        pygame.draw.polygon(bg, (15, 60, 44), [tl, tr, br, bl])
        pygame.draw.polygon(bg, (40, 130, 98), [tl, tr, br, bl], 3)
        # centre well as a flattened ellipse (reads as round on a tilted table)
        well = pygame.Rect(WIN_W // 2 - 210, WIN_H // 2 - 96, 420, 192)
        pygame.draw.ellipse(bg, (8, 38, 29), well)
        pygame.draw.ellipse(bg, _GOLD, well, 2)
        return bg, well

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        self._t += dt
        if self._t >= 1.0:
            w = self._well
            tx = self._rng.randint(w.left + 30, w.right - 70)
            ty = self._rng.randint(w.top + 20, w.bottom - 70)
            self._discards.append((self.drawn, tx, ty))
            if len(self._discards) > 12:
                self._discards.pop(0)
            self._t = 0.0

    def _drawn_pos(self) -> tuple[int, int]:
        p = _ease_out_cubic(min(1.0, self._t))
        sx, sy = self._drawn_home
        ex, ey = self._well.centerx - TW // 2, self._well.centery - TH // 2
        return int(_lerp(sx, ex, p)), int(_lerp(sy, ey, p))

    # -------------------------------------------------------------------- draw
    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        screen.blit(self._bg, (0, 0))

        # opponents recede into the distance (smaller = farther back)
        top_s, side_s, disc_s = 0.55, 0.72, 0.85
        back_top = self.sprite("back", top_s)
        bw = back_top.get_width()
        row_w = 11 * (bw - 4)
        for i in range(11):
            screen.blit(back_top, (WIN_W // 2 - row_w // 2 + i * (bw - 4), 70))
        back_side = self.sprite("back", side_s)
        sh = back_side.get_height()
        for i in range(4):
            screen.blit(back_side, (96, 150 + i * (sh - 18)))
            screen.blit(back_side, (WIN_W - back_side.get_width() - 96, 150 + i * (sh - 18)))

        # discards resting in the well (mid depth)
        for tid, x, y in self._discards:
            screen.blit(self.sprite(tid, disc_s), (x, y))

        # your hand — foreground, full size
        for i, tid in enumerate(self.hand):
            screen.blit(self.sprite(tid), (self._hand_x0 + i * (TW + GAP), self._hand_y))

        # the flying drawn tile shrinks slightly as it travels "into" the table
        p = _ease_out_cubic(min(1.0, self._t))
        screen.blit(self.sprite(self.drawn, _lerp(1.0, disc_s, p)), self._drawn_pos())

        # HUD
        screen.blit(font.render(
            "crackedMahjong — pygame-ce PoC (2.5D)   ·   hand dealt by the engine   ·   esc to quit",
            True, pygame.Color("#cfe8dd")), (20, WIN_H - 22))
        screen.blit(font.render(
            f"drew {tile_name(self.drawn).upper()}", True, _GOLD),
            (self._drawn_home[0], self._hand_y - 24))


async def main() -> None:
    pygame.init()
    pygame.display.set_caption("crackedMahjong — pygame-ce")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    poc = TablePoC()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
        poc.update(dt)
        poc.draw(screen, font)
        pygame.display.flip()
        await asyncio.sleep(0)            # required for pygbag (web) builds
    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
