"""
pygame-ce physics demo — the TUI physics feel, but with real graphics.

Ports the bouncing/rotating/colliding-tiles feel from `tui_physics_demo` into
pygame, where rotation is *true* sprite rotation (not half-block approximation)
and the tiles are our crisp `make_face` art. Equal-mass elastic collisions, wall
bounces, free spin. A fixed-timestep physics loop (not a tween) drives it.

Written async so the same code runs natively now and exports to the browser via
`pygbag` later.

Controls:  space = re-scatter (random impulses)   ·   esc / q = quit

Run it with:

    python -m cracked.pygame_physics
"""
from __future__ import annotations

import asyncio
import math
import random

import pygame

from cracked.tui_tiles import make_face, FW, FH, FELT
from cracked.pygame_table import _surface_from_face

SCALE = 2
TW, TH = FW * SCALE, FH * SCALE
WIN_W, WIN_H = 1024, 700
MARGIN = 26
RADIUS = min(TW, TH) * 0.5 * 0.92      # collision radius (circle approximation)
RESTITUTION = 0.992
FPS = 60

# Tile thickness: the body is extruded downward (screen space) under the face.
DEPTH = 10
SIDE = (206, 190, 142)                 # tile body (cream)
SIDE_DARK = (150, 136, 92)             # shaded bottom of the body

# A spread of suits/ranks so the artwork varies.
_TIDS = [0, 4, 8, 11, 13, 17, 18, 22, 26, 27, 29, 31, 32, 33]


def _tile_sprite(tid: int) -> pygame.Surface:
    surf = pygame.transform.scale(_surface_from_face(make_face(tid)), (TW, TH))
    # directional bevel → the tile reads as a raised, glossy 3D object
    ov = pygame.Surface((TW, TH), pygame.SRCALPHA)
    pygame.draw.line(ov, (255, 255, 255, 115), (4, 2), (TW - 5, 2), 3)     # top highlight
    pygame.draw.line(ov, (255, 255, 255, 75), (2, 4), (2, TH - 5), 3)      # left highlight
    pygame.draw.line(ov, (0, 0, 0, 125), (4, TH - 3), (TW - 5, TH - 3), 3)  # bottom shade
    pygame.draw.line(ov, (0, 0, 0, 85), (TW - 3, 4), (TW - 3, TH - 5), 3)   # right shade
    surf.blit(ov, (0, 0))
    pygame.draw.rect(surf, (50, 44, 24), surf.get_rect(), width=2, border_radius=4)
    return surf


def _make_shadow() -> pygame.Surface:
    """A faint, orientation-agnostic contact shadow under the tile slab."""
    pad = 14
    s = pygame.Surface((TW + 2 * pad, TH + 2 * pad), pygame.SRCALPHA)
    cx, cy = s.get_width() // 2, s.get_height() // 2
    for grow, alpha in [(14, 16), (8, 20), (2, 24)]:
        r = pygame.Rect(0, 0, TW + grow, TH + grow)
        r.center = (cx, cy)
        pygame.draw.ellipse(s, (0, 0, 0, alpha), r)
    return s


# --- oriented-rectangle (OBB) collision via the Separating Axis Theorem -------

_HW, _HH = TW / 2, TH / 2


def _corners(cx: float, cy: float, ang: float):
    c, s = math.cos(ang), math.sin(ang)
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = sx * _HW, sy * _HH
        pts.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return pts


def _project(corners, ax):
    dots = [px * ax[0] + py * ax[1] for px, py in corners]
    return min(dots), max(dots)


def _sat(ca, axes_a, cb, axes_b):
    """Return (nx, ny, depth) of the minimum-penetration axis, or None if apart."""
    best, bn = 1e18, None
    for ax in (*axes_a, *axes_b):
        amin, amax = _project(ca, ax)
        bmin, bmax = _project(cb, ax)
        if amax < bmin or bmax < amin:
            return None                      # found a separating axis → no hit
        ov = min(amax, bmax) - max(amin, bmin)
        if ov < best:
            best, bn = ov, ax
    return bn[0], bn[1], best


class _Tile:
    __slots__ = ("x", "y", "vx", "vy", "ang", "av", "sprite")

    def __init__(self, x, y, vx, vy, sprite):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.ang = random.uniform(0, math.tau)
        self.av = random.uniform(-3.0, 3.0)
        self.sprite = sprite


class PhysicsWorld:
    def __init__(self) -> None:
        self._sprites = {t: _tile_sprite(t) for t in _TIDS}
        self.left, self.top = MARGIN, MARGIN
        self.right, self.bottom = WIN_W - MARGIN, WIN_H - MARGIN
        self.tiles = self._spawn()
        self._bg = self._build_bg()
        self._shadow = _make_shadow()

    def _spawn(self) -> list[_Tile]:
        rng = random.Random(3)
        tiles = []
        for tid in _TIDS:
            x = rng.uniform(self.left + RADIUS, self.right - RADIUS)
            y = rng.uniform(self.top + RADIUS, self.bottom - RADIUS)
            a = rng.uniform(0, math.tau)
            sp = rng.uniform(160, 320)
            tiles.append(_Tile(x, y, math.cos(a) * sp, math.sin(a) * sp, self._sprites[tid]))
        return tiles

    def scatter(self) -> None:
        for t in self.tiles:
            a = random.uniform(0, math.tau)
            sp = random.uniform(200, 380)
            t.vx, t.vy = math.cos(a) * sp, math.sin(a) * sp
            t.av += random.uniform(-3, 3)

    def _build_bg(self) -> pygame.Surface:
        bg = pygame.Surface((WIN_W, WIN_H))
        top, bot = (7, 38, 28), (12, 52, 38)
        for y in range(WIN_H):
            p = y / WIN_H
            bg.fill(tuple(int(top[i] + (bot[i] - top[i]) * p) for i in range(3)), (0, y, WIN_W, 1))
        pygame.draw.rect(bg, (40, 130, 98),
                         pygame.Rect(MARGIN - 6, MARGIN - 6,
                                     WIN_W - 2 * (MARGIN - 6), WIN_H - 2 * (MARGIN - 6)),
                         width=3, border_radius=12)
        return bg

    # ------------------------------------------------------------------ physics
    def step(self, dt: float) -> None:
        ts = self.tiles
        for t in ts:
            t.x += t.vx * dt
            t.y += t.vy * dt
            t.ang += t.av * dt
            # walls: bounce at the tile's true rotated extent (not a circle)
            c, s = abs(math.cos(t.ang)), abs(math.sin(t.ang))
            ex = _HW * c + _HH * s
            ey = _HW * s + _HH * c
            if t.x - ex < self.left:
                t.x = self.left + ex; t.vx = abs(t.vx) * RESTITUTION; t.av += random.uniform(-1, 1)
            elif t.x + ex > self.right:
                t.x = self.right - ex; t.vx = -abs(t.vx) * RESTITUTION; t.av += random.uniform(-1, 1)
            if t.y - ey < self.top:
                t.y = self.top + ey; t.vy = abs(t.vy) * RESTITUTION; t.av += random.uniform(-1, 1)
            elif t.y + ey > self.bottom:
                t.y = self.bottom - ey; t.vy = -abs(t.vy) * RESTITUTION; t.av += random.uniform(-1, 1)

        # tile↔tile: oriented-rectangle (SAT) collisions at the real edges
        n = len(ts)
        diag2 = (_HW * 2) ** 2 + (_HH * 2) ** 2
        for i in range(n):
            a = ts[i]
            ca = _corners(a.x, a.y, a.ang)
            aa = (math.cos(a.ang), math.sin(a.ang))
            axes_a = (aa, (-aa[1], aa[0]))
            for j in range(i + 1, n):
                b = ts[j]
                dx, dy = b.x - a.x, b.y - a.y
                if dx * dx + dy * dy > diag2:       # broad-phase reject
                    continue
                cb = _corners(b.x, b.y, b.ang)
                ab = (math.cos(b.ang), math.sin(b.ang))
                hit = _sat(ca, axes_a, cb, (ab, (-ab[1], ab[0])))
                if hit is None:
                    continue
                nx, ny, depth = hit
                if nx * dx + ny * dy < 0:            # normal points a→b
                    nx, ny = -nx, -ny
                a.x -= nx * depth / 2; a.y -= ny * depth / 2
                b.x += nx * depth / 2; b.y += ny * depth / 2
                vn = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
                if vn < 0:
                    a.vx += vn * nx; a.vy += vn * ny
                    b.vx -= vn * nx; b.vy -= vn * ny
                    a.av += random.uniform(-2, 2)
                    b.av += random.uniform(-2, 2)
                ca = _corners(a.x, a.y, a.ang)       # a moved → refresh its corners

    # -------------------------------------------------------------------- draw
    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self._bg, (0, 0))
        # faint contact shadows under the slabs
        for t in self.tiles:
            screen.blit(self._shadow, self._shadow.get_rect(center=(int(t.x) + 5, int(t.y) + DEPTH + 6)))
        for t in self.tiles:
            rot = pygame.transform.rotate(t.sprite, -math.degrees(t.ang))
            cx, cy = int(t.x), int(t.y)
            # extrude the body downward (screen space) to give the tile thickness
            mask = pygame.mask.from_surface(rot)
            body = mask.to_surface(setcolor=(*SIDE, 255), unsetcolor=(0, 0, 0, 0))
            foot = mask.to_surface(setcolor=(*SIDE_DARK, 255), unsetcolor=(0, 0, 0, 0))
            for d in range(DEPTH, 0, -1):
                screen.blit(foot if d > DEPTH - 3 else body,
                            body.get_rect(center=(cx, cy + d)))
            # the printed face sits on top of the body
            screen.blit(rot, rot.get_rect(center=(cx, cy)))


async def main() -> None:
    pygame.init()
    pygame.display.set_caption("crackedMahjong — physics")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    world = PhysicsWorld()

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 1 / 30)      # clamp big hitches
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    world.scatter()
        world.step(dt)
        world.draw(screen)
        screen.blit(font.render(
            "crackedMahjong — physics   ·   space = re-scatter   ·   esc to quit",
            True, pygame.Color("#cfe8dd")), (20, WIN_H - 22))
        pygame.display.flip()
        await asyncio.sleep(0)
    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
