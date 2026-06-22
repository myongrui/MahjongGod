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
import math

import pygame

from cracked.tui_tiles import make_face, FW, FH
from cracked.engine import GameEngine
from cracked.tiles import Wind, tile_name

SCALE = 2                              # base integer scale → crisp pixel tiles
TW, TH = FW * SCALE, FH * SCALE        # 64 x 88 px per tile (at depth 1.0)
# Fixed 16:9 design canvas. The scene is always drawn at this resolution, then
# scaled to fit the (resizable) window with letterboxing — so the layout never
# reflows, it just grows/shrinks with the window.
WIN_W, WIN_H = 1456, 819               # 16:9

# The table as a perspective trapezoid: a wide near edge (your seat, bottom) and a
# narrow far edge (North, top). The four opponents' walls run along the four edges,
# shrinking toward the back so the scene reads as a table seen from your chair.
_NEAR_Y, _FAR_Y = WIN_H - 96, 188
_NL = (96, _NEAR_Y)                     # near-left  table corner
_NR = (WIN_W - 96, _NEAR_Y)            # near-right table corner
# gentle perspective: the far edge is ~82% of the near edge → a square table,
# not a triangle. (near width 1088 → far width 888)
_FL = (196, _FAR_Y)                     # far-left   table corner
_FR = (WIN_W - 196, _FAR_Y)           # far-right  table corner

_GOLD = pygame.Color("#c8a23a")
_DEPTH = 9                             # tile thickness (extruded body)
_SIDE = (206, 190, 142)                # tile body (cream)
_SIDE_DARK = (150, 136, 92)            # shaded bottom of the body
_BODY = (239, 228, 194)               # solid tile body (kills corner gaps)
_TOP_LIT = (245, 237, 212)            # lit top edge of a standing tile
_TOP_DARK = (208, 193, 150)           # top edge near the face seam
# opponent tiles are drawn as real 3D cuboids: an orange back/top, a cream body
# front, and a shaded side — the visible faces differ per seat.
_OPP_TOP = (226, 150, 46)             # tile back/top (orange, MS-style)
_OPP_FRONT = (236, 230, 214)          # tile body front (cream) facing you
_OPP_SIDE = (172, 112, 30)            # shaded side face
_OPP_EDGE = (96, 62, 20)              # cuboid outline
_HAND_S = 1.12                         # your hand tiles are a touch larger (MS look)


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


def _slab_rot(raw: pygame.Surface, scale: float, ang_deg: float):
    """A rotated extruded slab → a thrown/landed tile that shows real depth on top.

    Returns (surface, ax, ay): blit the surface at (cx - ax, cy - ay) to centre
    the *face* on (cx, cy); the body + contact shadow extend below it.
    """
    tw, th = max(1, int(TW * scale)), max(1, int(TH * scale))
    face = _bevel(pygame.transform.scale(raw, (tw, th)))   # nearest → crisp
    pygame.draw.rect(face, (50, 44, 24), face.get_rect(),
                     width=max(1, int(2 * scale)), border_radius=max(2, int(4 * scale)))
    rot = pygame.transform.rotate(face, ang_deg)
    rw, rh = rot.get_size()
    depth = max(3, int(_DEPTH * scale))
    surf = pygame.Surface((rw, rh + depth + 9), pygame.SRCALPHA)
    pygame.draw.ellipse(surf, (0, 0, 0, 45), (int(rw * 0.08), rh + depth - 4, int(rw * 0.84), 9))
    mask = pygame.mask.from_surface(rot)
    body = mask.to_surface(setcolor=(*_SIDE, 255), unsetcolor=(0, 0, 0, 0))
    foot = mask.to_surface(setcolor=(*_SIDE_DARK, 255), unsetcolor=(0, 0, 0, 0))
    for d in range(depth, 0, -1):
        surf.blit(foot if d > depth - 3 else body, (0, d))
    surf.blit(rot, (0, 0))
    return surf, rw / 2, rh / 2


def _standing_slab(raw: pygame.Surface, scale: float):
    """A *standing* tile seen from the player's seat: the face, with its depth
    (the tile's top edge) showing above it. Bodies are solid + square-sided so a
    row of them sits flush with no gaps. Returns (surface, cap) where `cap` is the
    top-edge height — blit at (face_x, face_y - cap) to place the face."""
    tw, th = max(1, int(TW * scale)), max(1, int(TH * scale))
    cap = max(3, int(_DEPTH * 1.7 * scale))
    face = _bevel(pygame.transform.scale(raw, (tw, th)))   # nearest → crisp
    surf = pygame.Surface((tw, th + cap), pygame.SRCALPHA)
    surf.fill((*_BODY, 255), (0, cap, tw, th))             # body fills corner gaps
    for i in range(cap):                                   # top edge: lit→shaded band
        p = i / max(1, cap - 1)
        col = tuple(int(_TOP_LIT[j] + (_TOP_DARK[j] - _TOP_LIT[j]) * p) for j in range(3))
        pygame.draw.line(surf, col, (0, i), (tw - 1, i))
    pygame.draw.line(surf, (120, 106, 66), (0, cap), (tw - 1, cap))   # top/face seam
    surf.blit(face, (0, cap))
    pygame.draw.line(surf, (0, 0, 0, 55), (tw - 1, cap), (tw - 1, th + cap - 1))  # right shade
    return surf, cap


def _lerp(a, b, p):
    return a + (b - a) * p


# --- oriented-rectangle (OBB) collision via the Separating Axis Theorem -------
# Lets the thrown tile bounce off the tiles already resting in the centre pool.

def _corners(cx, cy, ang, hw, hh):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + sx * hw * c - sy * hh * s, cy + sx * hw * s + sy * hh * c)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _sat(ca, axes_a, cb, axes_b):
    """Return (nx, ny, depth) of the minimum-penetration axis, or None if apart."""
    best, bn = 1e18, None
    for ax in (*axes_a, *axes_b):
        amin = amax = None
        for px, py in ca:
            d = px * ax[0] + py * ax[1]
            amin = d if amin is None else min(amin, d)
            amax = d if amax is None else max(amax, d)
        bmin = bmax = None
        for px, py in cb:
            d = px * ax[0] + py * ax[1]
            bmin = d if bmin is None else min(bmin, d)
            bmax = d if bmax is None else max(bmax, d)
        if amax < bmin or bmax < amin:
            return None                      # separating axis → no collision
        ov = min(amax, bmax) - max(amin, bmin)
        if ov < best:
            best, bn = ov, ax
    return bn[0], bn[1], best


class _Fly:
    """A tile in flight: thrown toward the centre, spinning, until it settles."""
    __slots__ = ("tid", "x", "y", "vx", "vy", "ang", "av", "tx", "ty", "d0")

    def __init__(self, tid, x, y, vx, vy, av, tx, ty, d0):
        self.tid, self.x, self.y = tid, x, y
        self.vx, self.vy, self.ang, self.av = vx, vy, 0.0, av
        self.tx, self.ty, self.d0 = tx, ty, d0


class TablePoC:
    def __init__(self) -> None:
        eng = GameEngine(human_seats={int(Wind.EAST)}, seed=7)
        eng.deal()
        concealed = eng.players[int(Wind.EAST)].hand.concealed_tiles_list()
        self.hand = concealed[:13]
        self.drawn = concealed[13] if len(concealed) > 13 else concealed[-1]

        self._raw: dict[int, pygame.Surface] = {}
        self._rot_cache: dict[tuple, tuple] = {}
        self._stand_cache: dict[tuple, tuple] = {}

        # discards keep the angle they landed at → the pool stays un-organised
        self._discards: list[tuple[int, float, float, float]] = []
        self._fly: _Fly | None = None
        self._t = 0.0
        import random
        self._rng = random.Random(1)

        self._bg, self._well = self._build_bg()

        # your hand: 13 tiles flush (no gaps), larger, drawn tile set apart by a gap
        self._htw = int(TW * _HAND_S)
        hand_w = 13 * self._htw
        gap = int(self._htw * 0.5)
        total = hand_w + gap + self._htw
        self._hand_x0 = (WIN_W - total) // 2
        self._hand_y = WIN_H - int(TH * _HAND_S) - 30
        self._drawn_home = (self._hand_x0 + hand_w + gap, self._hand_y)

    # ------------------------------------------------------------------ assets
    def _raw_face(self, tid: int) -> pygame.Surface:
        s = self._raw.get(tid)
        if s is None:
            s = self._raw[tid] = _surface_from_face(make_face(tid))
        return s

    def slab_rot(self, tid, scale: float, ang_deg: float):
        key = (tid, round(scale, 2), round(ang_deg))
        s = self._rot_cache.get(key)
        if s is None:
            s = self._rot_cache[key] = _slab_rot(self._raw_face(tid), scale, ang_deg)
        return s

    def _blit_slab_rot(self, screen, tid, scale, ang_deg, cx, cy) -> None:
        surf, ax, ay = self.slab_rot(tid, scale, ang_deg)
        screen.blit(surf, (int(cx - ax), int(cy - ay)))

    def standing(self, tid, scale: float = 1.0):
        key = (tid, round(scale, 2))
        s = self._stand_cache.get(key)
        if s is None:
            s = self._stand_cache[key] = _standing_slab(self._raw_face(tid), scale)
        return s

    def _blit_standing(self, screen, tid, scale, fx, fy) -> None:
        """Blit a standing tile so its face top-left lands on (fx, fy)."""
        surf, cap = self.standing(tid, scale)
        screen.blit(surf, (int(fx), int(fy - cap)))

    def _tile_box(self, screen, c, A, B, C) -> None:
        """One opponent tile as a 3D cuboid centred at `c`, drawn like a stair step:
        A = the tile's run (along the hand row), B = the white riser (front, toward you),
        C = the orange tread depth (toward the table centre). Visible faces: orange tread
        (A×C), white riser (A×B) and a shaded side (B×C) — so every seat shows real tiles
        seen from its own angle, not one rotated sprite."""
        ox = c[0] - 0.5 * (A[0] + B[0] + C[0])
        oy = c[1] - 0.5 * (A[1] + B[1] + C[1])

        def P(i, j, k):
            return (ox + i * A[0] + j * B[0] + k * C[0], oy + i * A[1] + j * B[1] + k * C[1])

        tread = [P(0, 0, 0), P(1, 0, 0), P(1, 0, 1), P(0, 0, 1)]   # A×C orange top
        riser = [P(0, 0, 0), P(1, 0, 0), P(1, 1, 0), P(0, 1, 0)]   # A×B white front
        side = [P(1, 0, 0), P(1, 1, 0), P(1, 1, 1), P(1, 0, 1)]    # B×C shaded side
        for poly, col in ((side, _OPP_SIDE), (riser, _OPP_FRONT), (tread, _OPP_TOP)):
            pygame.draw.polygon(screen, col, poly)
            pygame.draw.polygon(screen, _OPP_EDGE, poly, 1)

    @staticmethod
    def _line_pts(a, b, sa, sb, n):
        """n (centre, scale) samples from a→b (far→near), used to place a hand's tiles."""
        return [((_lerp(a[0], b[0], t), _lerp(a[1], b[1], t)), _lerp(sa, sb, t))
                for t in ((i + 0.5) / n for i in range(n))]

    def _back_tile(self, screen, c, s, sgn) -> None:
        """A left/right opponent tile-back as a 3D block: a wide orange top facing the
        table centre, with a cream body edge along its front + outer side toward you.
        `sgn`=+1 for the left hand, -1 for the right (mirrored)."""
        cx, cy = c
        Rx, Ry = sgn * 44 * s, -9 * s         # toward centre (orange-top width)
        Dx, Dy = -sgn * 5 * s, 24 * s          # down the column (stacking depth)
        hy = 16 * s                            # cream body thickness (extruded down)
        p0 = (cx - Rx / 2 - Dx / 2, cy - Ry / 2 - Dy / 2)   # outer-far corner
        p1 = (cx + Rx / 2 - Dx / 2, cy + Ry / 2 - Dy / 2)   # centre-far
        p2 = (cx + Rx / 2 + Dx / 2, cy + Ry / 2 + Dy / 2)   # centre-near
        p3 = (cx - Rx / 2 + Dx / 2, cy - Ry / 2 + Dy / 2)   # outer-near
        top = [p0, p1, p2, p3]
        outer = [p0, p3, (p3[0], p3[1] + hy), (p0[0], p0[1] + hy)]
        front = [p3, p2, (p2[0], p2[1] + hy), (p3[0], p3[1] + hy)]
        for poly, col in ((outer, _OPP_FRONT), (front, _OPP_FRONT), (top, _OPP_TOP)):
            pygame.draw.polygon(screen, col, poly)
            pygame.draw.polygon(screen, _OPP_EDGE, poly, 1)

    def _build_bg(self):
        bg = pygame.Surface((WIN_W, WIN_H))
        # the room: warm amber-lit back wall fading to a darker wood floor (MS vibe)
        wall, floor = (74, 54, 38), (26, 18, 13)
        for y in range(WIN_H):
            p = y / WIN_H
            bg.fill(tuple(int(_lerp(wall[i], floor[i], p)) for i in range(3)), (0, y, WIN_W, 1))

        # wooden table rim, then the felt inset on top of it
        def _expand(pts, dx, dy):
            cx = sum(p[0] for p in pts) / 4
            cy = sum(p[1] for p in pts) / 4
            return [(p[0] + (1 if p[0] > cx else -1) * dx,
                     p[1] + (1 if p[1] > cy else -1) * dy) for p in pts]

        felt = [_FL, _FR, _NR, _NL]
        rim = _expand(felt, 26, 22)
        pygame.draw.polygon(bg, (28, 30, 38), _expand(felt, 40, 34))   # rim drop-shadow
        pygame.draw.polygon(bg, (96, 66, 38), rim)                     # wood rim
        pygame.draw.polygon(bg, (70, 47, 26), rim, 3)
        # felt with a soft vertical gradient (lighter at the back → depth)
        ys = [p[1] for p in felt]
        y0, y1 = min(ys), max(ys)
        for y in range(int(y0), int(y1)):
            t = (y - y0) / max(1, y1 - y0)
            xl = _lerp(_FL[0], _NL[0], t)
            xr = _lerp(_FR[0], _NR[0], t)
            col = tuple(int(_lerp(a, b, t)) for a, b in ((20, 11), (74, 54), (54, 38)))
            pygame.draw.line(bg, col, (int(xl), y), (int(xr), y))
        pygame.draw.polygon(bg, (44, 120, 92), felt, 2)

        # centre discard well: a flattened ellipse (reads as round on the tilted table)
        cx = WIN_W // 2
        cy = int((_NEAR_Y + _FAR_Y) / 2) + 36
        well = pygame.Rect(cx - 270, cy - 130, 540, 260)
        pygame.draw.ellipse(bg, (10, 46, 34), well)
        pygame.draw.ellipse(bg, _GOLD, well, 2)

        # baked vignette: darken the corners so attention falls on the table (MS lighting)
        vg = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        steps = 60
        for i in range(steps):
            a = int(150 * (i / steps) ** 2)
            inset = i * 9
            pygame.draw.rect(vg, (0, 0, 0, a),
                             (inset, inset, WIN_W - 2 * inset, WIN_H - 2 * inset),
                             width=10, border_radius=40)
        bg.blit(vg, (0, 0))
        return bg, well

    # ------------------------------------------------------------------ update
    _DAMP = 0.90               # per-(1/60)s velocity decay → the throw settles
    _DISC_S = 0.8              # discard tile scale in the centre well

    def update(self, dt: float) -> None:
        if self._fly is not None:
            self._step_throw(dt)
            return
        self._t += dt
        if self._t >= 1.2:
            self._launch_throw()
            self._t = 0.0

    def _launch_throw(self) -> None:
        """Fling the drawn tile toward a random spot in the well, spinning."""
        w = self._well
        tx = self._rng.uniform(w.left + 40, w.right - 40)
        ty = self._rng.uniform(w.top + 30, w.bottom - 30)
        sx, sy = self._drawn_home[0] + TW / 2, self._drawn_home[1] + TH / 2
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy) or 1.0
        # v0 chosen so the damped path lands ≈ on target: Σ v0·dt·damp^n → dist
        v0 = dist * (1 - self._DAMP) * 60.0
        vx, vy = dx / dist * v0, dy / dist * v0
        av = self._rng.uniform(8.0, 16.0) * self._rng.choice((-1, 1))
        self._fly = _Fly(self.drawn, sx, sy, vx, vy, av, tx, ty, dist)

    def _step_throw(self, dt: float) -> None:
        f = self._fly
        if f is None:
            return
        k = self._DAMP ** (dt * 60.0)
        f.vx *= k; f.vy *= k; f.av *= k
        f.x += f.vx * dt; f.y += f.vy * dt; f.ang += f.av * dt
        # keep it inside the well — bounce softly off the rim
        w = self._well
        if f.x < w.left + 16:
            f.x = w.left + 16; f.vx = abs(f.vx) * 0.4
        elif f.x > w.right - 16:
            f.x = w.right - 16; f.vx = -abs(f.vx) * 0.4
        if f.y < w.top + 14:
            f.y = w.top + 14; f.vy = abs(f.vy) * 0.4
        elif f.y > w.bottom - 14:
            f.y = w.bottom - 14; f.vy = -abs(f.vy) * 0.4
        self._collide_pool(f)
        if math.hypot(f.vx, f.vy) < 32.0:               # come to rest → land it
            self._discards.append((f.tid, f.x, f.y, math.degrees(f.ang)))
            if len(self._discards) > 12:
                self._discards.pop(0)
            self._fly = None

    def _collide_pool(self, f: _Fly) -> None:
        """Bounce the thrown tile off the tiles already resting in the pool (OBB)."""
        fs = self._fly_scale()
        fhw, fhh = TW * fs / 2, TH * fs / 2
        dhw, dhh = TW * self._DISC_S / 2, TH * self._DISC_S / 2
        reach2 = (fhw + fhh + dhw + dhh) ** 2
        fc = _corners(f.x, f.y, f.ang, fhw, fhh)
        fax = (math.cos(f.ang), math.sin(f.ang))
        faxes = (fax, (-fax[1], fax[0]))
        for _tid, dx, dy, dang_deg in self._discards:
            if (dx - f.x) ** 2 + (dy - f.y) ** 2 > reach2:
                continue                                # broad-phase reject
            dang = math.radians(dang_deg)
            dc = _corners(dx, dy, dang, dhw, dhh)
            dax = (math.cos(dang), math.sin(dang))
            hit = _sat(fc, faxes, dc, (dax, (-dax[1], dax[0])))
            if hit is None:
                continue
            nx, ny, depth = hit
            if nx * (f.x - dx) + ny * (f.y - dy) < 0:    # normal points pool → flyer
                nx, ny = -nx, -ny
            f.x += nx * depth; f.y += ny * depth         # push out of penetration
            vn = f.vx * nx + f.vy * ny
            if vn < 0:                                   # moving in → reflect (e=0.45)
                f.vx -= 1.45 * vn * nx
                f.vy -= 1.45 * vn * ny
                f.av += self._rng.uniform(-4, 4)
            fc = _corners(f.x, f.y, f.ang, fhw, fhh)     # flyer moved → refresh

    def _fly_scale(self) -> float:
        """Thrown tile shrinks as it travels 'into' the table (front → mid depth)."""
        f = self._fly
        if f is None:
            return 1.0
        cur = math.hypot(f.tx - f.x, f.ty - f.y)
        p = max(0.0, min(1.0, 1.0 - cur / f.d0))
        return _lerp(1.0, 0.85, p)

    # -------------------------------------------------------------------- draw
    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        screen.blit(self._bg, (0, 0))

        disc_s = self._DISC_S
        cx = WIN_W // 2
        # North (opposite): a flush row of standing tile-backs — tall orange + white base
        ns = 0.82
        nbw = int(34 * ns)
        for i in range(13):
            nc = (cx - 13 * nbw // 2 + i * nbw + nbw // 2, _FAR_Y + 40)
            self._tile_box(screen, nc, (nbw, 0), (0, 13 * ns), (0, -42 * ns))
        # West (left): 3D tile-backs as a steep column down the left rim — orange top
        # facing the centre, cream front + outer edge toward you
        for c, s in self._line_pts((262, _FAR_Y + 56), (166, _NEAR_Y - 46), 0.84, 1.04, 13):
            self._back_tile(screen, c, s, +1)
        # East (right): mirror of West
        for c, s in self._line_pts((WIN_W - 262, _FAR_Y + 56), (WIN_W - 166, _NEAR_Y - 46), 0.84, 1.04, 13):
            self._back_tile(screen, c, s, -1)

        # discards lie flat in the centre well — each at its landed angle
        for tid, x, y, ang in self._discards:
            self._blit_slab_rot(screen, tid, disc_s, ang, x, y)

        # the drawn tile, mid-throw, lies flat as it spins toward the well
        if self._fly is not None:
            f = self._fly
            self._blit_slab_rot(screen, f.tid, self._fly_scale(), math.degrees(f.ang), f.x, f.y)

        # your hand — foreground, standing, larger, flush
        for i, tid in enumerate(self.hand):
            self._blit_standing(screen, tid, _HAND_S, self._hand_x0 + i * self._htw, self._hand_y)
        if self._fly is None:                          # drawn tile rests apart on the right
            self._blit_standing(screen, self.drawn, _HAND_S, *self._drawn_home)

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
    # SCALED|RESIZABLE: SDL renders at the fixed 16:9 logical size and scales it to the
    # window itself (aspect-preserved, letterboxed). No manual rescaling and no resize
    # handling — which avoids the set_mode-in-VIDEORESIZE feedback loop / flicker.
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SCALED | pygame.RESIZABLE)
    font = pygame.font.Font(None, 24)
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
