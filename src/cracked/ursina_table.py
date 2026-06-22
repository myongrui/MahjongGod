"""
Ursina 3D front-end — proof-of-concept for the mahjong table.

A *real* 3D scene (vs the pygame fake-2.5D): one tile is modelled as a cuboid with a
cream body, a jade back and an optional printed face. The four hands are placed around
the table with real transforms and a single perspective camera renders them all at
once — so the per-seat tile-orientation problems the 2D renderer keeps fighting simply
don't arise: each hand is the same row of tiles, just rotated 90° about the table and
viewed through one camera. It reuses the same `GameEngine` deal and the `tui_tiles`
pixel-art faces (rasterised into textures).

    pip install -e ".[threed]"
    python -m cracked.ursina_table        # drag to orbit, scroll to zoom, esc to quit

This is a deliberate sketch to compare against the pygame version. Colours use
`color.hsv` (Ursina 8.x); on older Ursina the printed-face `Texture(pil_image)` step or
the lit shader may need a one-line tweak — both degrade gracefully — but the scene
structure is the point.
"""
from __future__ import annotations

import math
import random
import sys

try:
    from ursina import (Ursina, Entity, color, camera, window, Vec3, EditorCamera,
                        DirectionalLight, AmbientLight, time, curve, destroy)
    _HAVE_URSINA = True
except Exception:                       # ursina not installed → module still imports
    _HAVE_URSINA = False

try:
    # lit_with_shadows_shader is the one that actually works in this Ursina build: it
    # shades from a DirectionalLight (point lights / colored_lights_shader render flat).
    # Note: directional light has no falloff, so this lights the whole table evenly.
    from ursina.shaders import lit_with_shadows_shader as _LIT
except Exception:
    _LIT = None

from cracked.engine import GameEngine
from cracked.tiles import Wind
from cracked.tui_tiles import make_face, FW, FH

# tile size + table layout, in Ursina world units
TW, TH, TD = 0.5, 0.72, 0.38            # tile width / height / thickness
STEP = TW + 0.05                        # spacing between tiles in a hand
EDGE = 5.4                              # distance of each hand from the table centre
WALL_R = 4.3                            # the draw-wall ring sits just inside the hands
N = 13                                  # tiles per concealed hand
WALL_N = 17                             # tiles per wall side (built two courses high)
FELT = 15                               # felt table size (square)

# Fake "lamp above the centre" pool: since Ursina's point lights don't work here, we
# tint each piece darker the further it sits from the table centre (rotation-invariant,
# so the same factor applies to every seat). This rides on top of the directional shading.
POOL_R = 7.0                            # radius over which the pool fades out
POOL_FLOOR = 0.3               # dimmest the rim gets (not pure black)

# discard into the centre (a la the pygame physics demo). Two styles:
#   "slide" — the tile slides flat across the felt, spinning, and friction stops it (active)
#   "arc"   — it tosses up, tumbles under gravity, and lands face-up (kept for reference)
THROW_G = -16.0                         # arc gravity (world units / s²)
THROW_T = 0.7                           # arc flight time used to aim the toss
SLIDE_A = 7.0                           # slide friction deceleration (world units / s²)
THROW_REST_Y = TD / 2 + 0.01            # resting height of a tile lying flat

_tex_cache: dict = {}


def _sat2d(a, b):
    """Oriented-box overlap in the flat XZ plane (the tiles lie face-up). `a`/`b` are
    (cx, cz, yaw, hw, hh). Returns (nx, nz, depth) of the minimum-penetration axis with
    the normal pointing a→b, or None if the boxes are apart (Separating Axis Theorem)."""
    acx, acz, ayaw, ahw, ahh = a
    bcx, bcz, byaw, bhw, bhh = b
    ac, asn = math.cos(ayaw), math.sin(ayaw)
    bc, bsn = math.cos(byaw), math.sin(byaw)
    dx, dz = bcx - acx, bcz - acz
    best, bn = 1e18, None
    for lx, lz in ((ac, asn), (-asn, ac), (bc, bsn), (-bsn, bc)):
        ra = ahw * abs(ac * lx + asn * lz) + ahh * abs(-asn * lx + ac * lz)
        rb = bhw * abs(bc * lx + bsn * lz) + bhh * abs(-bsn * lx + bc * lz)
        ov = ra + rb - abs(dx * lx + dz * lz)
        if ov <= 0:
            return None                     # found a separating axis → no collision
        if ov < best:
            best, bn = ov, (lx, lz)
    nx, nz = bn
    if dx * nx + dz * nz < 0:                # make the normal point a→b
        nx, nz = -nx, -nz
    return nx, nz, best


def _pool(dist: float) -> float:
    """Brightness multiplier for a piece `dist` from the table centre: 1 at the middle,
    falling off to POOL_FLOOR toward the edge."""
    return max(POOL_FLOOR, min(1.0, 1.0 - (dist / POOL_R) ** 2))


def _ivory(f: float = 1.0):
    return color.hsv(45, 0.20, 0.72 * f)   # warm bone body, dimmed by the pool factor


def _jade(f: float = 1.0):
    return color.hsv(158, 0.62, 0.44 * f)  # MS-style green back


def _felt(f: float = 1.0):
    return color.hsv(150, 0.45, 0.20 * f)  # table cloth


def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _face_texture(tid: int):
    """Rasterise our pixel-art tile face into an Ursina texture (cached)."""
    if tid in _tex_cache:
        return _tex_cache[tid]
    tex = None
    try:
        from PIL import Image
        from ursina import Texture
        img = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
        px = img.load()
        face = make_face(tid)
        for y in range(FH):
            for x in range(FW):
                c = face[y][x]
                if c is not None:
                    px[x, y] = (*_hex_rgb(c), 255)
        img = img.resize((FW * 10, FH * 10), Image.NEAREST)   # crisp pixels
        tex = Texture(img)
    except Exception:
        tex = None                       # fall back to a plain tile if it fails
    _tex_cache[tid] = tex
    return tex


def _felt_texture():
    """A radial-gradient cloth texture: bright green at the centre fading dark at the
    rim, so the felt itself shows the pool of light (cached)."""
    if "felt" in _tex_cache:
        return _tex_cache["felt"]
    tex = None
    try:
        from PIL import Image
        from ursina import Texture
        size = 128
        base = _felt(1.0)
        br, bg, bb = int(base[0] * 255), int(base[1] * 255), int(base[2] * 255)
        img = Image.new("RGB", (size, size))
        px = img.load()
        for y in range(size):
            for x in range(size):
                wx = (x / size - 0.5) * FELT
                wz = (y / size - 0.5) * FELT
                f = _pool(math.hypot(wx, wz))
                px[x, y] = (int(br * f), int(bg * f), int(bb * f))
        tex = Texture(img)
    except Exception:
        tex = None
    _tex_cache["felt"] = tex
    return tex


def make_tile(tid=None, reveal=False, parent=None, bright: float = 1.0):
    """A 3D mahjong tile: cream cuboid body, a jade back, and an optional printed face.
    The printed face is on the +z side (it points toward the tile's owner); the jade
    back is on the -z side (what the other players see). `bright` dims it for the pool."""
    tile = Entity(parent=parent)
    Entity(parent=tile, model="cube", scale=(TW, TH, TD), color=_ivory(bright), shader=_LIT)
    Entity(parent=tile, model="quad", scale=(TW * 0.98, TH * 0.98),
           z=-TD / 2 - 0.001, rotation_y=180, color=_jade(bright), shader=_LIT, double_sided=True)
    if reveal and tid is not None:
        tex = _face_texture(tid)
        # same lit shader as the body + a neutral grey tint at the pool brightness, so the
        # printed face shades and dims exactly like the tile instead of looking pasted on
        Entity(parent=tile, model="quad", scale=(TW * 0.94, TH * 0.94),
               z=TD / 2 + 0.001, texture=tex, shader=_LIT,
               color=color.hsv(0, 0, bright) if tex is not None else _ivory(bright),
               double_sided=True)
    return tile


def place_hand(tiles, seat: int, reveal: bool):
    """13 standing tiles in a row at `seat` (0 = you/front, then clockwise). The whole
    hand is one entity rotated 90°·seat about the table, so every seat reuses the exact
    same layout — the camera does the rest. The +180 puts seat 0 (you) at the front edge
    with its printed faces turned toward the camera."""
    holder = Entity(rotation_y=seat * 90 + 180)
    x0 = -(N - 1) / 2 * STEP
    for i, tid in enumerate(tiles):
        lx = x0 + i * STEP
        t = make_tile(tid, reveal, parent=holder, bright=_pool(math.hypot(lx, EDGE)))
        t.position = (lx, TH / 2, EDGE)
    return holder


def place_wall(seat: int):
    """The draw wall in front of `seat`: WALL_N face-down tiles, two courses high, with
    a jade back along the top — the live wall players draw from. Rotated per seat so the
    four walls form a square ring just inside the hands."""
    holder = Entity(rotation_y=seat * 90 + 180)        # match place_hand: you at the front
    x0 = -(WALL_N - 1) / 2 * TW
    for i in range(WALL_N):
        b = _pool(math.hypot(x0 + i * TW, WALL_R))
        for course in (0, 1):
            Entity(parent=holder, model="cube",
                   scale=(TW * 0.95, TD * 0.9, TH * 0.92),
                   position=(x0 + i * TW, TD * 0.5 + course * TD, WALL_R),
                   color=_ivory(b), shader=_LIT)
    # jade backs facing up along the top course
    Entity(parent=holder, model="cube",
           scale=(WALL_N * TW, 0.02, TH * 0.9),
           position=(0, 2 * TD + 0.01, WALL_R),
           color=_jade(_pool(WALL_R)), shader=_LIT)
    return holder


def build_scene(reveal_all: bool = False):
    """Felt table + the four dealt hands. Returns nothing; just populates the scene."""
    # dim felt slab (for the table's thickness/edge) + a radial-gradient top so the
    # centre of the cloth glows and the rim falls dark — the lamp's pool on the felt
    Entity(model="cube", scale=(FELT, 0.3, FELT), position=(0, -0.15, 0),
           color=_felt(POOL_FLOOR), shader=_LIT)
    Entity(model="plane", scale=(FELT, 1, FELT), position=(0, 0.005, 0),
           texture=_felt_texture(), color=color.white)

    eng = GameEngine(human_seats={int(Wind.EAST)}, seed=7)
    eng.deal()
    # you sit at East (front); going clockwise on screen: right=South, across=West, left=North
    seats = {0: int(Wind.EAST), 1: int(Wind.SOUTH), 2: int(Wind.WEST), 3: int(Wind.NORTH)}
    for seat, wind in seats.items():
        hand = eng.players[wind].hand.concealed_tiles_list()[:N]
        place_hand(hand, seat, reveal=reveal_all or seat == 0)
        # place_wall(seat)  # draw walls removed for now (place_wall kept for later)


if _HAVE_URSINA:

    class ThrowManager(Entity):
        """Every ~1.6 s sends a tile into the centre. `mode="slide"` (active) sends it
        flat across the felt, spinning, with friction bringing it to rest; `mode="arc"`
        keeps the older toss-and-tumble (lands face-up). The 3D take on the physics demo."""

        def __init__(self, mode: str = "slide", max_pile: int = 28):
            super().__init__()
            self.mode = mode
            self.max_pile = max_pile
            self._t = 1.0
            self._slide: list = []         # [entity, vx, vz, yaw_rate, launch_speed]
            self._pile: list = []          # resting discards: [entity, x, z, yaw_rad]
            self._arc: list = []           # [entity, velocity, angular_velocity]

        def update(self):
            self._t += time.dt
            if self._t >= 1.6:
                self._t = 0.0
                (self._toss_slide if self.mode == "slide" else self._toss_arc)()
            self._advance_slide()
            self._advance_arc()

        # -- slide: flat across the felt, spinning, colliding with the pile (active) ---
        def _toss_slide(self):
            tid = random.randint(0, 33)
            start = Vec3(random.uniform(-2.0, 2.0), THROW_REST_Y, -EDGE + 1.0)
            target = Vec3(random.uniform(-1.6, 1.6), THROW_REST_Y, random.uniform(-1.6, 1.6))
            e = Entity(position=start, rotation=Vec3(-90, random.uniform(0, 360), 0))
            make_tile(tid, reveal=True, parent=e, bright=0.9)   # flat, face-up
            dx, dz = target.x - start.x, target.z - start.z
            dist = math.hypot(dx, dz) or 0.1
            speed = math.sqrt(2 * SLIDE_A * dist) * 1.15        # a little extra for bounces
            yaw = random.uniform(160, 320) * random.choice((-1, 1))
            self._slide.append([e, dx / dist * speed, dz / dist * speed, yaw, speed])

        def _advance_slide(self):
            dt = time.dt
            hw, hh = TW / 2, TH / 2
            for f in self._slide[:]:
                e, vx, vz, yaw, v0 = f
                sp = math.hypot(vx, vz)
                nsp = sp - SLIDE_A * dt
                if nsp <= 0.06:                                 # friction won it → come to rest
                    self._pile.append([e, e.x, e.z, math.radians(e.rotation_y)])
                    self._slide.remove(f)
                    self._cap_pile()
                    continue
                vx *= nsp / sp; vz *= nsp / sp                  # apply friction to the velocity
                e.position += Vec3(vx * dt, 0, vz * dt)
                e.rotation_y += yaw * (nsp / v0) * dt           # spin fades with the slide
                # bump off the tiles already resting in the centre
                myaw = math.radians(e.rotation_y)
                for p in self._pile:
                    hit = _sat2d((p[1], p[2], p[3], hw, hh), (e.x, e.z, myaw, hw, hh))
                    if hit is None:
                        continue
                    nx, nz, depth = hit
                    e.x += nx * depth; e.z += nz * depth        # push out of the overlap
                    vn = vx * nx + vz * nz
                    if vn < 0:                                  # moving in → reflect (e≈0.6)
                        vx -= 1.6 * vn * nx; vz -= 1.6 * vn * nz
                        yaw += random.uniform(-120, 120)
                    myaw = math.radians(e.rotation_y)
                f[1], f[2], f[3] = vx, vz, yaw

        def _cap_pile(self):
            while len(self._pile) > self.max_pile:
                destroy(self._pile.pop(0)[0])                   # retire the oldest discard

        # -- arc: tossed up + tumbling, lands face-up (kept for reference) ------------
        def _toss_arc(self):
            tid = random.randint(0, 33)
            start = Vec3(random.uniform(-0.4, 0.4), TH / 2, -EDGE + 0.6)
            target = Vec3(random.uniform(-1.6, 1.6), THROW_REST_Y, random.uniform(-1.6, 1.6))
            e = Entity(position=start)
            make_tile(tid, reveal=True, parent=e, bright=0.9)
            vel = (target - start) / THROW_T - Vec3(0, THROW_G, 0) * (0.5 * THROW_T)
            avel = Vec3(random.uniform(-220, 220), random.uniform(-220, 220),
                        random.uniform(-340, 340))
            self._arc.append([e, vel, avel])

        def _advance_arc(self):
            dt = time.dt
            for f in self._arc[:]:
                e, vel, avel = f
                vel += Vec3(0, THROW_G, 0) * dt
                e.position += vel * dt
                e.rotation += avel * dt
                f[1] = vel
                if e.y <= THROW_REST_Y and vel.y < 0:
                    e.y = THROW_REST_Y
                    e.animate_rotation((-90, random.uniform(0, 360), 0),
                                       duration=0.18, curve=curve.out_cubic)
                    destroy(e, delay=5)
                    self._arc.remove(f)


def main():
    if not _HAVE_URSINA:
        sys.exit('Ursina is not installed — run:  pip install -e ".[threed]"')

    app = Ursina(title="crackedMahjong — Ursina 3D PoC")
    window.color = color.rgb32(6, 7, 12)             # dark room

    # a wide, very dark floor so the scene reads as a room rather than a void
    Entity(model="plane", scale=(60, 1, 60), y=-0.31,
           color=color.hsv(220, 0.25, 0.05), shader=_LIT)
    build_scene(reveal_all=False)                     # set True for the exposed view

    # one overhead key light (angled so it shades + casts a shadow) + a low fill so the
    # dark sides aren't pure black. lit_with_shadows_shader reads this DirectionalLight.
    AmbientLight(color=color.hsv(0, 0, 0.18))
    sun = DirectionalLight()
    sun.look_at(Vec3(0.45, -1.0, 0.55))

    ThrowManager()                       # auto-tosses a tile into the centre periodically

    EditorCamera(rotation=(30, 0, 0))    # orbit with the mouse, scroll to zoom
    camera.z = -26                       # pull back to frame the (now larger) table
    camera.fov = 50
    app.run()


if __name__ == "__main__":
    main()
