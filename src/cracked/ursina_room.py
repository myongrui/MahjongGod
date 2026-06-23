"""
Standalone dark-room demo (Ursina) — an Inscryption-style cabin you can orbit around.

A near-black wooden room lit by a single candle on the table. Point lights don't work in
this Ursina build, so the candle's glow is *baked* into each surface's texture (brightest
nearest the flame, falling to black at the edges) and the whole room is rendered unlit; a
per-frame tint makes the glow breathe with the flickering flame. Self-contained — it does
not touch the game's shared `setup_room()`.

    pip install -e ".[threed]"
    python -m cracked.ursina_room        # drag to orbit, scroll to zoom, esc to quit
"""
from __future__ import annotations

import math
import random
import sys

import numpy as np

try:
    from ursina import Ursina, Entity, color, camera, window, EditorCamera, time
    _HAVE_URSINA = True
except Exception:                       # ursina not installed → module still imports
    _HAVE_URSINA = False

ROOM = 26.0            # floor is ROOM × ROOM
WALL_H = 11.0          # ceiling height
CANDLE_Y = 1.6         # flame height above the table-top
POOL_R = 24.0          # reach of the candlelight before it fades to black
POOL_FLOOR = 0.40     # darkest any surface gets (≈ black at the rim)
TEX = 256              # texture resolution (px) — low-res for the PSX look (shader posterizes)
OAK = np.array([165.0, 105.0, 55.0])   # warm oak base colour


def _shade(d):
    """Brightness multiplier for points `d` world-units from the flame (array-aware): 1 at
    the flame, easing to POOL_FLOOR past POOL_R."""
    return np.clip(1.0 - (d / POOL_R) ** 2, POOL_FLOOR, 1.0) ** 1.4


def _grain(planks, vertical):
    """A high-res grainy timber brightness field (0..~1.3): planks with per-board colour,
    long flowing grain, fine streaks, and dense speckle noise for a rough, grainy look."""
    rng = np.random.default_rng(7)                         # stable grain between runs
    i = np.arange(TEX)
    U, V = np.meshgrid(i, i)                                # U across, V along the boards
    if vertical:
        U, V = V, U
    plank_w = TEX / planks
    pi = (U // plank_w).astype(int) % planks
    base = rng.uniform(0.7, 1.0, planks)[pi]
    streak = rng.uniform(0, math.tau, planks)[pi]
    # grain streaks run ALONG the board (vary across U), gently wavering down its length (V)
    warp = 2.5 * np.sin(V * 0.018 + streak) + 1.3 * np.sin(V * 0.005 + 2 * streak)
    g = 0.85 + 0.15 * np.sin(U * 0.16 + warp + streak)     # main lengthwise grain lines
    g *= 0.93 + 0.07 * np.sin(U * 0.55 + 1.7 * warp)       # finer grain lines
    edge = U % plank_w
    g *= np.where(np.minimum(edge, plank_w - edge) < TEX * 0.004, 0.55, 1.0)  # darker seams between boards
    g *= rng.normal(1.0, 0.04, (TEX, TEX))                 # fine speckle (the graininess)
    return base * np.clip(g, 0.2, 1.4)


def _to_texture(rgb):
    from PIL import Image
    from ursina import Texture
    tex = Texture(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB"))
    tex.filtering = None                                    # point sampling — chunky unfiltered texels
    return tex


def _horizontal_texture(planks, perp):
    """Wood floor/ceiling with the candle's radial pool baked in. `perp` is the surface's
    vertical distance from the flame (so the floor and ceiling fade correctly)."""
    g = _grain(planks, vertical=False)
    i = np.arange(TEX)
    X, Y = np.meshgrid(i, i)
    wx = (X / TEX - 0.5) * ROOM
    wz = (Y / TEX - 0.5) * ROOM
    f = g * _shade(np.sqrt(wx * wx + wz * wz + perp * perp))
    return _to_texture(f[..., None] * OAK)


def _wall_texture(planks):
    """Vertical timber boards with the candle glow baked in — brightest at the point of the
    wall nearest the flame (centre, candle height), fading out and up into the dark."""
    g = _grain(planks, vertical=True)
    i = np.arange(TEX)
    X, Y = np.meshgrid(i, i)
    wx = (X / TEX - 0.5) * ROOM                             # along the wall
    wy = (1 - Y / TEX) * WALL_H                             # height up the wall (image y=0 is top)
    dy = wy - CANDLE_Y
    perp = ROOM / 2                                         # candle sits at room centre
    f = g * _shade(np.sqrt(wx * wx + dy * dy + perp * perp))
    return _to_texture(f[..., None] * OAK)


def build_shell(floor_y=-0.1):
    """The cabin shell — wood floor, ceiling, and four walls (unlit; lighting is baked into
    the textures). Plank counts scale with room size so boards keep a sane real-world width.
    Returns the list of shell entities."""
    h = ROOM / 2
    fplanks = max(6, round(ROOM / 2.6))                     # floor/ceiling board count
    wplanks = max(6, round(ROOM / 3.5))                     # wall board count
    wall_tex = _wall_texture(wplanks)
    shell = [
        Entity(model="cube", scale=(ROOM, 0.2, ROOM), y=floor_y,
               texture=_horizontal_texture(fplanks, CANDLE_Y)),
        Entity(model="cube", scale=(ROOM, 0.2, ROOM), y=WALL_H,
               texture=_horizontal_texture(fplanks, WALL_H - CANDLE_Y)),
    ]
    # four timber walls (thin cubes — render from inside regardless of face normals)
    for pos, sc in (((0, WALL_H / 2, h), (ROOM, WALL_H, 0.2)),
                    ((0, WALL_H / 2, -h), (ROOM, WALL_H, 0.2)),
                    ((h, WALL_H / 2, 0), (0.2, WALL_H, ROOM)),
                    ((-h, WALL_H / 2, 0), (0.2, WALL_H, ROOM))):
        shell.append(Entity(model="cube", scale=sc, position=pos, texture=wall_tex))
    return shell


def build_room():
    """Cabin shell + a candle on the table (all unlit; brightness is baked). Returns the
    flame entity and the list of shell entities the Flicker tints each frame."""
    shell = build_shell()
    # candle: a pale wax stub on a dark holder, topped by a warm flickering flame
    Entity(model="cylinder", scale=(0.45, 0.18, 0.45), position=(0, 0.09, 0),
           color=color.hsv(30, 0.4, 0.35))                       # holder
    Entity(model="cylinder", scale=(0.22, CANDLE_Y - 0.4, 0.22), position=(0, (CANDLE_Y - 0.4) / 2 + 0.18, 0),
           color=color.hsv(40, 0.18, 0.85))                      # wax
    flame = Entity(model="sphere", scale=(0.16, 0.34, 0.16), position=(0, CANDLE_Y, 0),
                   color=color.hsv(35, 0.85, 1.0))               # flame
    return flame, shell


# Fullscreen post-process: posterize + ordered dither + scanlines + vignette, applied to
# the WHOLE rendered scene (tiles included) for one unified PSX/CRT look.
RETRO_FRAG = '''
#version 430
uniform sampler2D tex;
in vec2 uv;
out vec4 color;

const float L = 6.0;                 // colour levels per channel
const float bayer[16] = float[](
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0);

void main() {
    vec3 c = texture(tex, uv).rgb;
    // ordered dither, then posterize to L levels
    int bx = int(mod(gl_FragCoord.x, 4.0));
    int by = int(mod(gl_FragCoord.y, 4.0));
    float d = (bayer[by * 4 + bx] / 16.0 - 0.5) / L;
    c = floor(c * (L - 1.0) + 0.5 + d * (L - 1.0)) / (L - 1.0);
    // CRT scanlines
    c *= 0.85 + 0.15 * sin(uv.y * 900.0);
    // vignette toward the corners
    vec2 p = uv - 0.5;
    c *= clamp(1.0 - dot(p, p) * 0.9, 0.0, 1.0);
    color = vec4(c, 1.0);
}
'''


def retro_shader():
    from ursina import Shader
    return Shader(fragment=RETRO_FRAG, geometry='')


def setup_cabin(room=70.0, wall_h=22.0):
    """Drop-in replacement for ursina_table.setup_room used by the 3D game: a big dark wood
    cabin around the table (walls far off in the gloom), the directional + ambient lights the
    lit tiles/felt need, the table camera, and the whole-scene retro post-process shader."""
    global ROOM, WALL_H
    ROOM, WALL_H = room, wall_h
    from ursina import AmbientLight, DirectionalLight, Vec3
    window.color = color.rgb32(3, 2, 2)
    AmbientLight(color=color.hsv(28, 0.25, 0.34))           # warm fill so the lit tiles read
    sun = DirectionalLight()
    sun.color = color.hsv(28, 0.45, 1.0)                    # warm, candle-like key light (gently tinted)
    sun.look_at(Vec3(0.45, -1.0, 0.55))                    # from above the table
    build_shell(floor_y=-0.4)                               # wood floor just under the felt table
    EditorCamera(rotation=(25, 0, 0))                       # same framing as the old table view
    camera.z = -31
    camera.fov = 50
    return sun


if _HAVE_URSINA:

    class Flicker(Entity):
        """Candle flicker: jitters each flame's size + colour and tints the whole room each
        frame so the baked pool of light seems to breathe, the way a real flame does."""

        def __init__(self, flame, shell):
            super().__init__()
            self.flame = flame
            self.shell = shell
            self._level = 1.0

        def update(self):
            # ease toward a new random level so the flicker is jittery but not strobing
            target = random.uniform(0.8, 1.0)
            self._level += (target - self._level) * min(1.0, time.dt * 12)
            lv = self._level
            # self.flame.scale_y = 0.34 * (0.85 + 0.3 * lv)
            # self.flame.scale_x = self.flame.scale_z = 0.16 * (0.92 + 0.12 * lv)
            # self.flame.color = color.hsv(35, 0.85, 0.7 + 0.3 * lv)elf.flame.scale_y = 0.34 * (0.85 + 0.3 * lv)
            # self.flame.scale_x = self.flame.scale_z = 0.16 * (0.92 + 0.12 * lv)
            # self.flame.color = color.hsv(35, 0.85, 0.7 + 0.3 * lv)
            tint = color.hsv(0, 0, lv)                       # neutral multiplier on the baked textures
            for e in self.shell:
                e.color = tint


def main():
    if not _HAVE_URSINA:
        sys.exit('Ursina is not installed — run:  pip install -e ".[threed]"')
    app = Ursina(title="crackedMahjong — cabin")
    window.color = color.rgb32(3, 2, 2)
    flame, shell = build_room()
    Flicker(flame, shell)
    camera.shader = retro_shader()                           # whole-scene posterize + dither + CRT
    EditorCamera(rotation=(20, 0, 0), position=(0, 1.5, 0))  # orbit pivot at the table
    camera.z = -8                                            # pull the camera back, but stay near the centre
    camera.fov = 60
    app.run()


if __name__ == "__main__":
    main()
