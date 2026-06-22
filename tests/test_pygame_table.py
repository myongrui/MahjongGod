"""
Headless smoke test for the pygame-ce proof-of-concept.

Skips without pygame; runs SDL in dummy mode (no display needed). Verifies the
PoC deals a real hand from the engine, builds crisp scaled tile sprites, and
renders a frame without error.
"""

import os

import pytest

pygame = pytest.importorskip("pygame")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture(scope="module")
def _pg():
    pygame.init()
    yield
    pygame.quit()


def test_poc_renders_a_frame(_pg):
    from cracked.pygame_table import TablePoC, TW, TH, WIN_W, WIN_H

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    font = pygame.font.Font(None, 22)
    poc = TablePoC()

    assert len(poc.hand) == 13           # real hand from the engine
    assert all(0 <= t < 34 for t in poc.hand)

    surf, cap = poc.standing(poc.hand[0])
    # a standing tile shows its top-edge depth above the face → taller than flat
    assert surf.get_width() == TW
    assert cap > 0 and surf.get_height() == TH + cap

    poc.update(0.02)
    poc.draw(screen, font)               # renders without raising
    assert screen.get_size() == (WIN_W, WIN_H)


def test_throw_settles_into_pool(_pg):
    from cracked.pygame_table import TablePoC

    poc = TablePoC()
    poc._launch_throw()
    assert poc._fly is not None

    moved = False
    for _ in range(600):
        if poc._fly is None:
            break
        prev = (poc._fly.x, poc._fly.y, poc._fly.ang)
        poc._step_throw(1 / 60)
        if poc._fly is None or (poc._fly.x, poc._fly.y, poc._fly.ang) != prev:
            moved = True

    assert moved                         # the thrown tile slid + spun under physics
    assert poc._discards                 # and came to rest in the discard pool


def test_thrown_tile_bounces_off_pool(_pg):
    from cracked.pygame_table import TablePoC, _Fly

    poc = TablePoC()
    cx, cy = poc._well.center
    poc._discards = [(0, cx, cy, 0.0)]               # one resting tile at the centre
    # a flyer overlapping it from above, moving straight down into it
    f = _Fly(0, cx, cy - 55, 0.0, 300.0, 0.0, cx, cy - 255, 200.0)
    y_before = f.y
    poc._fly = f
    poc._collide_pool(f)

    assert f.vy < 0                                   # reflected back up out of the tile
    assert f.y < y_before                             # pushed clear of the penetration


def test_canvas_is_16_9():
    from cracked.pygame_table import WIN_W, WIN_H

    assert abs(WIN_W / WIN_H - 16 / 9) < 1e-3          # design canvas is 16:9
