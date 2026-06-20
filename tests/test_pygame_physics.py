"""
Headless smoke test for the pygame physics demo (skips without pygame).

Runs SDL in dummy mode (no display). Verifies the physics advances (tiles move +
spin), tiles stay in bounds, and a frame renders without error.
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


def test_physics_advances_and_renders(_pg):
    from cracked.pygame_physics import PhysicsWorld, WIN_W, WIN_H

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    world = PhysicsWorld()

    before = [(t.x, t.y, t.ang) for t in world.tiles]
    for _ in range(30):
        world.step(1 / 60)
    after = [(t.x, t.y, t.ang) for t in world.tiles]

    assert any(a[:2] != b[:2] for a, b in zip(before, after))   # moved
    assert any(a[2] != b[2] for a, b in zip(before, after))     # spun
    assert all(world.left - 1 <= t.x <= world.right + 1
               and world.top - 1 <= t.y <= world.bottom + 1 for t in world.tiles)

    world.draw(screen)              # renders without raising
    assert screen.get_size() == (WIN_W, WIN_H)
