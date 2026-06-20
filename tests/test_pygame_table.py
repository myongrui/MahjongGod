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

    sprite = poc.sprite(poc.hand[0])
    # tile carries thickness + drop-shadow margin, so it's a bit larger than flat
    assert sprite.get_width() >= TW and sprite.get_height() >= TH

    poc.update(0.02)
    poc.draw(screen, font)               # renders without raising
    assert screen.get_size() == (WIN_W, WIN_H)


def test_discard_animation_advances(_pg):
    from cracked.pygame_table import TablePoC

    poc = TablePoC()
    before = poc._drawn_pos()
    poc._t = 0.5
    after = poc._drawn_pos()
    assert before != after               # the drawn tile tweens toward the well
