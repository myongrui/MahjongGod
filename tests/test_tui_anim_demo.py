"""
Smoke test for the standalone animation demo.

Skips when the optional `ui` extra (Textual) is not installed. Verifies the app
mounts and that a thrown tile's offset actually animates over time.
"""

import asyncio

import pytest

pytest.importorskip("textual")

from cracked.tui_anim_demo import AnimDemoApp  # noqa: E402


def test_throw_animation_moves_a_tile():
    async def go():
        app = AnimDemoApp()
        async with app.run_test(size=(80, 28)) as pilot:
            await pilot.pause(0.3)
            tiles = app.query(".tile")
            assert len(tiles) >= 1
            tile = tiles.first()
            o1 = (tile.offset.x, tile.offset.y)
            await pilot.pause(0.18)
            o2 = (tile.offset.x, tile.offset.y)
            assert o1 != o2  # the tile slid

    asyncio.run(go())
