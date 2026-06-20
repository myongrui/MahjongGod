"""
Smoke test for the standalone physics demo.

Skips when the optional `ui` extra (Textual) is not installed. Verifies the
physics loop advances (tiles move + spin), tiles stay in bounds, and a frame
renders to non-empty half-block output.
"""

import asyncio

import pytest

pytest.importorskip("textual")

from cracked.tui_physics_demo import PhysicsDemoApp  # noqa: E402


def test_tiles_move_spin_and_stay_in_bounds():
    async def go():
        app = PhysicsDemoApp()
        async with app.run_test(size=(70, 24)) as pilot:
            await pilot.pause(0.2)
            before = [(t.x, t.y, t.ang) for t in app._tiles]
            await pilot.pause(0.6)
            after = [(t.x, t.y, t.ang) for t in app._tiles]
            assert any(a[:2] != b[:2] for a, b in zip(before, after))   # moved
            assert any(a[2] != b[2] for a, b in zip(before, after))     # spun
            assert all(0 <= t.x <= app._pw and 0 <= t.y <= app._ph for t in app._tiles)
            assert "▀" in app._render().plain

    asyncio.run(go())
