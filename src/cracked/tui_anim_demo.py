"""
Standalone animation demo — throwing tiles into the centre.

A throwaway, engine-free preview of how tile motion will feel in the live
table: tiles flick one by one from the hand (bottom) into the centre well with
`out_cubic` easing and pile up unsorted, then the pile clears and it loops.
Its only purpose is to validate the animation feel before wiring `animate()`
into the real table.

Run it with:

    python -m cracked.tui_anim_demo
"""
from __future__ import annotations

import asyncio
import random

from textual import work
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.widgets import Static

from cracked.tui_table import _chip   # bare suit-colored Unicode glyph

# A spread of tiles across suits/honours to throw.
_TILES = [0, 4, 8, 9, 13, 17, 18, 22, 26, 27, 28, 29, 31, 32, 33]

_THROW_DURATION = 0.34   # seconds per throw
_THROW_GAP = 0.22        # pause between throws


class AnimDemoApp(App):
    TITLE = "crackedMahjong — animation demo"

    CSS = """
    Screen { background: #0a3024; layers: felt tiles; }

    #well {
        layer: felt;
        width: 30; height: 11;
        border: round #c8a23a; background: #08261d; border-title-color: #e3c75a;
    }
    #hand {
        dock: bottom; height: 1; offset: 0 -1;
        color: $text-muted; text-align: center;
    }
    #hint { dock: bottom; height: 1; color: $text-muted; text-align: center; }

    .tile { layer: tiles; width: 2; height: 1; }
    """

    BINDINGS = [("q", "quit", "Quit"), ("escape", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        well = Static(id="well")
        well.border_title = "discards"
        yield well
        yield Static("[dim]· · · · · · · · · ·  your hand  · · · · · · · · · ·[/dim]", id="hand")
        yield Static("animation demo — tiles thrown into the centre   ·   q / esc to quit", id="hint")

    def on_mount(self) -> None:
        # Centre the well, then start throwing.
        well = self.query_one("#well")
        well.styles.offset = ((self.size.width - 30) // 2, (self.size.height - 11) // 2)
        self.run_throws()

    @work(exclusive=True)
    async def run_throws(self) -> None:
        await asyncio.sleep(0.2)
        thrown = 0
        i = 0
        while True:
            w, h = self.size.width, self.size.height
            tid = _TILES[i % len(_TILES)]

            tile = Static(_chip(tid), classes="tile")
            await self.mount(tile)

            # Start in the hand: spread along the bottom.
            start_x = w // 2 - 20 + (i % 11) * 4
            tile.styles.offset = (start_x, h - 3)

            # Land somewhere random inside the well — discards are not organised.
            tx = w // 2 - 12 + random.randint(0, 22)
            ty = h // 2 - 4 + random.randint(0, 7)
            tile.animate(
                "offset", value=Offset(tx, ty),
                duration=_THROW_DURATION, easing="out_cubic",
            )

            await asyncio.sleep(_THROW_DURATION + _THROW_GAP)
            thrown += 1
            i += 1

            # Clear the pile and loop so it doesn't grow forever.
            if thrown >= 18:
                await self.query(".tile").remove()
                thrown = 0
                i = 0
                await asyncio.sleep(0.4)


def main() -> None:
    AnimDemoApp().run()


if __name__ == "__main__":
    main()
