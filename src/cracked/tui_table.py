"""
Static prototype of the spatial mahjong-table TUI (Mahjong Soul-styled).

A standalone, NON-interactive preview of the planned table look, leaning on
Mahjong Soul cues adapted to the terminal: framed corner info cards with CJK
wind badges (東/南/西/北) and points, a dealer highlight, a central "well" that
carries the round wind + tiles remaining, a deep-green felt with gold accents,
and the just-drawn tile separated from the hand. Per request, the discards are
tossed *messily* into the centre well rather than organised into neat rows.

It is driven entirely by a hard-coded sample snapshot — there is no engine and
no game logic here. Its only purpose is to let the layout/look be approved
before it is wired to the live `GameMatch`/`engine.step()` stream and animated.

Run it with:

    python -m cracked.tui_table
"""
from __future__ import annotations

import random

from rich.style import Style
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Static

from cracked.hand import Meld, MeldType
# Hybrid rendering: dense tiles (concealed hands, discards) stay compact Unicode
# glyphs; "big" tiles (exposed melds, the drawn tile) use the custom face art.
from cracked.tui_game import tile_glyph, _tile_color
from cracked.tui_tiles import make_face, FW as _FW, FH as _FH, FELT as _FELT

# ---------------------------------------------------------------------------
# Tile-id shorthands (encoding: bamboo 0-8, char 9-17, circle 18-26,
# winds 27-30 = E/S/W/N, dragons 31-33 = Red/Green/White)
# ---------------------------------------------------------------------------

def _b(n: int) -> int: return n - 1          # bamboo 1-9
def _c(n: int) -> int: return 9 + n - 1      # characters 1-9
def _d(n: int) -> int: return 18 + n - 1     # circles 1-9

EW, SW, WW, NW = 27, 28, 29, 30
RD, GD, WD = 31, 32, 33

_WIND_CJK = {"East": "東", "South": "南", "West": "西", "North": "北"}

_GOLD = "#e3c75a"
_BACK_FACE = "#2a6f9e"   # blue tile back

def _chip(tid: int, bold: bool = False) -> str:
    """One tile as a bare suit-colored Unicode glyph (no background)."""
    col = _tile_color(tid)
    edge = "bold " if bold else ""
    return f"[{edge}{col}]{tile_glyph(tid)}[/]"


def _back_chip() -> str:
    """A face-down tile: solid blue tile back."""
    return f"[{_BACK_FACE} on {_BACK_FACE}]██[/]"


def _meld_label(meld: Meld) -> str:
    if meld.type.value == "chow":
        return "Chow"
    if meld.type.value == "pong":
        return "Pong"
    return "Kong(hidden)" if meld.concealed else "Kong"


# --- custom-face rendering (downscaled half-block) for the "big" tiles --------

def _scaled_face(tid: int, cw: int, ch: int) -> list[Text]:
    """Render a custom tile face downscaled to cw cols x ch cells (half-block)."""
    face = make_face(tid)
    ph = ch * 2
    out: list[Text] = []
    for cy in range(ch):
        line = Text()
        for cx in range(cw):
            fx = min(_FW - 1, int((cx + 0.5) / cw * _FW))
            ft = min(_FH - 1, int((2 * cy + 0.5) / ph * _FH))
            fb = min(_FH - 1, int((2 * cy + 1.5) / ph * _FH))
            top = face[ft][fx]
            bot = face[fb][fx]
            if top is None and bot is None:
                line.append(" ")
            else:
                line.append("▀", Style(color=top or _FELT, bgcolor=bot or _FELT))
        out.append(line)
    return out


def _faces_row(tids, cw: int, ch: int, gap: int = 1) -> list[Text]:
    """A horizontal row of custom faces, returned as ch Text lines."""
    grids = [_scaled_face(t, cw, ch) for t in tids]
    rows: list[Text] = []
    for r in range(ch):
        line = Text()
        for i, g in enumerate(grids):
            if i:
                line.append(" " * gap)
            line.append_text(g[r])
        rows.append(line)
    return rows


def _join(parts) -> Text:
    """Stack a list of Text/markup-str pieces into one multi-line Text."""
    out = Text()
    for i, p in enumerate(parts):
        if i:
            out.append("\n")
        out.append_text(Text.from_markup(p) if isinstance(p, str) else p)
    return out

# ---------------------------------------------------------------------------
# Hard-coded sample mid-game snapshot
#   Screen position (top/left/right/bottom) is fixed; each seat's mahjong wind
#   is distinct E/S/W/N. You sit at the bottom as East = the dealer.
# ---------------------------------------------------------------------------

ROUND_WIND = "East"
WALL_REMAINING = 58
HAND_NO = 7
TURN_NO = 12

# You (bottom): one exposed Pong, nine concealed + the fresh draw set apart.
YOU = {
    "wind": "East", "player": 1, "chips": 478, "dealer": True,
    "concealed": [_b(1), _b(2), _b(3), _c(1), _c(2), _c(3), _d(1), _d(2), _d(3)],
    "drawn": _b(4),
    "meld": Meld(type=MeldType.PONG, tiles=(RD, RD, RD)),
    "tiles_away": 1, "waiting": True,
}
TOP = {  # player across from you
    "wind": "West", "player": 3, "chips": 512, "backs": 10,
    "meld": Meld(type=MeldType.PONG, tiles=(_d(5), _d(5), _d(5))),
}
LEFT = {
    "wind": "North", "player": 2, "chips": 480, "backs": 10,
    "meld": Meld(type=MeldType.CHOW, tiles=(_b(1), _b(2), _b(3))),
}
RIGHT = {
    "wind": "South", "player": 4, "chips": 530, "backs": 10,
    "meld": Meld(type=MeldType.PONG, tiles=(WW, WW, WW)),
}

# One shared centre pool holding every discard at once — thrown in, not sorted.
DISCARDS = [_c(1), _b(1), _d(3), WW, _b(9), RD, _d(1), _b(7),
            _c(3), _d(5), NW, _b(3), WD, _c(6), _d(2), _b(4),
            EW, _c(9)]


# ---------------------------------------------------------------------------
# Markup builders for the sample data
# ---------------------------------------------------------------------------

def _wind_badge(wind: str) -> str:
    return f"[{_GOLD}]{_WIND_CJK[wind]}[/{_GOLD}] {wind}"


def _card_title(seat: dict) -> str:
    tag = "  [b]◆DEALER[/b]" if seat.get("dealer") else ""
    return f"{_wind_badge(seat['wind'])}  ·  P{seat['player']}  ·  [{_GOLD}]★{seat['chips']}[/{_GOLD}]{tag}"


def _backs_row(n: int) -> str:
    return " ".join(_back_chip() for _ in range(n))


def _backs_grid(n: int, cols: int = 3) -> str:
    """Face-down backs in a compact grid (keeps the side seats short)."""
    cells = [_back_chip()] * n
    return "\n".join(" ".join(cells[i:i + cols]) for i in range(0, n, cols))


def _scatter_pool(discards: list[int], rows: int = 8, seed: int = 7) -> str:
    """Discards tossed into the well at random offsets — deliberately unsorted."""
    rng = random.Random(seed)
    buckets: list[list[int]] = [[] for _ in range(rows)]
    for tid in discards:
        buckets[rng.randrange(rows)].append(tid)
    lines: list[str] = []
    for bucket in buckets:
        if not bucket:
            lines.append("")
            continue
        parts = [" " * rng.randint(0, 6)]
        for tid in bucket:
            parts.append(_chip(tid) + " " * rng.randint(1, 2))
        lines.append("".join(parts))
    return "\n".join(lines)


def _center_text() -> str:
    return (
        f"[{_GOLD}]❖[/{_GOLD}] [b {_GOLD}]{_WIND_CJK[ROUND_WIND]} {ROUND_WIND} Round[/b {_GOLD}]"
        f"   ·   [b]{WALL_REMAINING}[/b] tiles left\n\n"
        f"{_scatter_pool(DISCARDS)}"
    )


def _top_renderable() -> Text:
    """Across seat: a row of compact backs + the exposed meld as custom faces."""
    parts = [
        _backs_row(TOP["backs"]),
        f"[dim]{_meld_label(TOP['meld'])}[/dim]",
        *_faces_row(TOP["meld"].tiles, 6, 4),
    ]
    return _join(parts)


def _side_renderable(seat: dict) -> Text:
    parts = [
        _backs_grid(seat["backs"], cols=5),
        f"[dim]{_meld_label(seat['meld'])}[/dim]",
        *_faces_row(seat["meld"].tiles, 5, 4),
    ]
    return _join(parts)


def _you_renderable() -> Text:
    """Hybrid: exposed meld + drawn tile as custom faces; concealed hand glyphs."""
    meld_faces = _faces_row(YOU["meld"].tiles, 7, 5)
    drawn_faces = _faces_row([YOU["drawn"]], 7, 5)
    block: list[Text] = []
    for i in range(5):
        line = Text()
        line.append_text(meld_faces[i])
        line.append("     ")
        line.append_text(drawn_faces[i])
        block.append(line)
    header = (f"[dim]▸ {_meld_label(YOU['meld'])} (exposed)[/dim]"
              f"                        [dim]drew →[/dim]")
    hand = "   " + "  ".join(_chip(t) for t in YOU["concealed"])
    status = (f"concealed hand ↑      tiles away: [b]{YOU['tiles_away']}[/b]"
              f"   ·   [green]● WAITING[/green]")
    return _join([header, *block, hand, status])


# ---------------------------------------------------------------------------
# Screen / App
# ---------------------------------------------------------------------------

class TableScreen(Screen):
    # Deep-green felt; framed seat cards around a gold-rimmed centre well.
    CSS = """
    Screen { background: #0a3024; }

    #topbar {
        dock: top; height: 1;
        background: #07241b; color: $text-muted; text-align: center;
    }
    #hint {
        dock: bottom; height: 1;
        color: $text-muted; text-align: center;
    }

    .seat {
        background: #0f3a2c; color: $text;
        border: round #2a8c6a; border-title-color: #cfe8dd;
        content-align: center middle; text-align: center;
    }
    #top { height: 8; content-align: center top; }
    #middle { height: 1fr; }
    #left, #right { width: 24; }
    #center {
        width: 1fr;
        background: #08261d; border: round #c8a23a; border-title-color: #e3c75a;
        content-align: center middle; text-align: center;
    }
    /* Dealer seat (you) gets the gold frame. */
    #you {
        height: 10;
        background: #0f3a2c; color: $text;
        border: round #e3c75a; border-title-color: #e3c75a;
        content-align: center middle; text-align: center;
    }
    /* Mouse groundwork: hovering your hand highlights it (proof-of-concept). */
    #you:hover { background: #154536; }
    """

    BINDINGS = [("q", "quit", "Quit"), ("escape", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Static(f"crackedMahjong  ·  Hand {HAND_NO}  ·  Turn {TURN_NO}", id="topbar")

        top = Static(_top_renderable(), id="top", classes="seat")
        top.border_title = _card_title(TOP)
        yield top

        with Horizontal(id="middle"):
            left = Static(_side_renderable(LEFT), id="left", classes="seat")
            left.border_title = _card_title(LEFT)
            yield left

            center = Static(_center_text(), id="center")
            center.border_title = "discards"
            yield center

            right = Static(_side_renderable(RIGHT), id="right", classes="seat")
            right.border_title = _card_title(RIGHT)
            yield right

        you = Static(_you_renderable(), id="you")
        you.border_title = _card_title(YOU)
        yield you

        yield Static(
            "static prototype — sample data, no engine   ·   q / esc to quit",
            id="hint",
        )

    # TODO(live): on_click → map the clicked offset to a hand tile and discard it.


class TablePrototypeApp(App):
    TITLE = "crackedMahjong — table prototype"

    def on_mount(self) -> None:
        self.push_screen(TableScreen())


def main() -> None:
    TablePrototypeApp().run()


if __name__ == "__main__":
    main()
