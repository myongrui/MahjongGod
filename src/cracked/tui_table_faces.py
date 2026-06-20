"""
Static prototype variant — the table rendered with custom faces for ALL tiles.

Unlike `tui_table` (which is a hybrid: glyphs for dense hands/discards, custom
faces only for big tiles), this renders *every* tile — concealed hands, the
drawn tile, exposed melds, and the centre discards — using the downscaled
custom face artwork from `cracked.tui_tiles`. Opponent tiles are face-down
backs.

It exists to compare the "all custom faces" look against the hybrid at realistic
table scale. Sample data is reused from `cracked.tui_table`.

Run it with:

    python -m cracked.tui_table_faces
"""
from __future__ import annotations

from rich.style import Style
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Static

from cracked.tui_tiles import make_face, _blank_face, _IVORY, FW, FH, FELT
from cracked.tui_table import (
    YOU, TOP, LEFT, RIGHT, DISCARDS,
    ROUND_WIND, WALL_REMAINING, HAND_NO, TURN_NO,
    _card_title, _meld_label, _WIND_CJK, _GOLD,
)

_BACK_BLUE = "#2a6f9e"

# Tile display sizes (cols, cells). Big readable faces for the player's visible
# tiles; compact for face-down opponent backs (no detail to show).
_VIS = (16, 11)     # your hand / meld / drawn  — the "16-width" tiles
_DISC = (12, 8)     # centre discards
_OBACK = (7, 5)     # opponent face-down backs
_OMELD = (11, 7)    # opponent exposed melds


def _back_face():
    f = _blank_face()
    for y in range(FH):
        for x in range(FW):
            if f[y][x] == _IVORY:
                f[y][x] = _BACK_BLUE
    return f


def _scale(face, cw: int, ch: int) -> list[Text]:
    ph = ch * 2
    out: list[Text] = []
    for cy in range(ch):
        line = Text()
        for cx in range(cw):
            fx = min(FW - 1, int((cx + 0.5) / cw * FW))
            ft = min(FH - 1, int((2 * cy + 0.5) / ph * FH))
            fb = min(FH - 1, int((2 * cy + 1.5) / ph * FH))
            top = face[ft][fx]
            bot = face[fb][fx]
            if top is None and bot is None:
                line.append(" ")
            else:
                line.append("▀", Style(color=top or FELT, bgcolor=bot or FELT))
        out.append(line)
    return out


def _faces(faces, cw: int, ch: int, gap: int = 1) -> list[Text]:
    grids = [_scale(f, cw, ch) for f in faces]
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
    out = Text()
    for i, p in enumerate(parts):
        if i:
            out.append("\n")
        out.append_text(Text.from_markup(p) if isinstance(p, str) else p)
    return out


# ---------------------------------------------------------------------------
# Per-zone renderables (everything as custom faces)
# ---------------------------------------------------------------------------

def _you_renderable() -> Text:
    meld = [make_face(t) for t in YOU["meld"].tiles]
    hand = [make_face(t) for t in YOU["concealed"]]
    drawn = [make_face(YOU["drawn"])]
    block = _faces(meld + hand + drawn, *_VIS)
    sep = len(YOU["meld"].tiles) * (_VIS[0] + 1) - 1
    header = "[dim]" + " " * sep + "│ concealed hand" + " " * 90 + "drew →[/dim]"
    status = (f"[dim]{_meld_label(YOU['meld'])}[/dim]   tiles away: [b]{YOU['tiles_away']}[/b]"
              f"   ·   [green]● WAITING[/green]")
    return _join([header, *block, status])


def _top_renderable() -> Text:
    backs = _faces([_back_face()] * TOP["backs"], *_OBACK)
    meld = _faces([make_face(t) for t in TOP["meld"].tiles], *_OMELD)
    return _join([*backs, f"[dim]{_meld_label(TOP['meld'])}[/dim]", *meld])


def _side_renderable(seat: dict) -> Text:
    rows: list[Text] = []
    backs = [_back_face()] * seat["backs"]
    for s in range(0, len(backs), 3):
        rows += _faces(backs[s:s + 3], *_OBACK)
    meld = _faces([make_face(t) for t in seat["meld"].tiles], *_OMELD)
    return _join([*rows, f"[dim]{_meld_label(seat['meld'])}[/dim]", *meld])


def _center_renderable() -> Text:
    head = (f"[{_GOLD}]❖[/{_GOLD}] [b {_GOLD}]{_WIND_CJK[ROUND_WIND]} {ROUND_WIND} Round[/b {_GOLD}]"
            f"   ·   [b]{WALL_REMAINING}[/b] tiles left")
    rows: list = [head, ""]
    faces = [make_face(t) for t in DISCARDS]
    for s in range(0, len(faces), 6):
        rows += _faces(faces[s:s + 6], *_DISC)
        rows.append("")
    return _join(rows)


# ---------------------------------------------------------------------------
# Screen / App
# ---------------------------------------------------------------------------

class TableScreen(Screen):
    CSS = """
    Screen { background: #0a3024; }
    #topbar { dock: top; height: 1; background: #07241b; color: $text-muted; text-align: center; }
    #hint { dock: bottom; height: 1; color: $text-muted; text-align: center; }
    .seat {
        background: #0f3a2c; color: $text;
        border: round #2a8c6a; border-title-color: #cfe8dd;
        content-align: center middle;
    }
    #top { height: 16; content-align: center top; }
    #middle { height: 1fr; }
    #left, #right { width: 28; }
    #center {
        width: 1fr; background: #08261d;
        border: round #c8a23a; border-title-color: #e3c75a;
        content-align: center middle;
    }
    #you {
        height: 15; background: #0f3a2c; color: $text;
        border: round #e3c75a; border-title-color: #e3c75a;
        content-align: center middle;
    }
    """

    BINDINGS = [("q", "quit", "Quit"), ("escape", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Static(f"crackedMahjong  ·  Hand {HAND_NO}  ·  Turn {TURN_NO}  ·  all custom faces", id="topbar")
        top = Static(_top_renderable(), id="top", classes="seat")
        top.border_title = _card_title(TOP)
        yield top
        with Horizontal(id="middle"):
            left = Static(_side_renderable(LEFT), id="left", classes="seat")
            left.border_title = _card_title(LEFT)
            yield left
            center = Static(_center_renderable(), id="center")
            center.border_title = "discards"
            yield center
            right = Static(_side_renderable(RIGHT), id="right", classes="seat")
            right.border_title = _card_title(RIGHT)
            yield right
        you = Static(_you_renderable(), id="you")
        you.border_title = _card_title(YOU)
        yield you
        yield Static("static prototype — all custom faces, sample data, no engine   ·   q / esc to quit", id="hint")


class TableFacesApp(App):
    TITLE = "crackedMahjong — table prototype (all custom faces)"

    def on_mount(self) -> None:
        self.push_screen(TableScreen())


def main() -> None:
    TableFacesApp().run()


if __name__ == "__main__":
    main()
