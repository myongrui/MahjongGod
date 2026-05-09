"""Textual TUI — Spectator and Interactive game modes for crackedMahjong."""
from __future__ import annotations

from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Button, DataTable, Label, Static,
    Input, RichLog,
)
from textual import work
from textual.timer import Timer

from cracked.tiles import tile_name, NTILES, Wind
from cracked.engine import GameEngine, EventType, GameEvent

# ---------------------------------------------------------------------------
# Tile rendering helpers
# ---------------------------------------------------------------------------

_GLYPH_MAP: list[str] = (
    [chr(0x1F010 + i) for i in range(9)] +  # Bamboo 1-9:   🀐-🀘
    [chr(0x1F007 + i) for i in range(9)] +  # Chars 1-9:    🀇-🀏
    [chr(0x1F019 + i) for i in range(9)] +  # Circles 1-9:  🀙-🀡
    [chr(0x1F000 + i) for i in range(4)] +  # Winds:        🀀-🀃
    [chr(0x1F004 + i) for i in range(3)]    # Dragons:      🀄-🀆
)


def tile_glyph(tid: int) -> str:
    return _GLYPH_MAP[tid]


def _tile_color(tid: int) -> str:
    if tid < 9:  return "green"
    if tid < 18: return "red"
    if tid < 27: return "cyan"
    if tid < 31: return "yellow"
    return "magenta"


def _g(tid: int, bold: bool = False) -> str:
    """Rich-markup glyph for one tile."""
    c = _tile_color(tid)
    g = tile_glyph(tid)
    style = f"bold {c}" if bold else c
    return f"[{style}]{g}[/{style}]"


def _gl(tid: int, bold: bool = False) -> str:
    """Rich-markup label (e.g. B1, EW) for one tile."""
    c = _tile_color(tid)
    n = tile_name(tid).upper()
    style = f"bold {c}" if bold else c
    return f"[{style}]{n}[/{style}]"


def _hand_markup(tiles: list[int], drawn_tid: Optional[int] = None) -> str:
    """Two-line Rich markup: glyph row + label row.  Drawn tile is bold."""
    seen_drawn = False
    glyph_row: list[str] = []
    label_row: list[str] = []
    for tid in tiles:
        highlight = (not seen_drawn) and (tid == drawn_tid)
        if highlight:
            seen_drawn = True
        glyph_row.append(_g(tid, bold=highlight))
        label_row.append(_gl(tid, bold=highlight))
    return " ".join(glyph_row) + "\n" + " ".join(label_row)


def _discards_markup(discards: list[int]) -> str:
    """Compact single-line markup of discard pile (last 24 tiles)."""
    return " ".join(_g(t) for t in discards[-24:])


SEAT_NAMES = {27: "East", 28: "South", 29: "West", 30: "North"}

# ---------------------------------------------------------------------------
# Mode selection screen
# ---------------------------------------------------------------------------


class ModeSelectScreen(Screen):
    CSS = """
    ModeSelectScreen {
        align: center middle;
    }
    #title {
        text-style: bold;
        color: $accent;
        text-align: center;
        padding-bottom: 1;
        width: 100%;
    }
    #subtitle {
        color: $text-muted;
        text-align: center;
        padding-bottom: 2;
        width: 100%;
    }
    #btn-box {
        width: auto;
        height: auto;
        align: center middle;
    }
    Button {
        width: 30;
        margin: 1 2;
    }
    #hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("crackedMahjong 🀄", id="title")
        yield Label("Singapore Mahjong Discard Optimizer", id="subtitle")
        with Horizontal(id="btn-box"):
            yield Button("Spectator Mode", id="btn-spectator", variant="primary")
            yield Button("Interactive Mode", id="btn-interactive", variant="success")
        yield Label("Spectator: watch 4 AI bots play  |  Interactive: you play as East", id="hint")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-spectator":
            self.app.push_screen(SpectatorScreen())
        elif event.button.id == "btn-interactive":
            self.app.push_screen(InteractiveScreen())


# ---------------------------------------------------------------------------
# Spectator screen
# ---------------------------------------------------------------------------


class SpectatorScreen(Screen):
    CSS = """
    SpectatorScreen {
        layout: vertical;
    }
    #top-area {
        height: 1fr;
    }
    #player-col {
        width: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    #log-col {
        width: 50;
        border: round $primary;
        padding: 0 1;
    }
    .player-panel {
        height: auto;
        min-height: 4;
        border-bottom: dashed $surface-darken-2;
        padding: 0 0 1 0;
    }
    .player-panel.current {
        border-left: thick $accent;
        padding-left: 1;
    }
    #controls {
        height: 5;
        border-top: solid $primary;
        padding: 1 2;
        align: left middle;
    }
    #status-bar {
        color: $text-muted;
        margin-left: 2;
    }
    Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        ("p", "toggle_pause", "Pause/Resume"),
        ("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-area"):
            with ScrollableContainer(id="player-col"):
                for seat in [int(Wind.NORTH), int(Wind.WEST),
                             int(Wind.EAST), int(Wind.SOUTH)]:
                    yield Static("", id=f"panel-{seat}", classes="player-panel")
            with ScrollableContainer(id="log-col"):
                yield Label("[bold]Game Log[/bold]")
                yield RichLog(id="game-log", highlight=False, markup=True, wrap=True)
        with Horizontal(id="controls"):
            yield Button("◀ Slower", id="btn-slower")
            yield Button("Faster ▶", id="btn-faster")
            yield Button("⏸ Pause", id="btn-pause")
            yield Button("New Game", id="btn-new", variant="primary")
            yield Button("← Back", id="btn-back")
            yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._speed: float = 0.8   # seconds per step
        self._paused: bool = False
        self._engine = GameEngine(seed=None)
        self._last_drawn: dict[int, Optional[int]] = {s: None for s in [27, 28, 29, 30]}
        self._new_game()

    def _new_game(self) -> None:
        self._engine = GameEngine(seed=None)
        events = self._engine.deal()
        self._last_drawn = {s: None for s in [27, 28, 29, 30]}
        log = self.query_one("#game-log", RichLog)
        log.clear()
        log.write("[bold]New game started.[/bold]")
        self._refresh_all_panels()
        self._update_status()
        if hasattr(self, "_timer"):
            self._timer.stop()
        self._timer: Timer = self.set_interval(self._speed, self._tick)

    def _tick(self) -> None:
        if self._engine.is_finished:
            self._timer.stop()
            return
        events = self._engine.step()
        self._process_events(events)

    def _process_events(self, events: list[GameEvent]) -> None:
        log = self.query_one("#game-log", RichLog)
        for ev in events:
            if ev.type == EventType.DRAW:
                self._last_drawn[ev.seat] = ev.tile
                self._refresh_panel(ev.seat)
                self._update_status()
                log.write(
                    f"T{self._engine.turn_number}  "
                    f"[bold]{SEAT_NAMES[ev.seat]}[/bold] drew "
                    f"{_g(ev.tile)} {_gl(ev.tile)}"
                )
            elif ev.type == EventType.DISCARD:
                self._last_drawn[ev.seat] = None
                self._refresh_panel(ev.seat)
                log.write(
                    f"   [dim]{SEAT_NAMES[ev.seat]} discarded "
                    f"{_g(ev.tile)} {_gl(ev.tile)}[/dim]"
                )
            elif ev.type == EventType.WIN_SELF_DRAW:
                self._timer.stop()
                self._refresh_all_panels()
                log.write(
                    f"[bold green]🎉 {SEAT_NAMES[ev.seat]} wins by self-draw "
                    f"on {_g(ev.tile)} {_gl(ev.tile)}![/bold green]"
                )
                self.query_one("#status-bar", Label).update(
                    f"[bold green]{SEAT_NAMES[ev.seat]} wins by tsumo![/bold green]"
                )
            elif ev.type == EventType.WIN_DISCARD:
                self._timer.stop()
                shooter = ev.detail.get("shooter")
                self._refresh_all_panels()
                log.write(
                    f"[bold green]🎉 {SEAT_NAMES[ev.seat]} wins!  "
                    f"{SEAT_NAMES[shooter]} dealt {_g(ev.tile)} {_gl(ev.tile)}![/bold green]"
                )
                self.query_one("#status-bar", Label).update(
                    f"[bold green]{SEAT_NAMES[ev.seat]} wins by ron![/bold green]"
                )
            elif ev.type == EventType.WALL_EXHAUSTED:
                self._timer.stop()
                log.write("[yellow]Wall exhausted — draw game.[/yellow]")
                self.query_one("#status-bar", Label).update("[yellow]Draw game.[/yellow]")

    def _refresh_panel(self, seat: int) -> None:
        panel = self.query_one(f"#panel-{seat}", Static)
        player = self._engine.players[seat]
        is_current = (seat == self._engine.current_seat and not self._engine.is_finished)
        arrow = "▶ " if is_current else "  "
        name_color = "bold yellow" if is_current else "bold"
        tiles = player.hand.concealed_tiles_list()
        drawn = self._last_drawn.get(seat)
        discards_str = _discards_markup(player.discards)
        hand_str = _hand_markup(tiles, drawn)
        text = (
            f"[{name_color}]{arrow}{SEAT_NAMES[seat]}[/{name_color}]"
            f"  ({len(tiles)} tiles, {len(player.discards)} discards)\n"
            f"{hand_str}\n"
            f"[dim]Discards:[/dim] {discards_str}"
        )
        panel.update(text)

    def _refresh_all_panels(self) -> None:
        for seat in [27, 28, 29, 30]:
            self._refresh_panel(seat)

    def _update_status(self) -> None:
        seat = self._engine.current_seat
        self.query_one("#status-bar", Label).update(
            f"Wall: {self._engine.wall_remaining}  "
            f"Turn: {self._engine.turn_number}  "
            f"Current: {SEAT_NAMES.get(seat, '—')}  "
            f"Speed: {self._speed:.1f}s"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-slower":
            self._speed = min(3.0, self._speed + 0.2)
            self._timer.stop()
            self._timer = self.set_interval(self._speed, self._tick)
            self._update_status()
        elif bid == "btn-faster":
            self._speed = max(0.15, self._speed - 0.2)
            self._timer.stop()
            self._timer = self.set_interval(self._speed, self._tick)
            self._update_status()
        elif bid == "btn-pause":
            self.action_toggle_pause()
        elif bid == "btn-new":
            self._new_game()
        elif bid == "btn-back":
            self.action_go_back()

    def action_toggle_pause(self) -> None:
        btn = self.query_one("#btn-pause", Button)
        if self._paused:
            self._timer.resume()
            btn.label = "⏸ Pause"
        else:
            self._timer.pause()
            btn.label = "▶ Resume"
        self._paused = not self._paused

    def action_go_back(self) -> None:
        if hasattr(self, "_timer"):
            self._timer.stop()
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Interactive screen
# ---------------------------------------------------------------------------


class InteractiveScreen(Screen):
    CSS = """
    InteractiveScreen {
        layout: vertical;
    }
    #opp-row {
        height: 8;
        border-bottom: solid $primary;
    }
    .opp-panel {
        width: 1fr;
        padding: 0 1;
        border-right: dashed $surface-darken-2;
    }
    .opp-panel-title {
        text-style: bold;
        color: $accent;
    }
    #hand-area {
        height: 7;
        padding: 0 2;
        border-bottom: solid $primary;
    }
    #hand-title {
        text-style: bold;
        color: $accent;
    }
    #hand-display {
        height: 3;
    }
    #discard-row {
        height: 3;
        align: left middle;
    }
    #discard-input {
        width: 14;
        margin-right: 2;
    }
    #discard-hint {
        color: $text-muted;
    }
    #bottom-area {
        height: 1fr;
    }
    #rec-col {
        width: 1fr;
        padding: 0 1;
        border-right: solid $primary;
    }
    #log-col {
        width: 44;
        padding: 0 1;
    }
    .section-label {
        text-style: bold;
        color: $accent;
        margin-bottom: 0;
    }
    DataTable {
        height: 1fr;
    }
    Button {
        margin-right: 1;
    }
    #action-row {
        height: 3;
        padding: 0 2;
        border-top: solid $primary;
        align: left middle;
    }
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
    ]

    MY_SEAT = int(Wind.EAST)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="opp-row"):
            for seat in [int(Wind.NORTH), int(Wind.SOUTH), int(Wind.WEST)]:
                with Vertical(classes="opp-panel", id=f"opp-{seat}"):
                    yield Label(SEAT_NAMES[seat], classes="opp-panel-title")
                    yield Static("", id=f"opp-disc-{seat}")
        with Vertical(id="hand-area"):
            yield Label("Your Hand (East)", id="hand-title")
            yield Static("—", id="hand-display")
            with Horizontal(id="discard-row"):
                yield Input(placeholder="tile (e.g. b1, ew)", id="discard-input",
                            disabled=True)
                yield Label("", id="discard-hint")
        with Horizontal(id="bottom-area"):
            with Vertical(id="rec-col"):
                yield Label("Recommendations", classes="section-label")
                yield DataTable(id="rec-table")
            with ScrollableContainer(id="log-col"):
                yield Label("Game Log", classes="section-label")
                yield RichLog(id="game-log", highlight=False, markup=True, wrap=True)
        with Horizontal(id="action-row"):
            yield Button("New Game", id="btn-new", variant="primary")
            yield Button("← Back", id="btn-back")
            yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        rec = self.query_one("#rec-table", DataTable)
        rec.add_columns("#", "Discard", "Shanten", "Accepts", "Danger", "Utility")
        rec.cursor_type = "none"

        self._drawn_tid: Optional[int] = None
        self._engine: Optional[GameEngine] = None
        self._ai_timer: Optional[Timer] = None
        self._new_game()

    def _new_game(self) -> None:
        if self._ai_timer is not None:
            self._ai_timer.stop()
        self._engine = GameEngine(human_seats={self.MY_SEAT}, seed=None)
        self._drawn_tid = None
        events = self._engine.deal()
        log = self.query_one("#game-log", RichLog)
        log.clear()
        log.write("[bold]New game — you are East.[/bold]")
        self._update_opponent_panels()
        self._clear_recommendations()
        self.query_one("#discard-input", Input).disabled = True
        self.query_one("#discard-hint", Label).update("")
        # Start advancing (first step will be East's draw → AWAIT_DISCARD)
        self._ai_timer = self.set_interval(0.35, self._tick)

    def _tick(self) -> None:
        if self._engine is None or self._engine.is_finished:
            if self._ai_timer:
                self._ai_timer.stop()
            return
        events = self._engine.step()
        self._process_events(events)

    def _process_events(self, events: list[GameEvent]) -> None:
        log = self.query_one("#game-log", RichLog)
        for ev in events:
            if ev.type == EventType.DEAL:
                pass
            elif ev.type == EventType.DRAW:
                if ev.seat == self.MY_SEAT:
                    self._drawn_tid = ev.tile
                    self._refresh_hand()
                log.write(
                    f"T{self._engine.turn_number}  "
                    f"[bold]{SEAT_NAMES[ev.seat]}[/bold] drew "
                    + (f"{_g(ev.tile)} {_gl(ev.tile)}" if ev.seat == self.MY_SEAT else "[dim]a tile[/dim]")
                )
            elif ev.type == EventType.AWAIT_DISCARD:
                if self._ai_timer:
                    self._ai_timer.stop()
                inp = self.query_one("#discard-input", Input)
                inp.disabled = False
                inp.value = ""
                inp.focus()
                self.query_one("#discard-hint", Label).update(
                    "[bold yellow]Your turn — type tile to discard[/bold yellow]"
                )
                self._compute_recommendations()
            elif ev.type == EventType.DISCARD:
                if ev.seat == self.MY_SEAT:
                    self._drawn_tid = None
                    self._refresh_hand()
                    self.query_one("#discard-hint", Label).update("")
                else:
                    self._update_opponent_panels()
                log.write(
                    f"[dim]   {SEAT_NAMES[ev.seat]} discarded "
                    f"{_g(ev.tile)} {_gl(ev.tile)}[/dim]"
                )
            elif ev.type == EventType.WIN_SELF_DRAW:
                if self._ai_timer:
                    self._ai_timer.stop()
                self._refresh_hand()
                msg = (
                    f"[bold green]🎉 You win by self-draw on {_g(ev.tile)} {_gl(ev.tile)}![/bold green]"
                    if ev.seat == self.MY_SEAT
                    else f"[bold red]{SEAT_NAMES[ev.seat]} wins by tsumo![/bold red]"
                )
                log.write(msg)
                self.query_one("#status-bar", Label).update(msg)
                self.query_one("#discard-input", Input).disabled = True
                self.query_one("#discard-hint", Label).update("[bold]Game over. Press New Game.[/bold]")
            elif ev.type == EventType.WIN_DISCARD:
                if self._ai_timer:
                    self._ai_timer.stop()
                shooter = ev.detail.get("shooter")
                if ev.seat == self.MY_SEAT:
                    msg = (
                        f"[bold green]🎉 You win! {SEAT_NAMES[shooter]} dealt "
                        f"{_g(ev.tile)} {_gl(ev.tile)}![/bold green]"
                    )
                elif shooter == self.MY_SEAT:
                    msg = (
                        f"[bold red]You shot! {SEAT_NAMES[ev.seat]} wins from your "
                        f"{_g(ev.tile)} {_gl(ev.tile)} discard![/bold red]"
                    )
                else:
                    msg = (
                        f"[bold red]{SEAT_NAMES[ev.seat]} wins! "
                        f"{SEAT_NAMES[shooter]} dealt {_g(ev.tile)} {_gl(ev.tile)}![/bold red]"
                    )
                log.write(msg)
                self.query_one("#status-bar", Label).update(msg)
                self.query_one("#discard-input", Input).disabled = True
                self.query_one("#discard-hint", Label).update("[bold]Game over. Press New Game.[/bold]")
            elif ev.type == EventType.WALL_EXHAUSTED:
                if self._ai_timer:
                    self._ai_timer.stop()
                log.write("[yellow]Wall exhausted — draw game.[/yellow]")
                self.query_one("#status-bar", Label).update("[yellow]Draw game.[/yellow]")
                self.query_one("#discard-input", Input).disabled = True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not self._engine or not self._engine.awaiting_human_discard:
            return
        raw = event.value.strip().lower()
        if not raw:
            return
        from cracked.tiles import tile_id as _tile_id
        try:
            tid = _tile_id(raw)
        except ValueError:
            self.query_one("#discard-hint", Label).update(
                f"[red]Unknown tile '{raw}'. Use b1-b9, c1-c9, d1-d9, ew/sw/ww/nw, rd/gd/wd[/red]"
            )
            inp = self.query_one("#discard-input", Input)
            inp.value = ""
            return

        try:
            discard_events = self._engine.submit_discard(tid)
        except ValueError as exc:
            self.query_one("#discard-hint", Label).update(f"[red]{exc}[/red]")
            self.query_one("#discard-input", Input).value = ""
            return

        inp = self.query_one("#discard-input", Input)
        inp.disabled = True
        inp.value = ""
        self._clear_recommendations()
        self._process_events(discard_events)

        if not self._engine.is_finished:
            self._ai_timer = self.set_interval(0.35, self._tick)

    def _refresh_hand(self) -> None:
        if self._engine is None:
            return
        player = self._engine.players[self.MY_SEAT]
        tiles = player.hand.concealed_tiles_list()
        from cracked.shanten import shanten
        s = shanten(player.hand.concealed, len(player.hand.melds))
        shanten_str = "tenpai!" if s == 0 else ("win!" if s == -1 else str(s))
        wall = self._engine.wall_remaining
        self.query_one("#hand-title", Label).update(
            f"[bold]Your Hand (East)[/bold]  —  Shanten: [bold]{shanten_str}[/bold]"
            f"  |  Wall: {wall}  Turn: {self._engine.turn_number}"
        )
        self.query_one("#hand-display", Static).update(
            _hand_markup(tiles, self._drawn_tid)
        )

    def _update_opponent_panels(self) -> None:
        if self._engine is None:
            return
        for seat in [int(Wind.NORTH), int(Wind.SOUTH), int(Wind.WEST)]:
            player = self._engine.players[seat]
            discs = _discards_markup(player.discards)
            self.query_one(f"#opp-disc-{seat}", Static).update(
                f"[dim]Discards:[/dim] {discs if discs else '—'}"
            )

    @work(thread=True)
    def _compute_recommendations(self) -> None:
        if self._engine is None or not self._engine.awaiting_human_discard:
            return
        try:
            recs = self._engine.get_recommendations()
        except Exception:
            return

        def _update() -> None:
            table = self.query_one("#rec-table", DataTable)
            table.clear()
            for i, r in enumerate(recs[:8], 1):
                s_str = "tenpai" if r.shanten_after == 0 else ("win" if r.shanten_after == -1 else str(r.shanten_after))
                table.add_row(
                    str(i),
                    f"{tile_glyph(r.tile_id)} {tile_name(r.tile_id).upper()}",
                    s_str,
                    str(r.weighted_acceptance),
                    f"{r.danger_score:.2f}",
                    f"{r.utility:.3f}",
                )

        self.call_from_thread(_update)

    def _clear_recommendations(self) -> None:
        self.query_one("#rec-table", DataTable).clear()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-new":
            self._new_game()
        elif event.button.id == "btn-back":
            self.action_go_back()

    def action_go_back(self) -> None:
        if self._ai_timer:
            self._ai_timer.stop()
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class CrackedPlayApp(App):
    TITLE = "crackedMahjong"
    SUB_TITLE = "Singapore Mahjong"

    CSS = """
    Screen {
        background: $surface;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(ModeSelectScreen())


def main() -> None:
    CrackedPlayApp().run()


if __name__ == "__main__":
    main()
