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

from cracked.tiles import tile_name, NTILES, Wind, bonus_tile_name, is_animal
from cracked.engine import GameEngine, EventType, GameEvent
from cracked.match import GameMatch, WIND_NAMES
from cracked.shanten import shanten
from cracked.scoring import calculate_tai, WinContext, DEFAULT_RULES

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
    """Two-line markup: glyph row then label row directly below.
    Glyph sep is 1 wider than label sep so each tile slot has equal display width
    (glyphs render as 1-wide text chars; labels are 2 chars → slot = glyph+4 = label+3 = 5).
    """
    seen_drawn = False
    glyph_parts: list[str] = []
    label_parts: list[str] = []
    for tid in tiles:
        highlight = (not seen_drawn) and (tid == drawn_tid)
        if highlight:
            seen_drawn = True
        glyph_parts.append(_g(tid, bold=highlight))
        label_parts.append(_gl(tid, bold=highlight))
    return "    ".join(glyph_parts) + "\n" + "   ".join(label_parts)


def _discards_markup(discards: list[int]) -> str:
    """Compact single-line markup of discard pile (last 24 tiles)."""
    return " ".join(_g(t) for t in discards[-24:])


def _meld_markup(meld) -> str:
    """Render one exposed meld as a compact Rich markup group."""
    tiles_str = " ".join(f"{_g(t)} {_gl(t)}" for t in meld.tiles)
    if meld.type.value == "chow":
        label = "Chow"
    elif meld.type.value == "pong":
        label = "Pong"
    else:
        label = "Kong (hidden)" if meld.concealed else "Kong"
    return f"[bold]({label}:[/bold] {tiles_str}[bold])[/bold]"


def _bonus_tiles_markup(flowers: list[int], animals: list[int]) -> str:
    """Render bonus tiles (flowers/seasons green, animals yellow)."""
    parts: list[str] = []
    for bid in flowers:
        parts.append(f"[green]{bonus_tile_name(bid)}[/green]")
    for bid in animals:
        parts.append(f"[yellow]{bonus_tile_name(bid)}[/yellow]")
    return "  ".join(parts)


def _exposed_row(hand) -> str:
    """Single-line summary of a hand's exposed melds and bonus tiles."""
    meld_parts = [_meld_markup(m) for m in hand.melds]
    bonus_str = _bonus_tiles_markup(hand.flowers, hand.animals)
    sections: list[str] = []
    if meld_parts:
        sections.append("  ┃  ".join(meld_parts))
    if bonus_str:
        sections.append(f"[dim]Bonus:[/dim] {bonus_str}")
    return "   ".join(sections)


SEAT_NAMES = {27: "East", 28: "South", 29: "West", 30: "North"}


def _player_label(match: "GameMatch", seat: int) -> str:
    """Return e.g. 'Player 2 (South)' for the player currently at this wind seat."""
    num = match.player_at.get(seat, "?")
    wind = SEAT_NAMES.get(seat, "?")
    return f"Player {num} ({wind})"


def _win_tai_str(engine: GameEngine, seat: int, tile: int, self_draw: bool) -> str:
    """Score a completed hand and return a short display string e.g. '4 tai ✓'."""
    hand = engine.players[seat].hand.copy()
    if not self_draw:
        hand.concealed[tile] += 1
    ctx = WinContext(winning_tile=tile, is_self_draw=self_draw,
                     prevailing_wind=engine.prevailing_wind)
    try:
        r = calculate_tai(hand, ctx, DEFAULT_RULES)
        suffix = " ✓" if r.is_valid_win() else " (below min)"
        return f"{r.total} tai{suffix}"
    except Exception:
        return "? tai"


def _waiting_tai_label(hand, prevailing_wind: int) -> str:
    """Return 'Waiting [X–Y tai]' (or 'Waiting [N tai]') for a 13-tile waiting hand."""
    tai_values: list[int] = []
    for wt in range(NTILES):
        test = hand.concealed.copy()
        test[wt] += 1
        if shanten(test, len(hand.melds)) == -1:
            h = hand.copy()
            h.concealed[wt] += 1
            ctx = WinContext(winning_tile=wt, prevailing_wind=prevailing_wind)
            try:
                tai_values.append(calculate_tai(h, ctx, DEFAULT_RULES).total)
            except Exception:
                pass
    if not tai_values:
        return "Waiting"
    lo, hi = min(tai_values), max(tai_values)
    return f"Waiting [{lo}–{hi} tai]" if lo != hi else f"Waiting [{lo} tai]"


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
            yield Button("Quit", id="btn-quit", variant="error")
        yield Label("Spectator: watch 4 AI bots play  |  Interactive: you play as East", id="hint")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-spectator":
            self.app.push_screen(SpectatorScreen())
        elif event.button.id == "btn-interactive":
            self.app.push_screen(InteractiveScreen())
        elif event.button.id == "btn-quit":
            self.app.exit()


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
                for seat in [int(Wind.EAST), int(Wind.SOUTH),
                             int(Wind.WEST), int(Wind.NORTH)]:
                    yield Static("", id=f"panel-{seat}", classes="player-panel")
            with ScrollableContainer(id="log-col"):
                yield Label("[bold]Game Log[/bold]")
                yield RichLog(id="game-log", highlight=False, markup=True, wrap=True)
        with Horizontal(id="controls"):
            yield Button("◀ Slower", id="btn-slower")
            yield Button("Faster ▶", id="btn-faster")
            yield Button("⏸ Pause", id="btn-pause")
            yield Button("New Match", id="btn-new", variant="primary")
            yield Button("← Back", id="btn-back")
            yield Button("Quit", id="btn-quit", variant="error")
            yield Label("", id="status-bar")
            yield Label("", id="table-wind-label")
        yield Footer()

    def on_mount(self) -> None:
        self._speed: float = 0.8   # seconds per step
        self._paused: bool = False
        self._match: GameMatch = GameMatch()
        self._last_drawn: dict[int, Optional[int]] = {s: None for s in [27, 28, 29, 30]}
        self._new_match()

    def _new_match(self) -> None:
        self._match = GameMatch()
        self._start_hand("[bold]New match started — East Round, Hand 1.[/bold]")

    def _start_hand(self, intro: str = "") -> None:
        events = self._match.start_hand()
        self._last_drawn = {s: None for s in [27, 28, 29, 30]}
        log = self.query_one("#game-log", RichLog)
        if intro:
            log.write(intro)
        self._process_events(events)
        self._refresh_all_panels()
        self._update_status()
        if hasattr(self, "_timer"):
            self._timer.stop()
        self._timer: Timer = self.set_interval(self._speed, self._tick)

    def _next_hand(self) -> None:
        result = self._match.finish_hand()
        log = self.query_one("#game-log", RichLog)
        if result.rotated:
            new_east_player = _player_label(self._match, int(Wind.EAST))
            log.write(
                f"[cyan]Winds rotate (S→E, E→N, N→W, W→S) — {new_east_player} deals next.[/cyan]"
            )
        else:
            log.write("[cyan]Dealer wins — no rotation, Dealer deals again.[/cyan]")
        if self._match.is_complete:
            log.write("[bold green]Match complete![/bold green]")
            self._show_final_standings(log)
            self.query_one("#status-bar", Label).update("[bold green]Match over.[/bold green]")
            return
        intro = (
            f"[bold]{self._match.round_label}, Hand {self._match.hand_number}[/bold]"
        )
        self._start_hand(intro)

    def _show_final_standings(self, log: RichLog) -> None:
        log.write("[bold]Final standings:[/bold]")
        sorted_chips = sorted(self._match.chips.items(), key=lambda x: -x[1])
        for wind, chips in sorted_chips:
            log.write(f"  {_player_label(self._match, wind):20s}  ★{chips}")

    def _tick(self) -> None:
        if self._match.engine is None or self._match.engine.is_finished:
            self._timer.stop()
            return
        events = self._match.engine.step()
        self._process_events(events)

    def _process_events(self, events: list[GameEvent]) -> None:
        log = self.query_one("#game-log", RichLog)
        engine = self._match.engine
        for ev in events:
            if ev.type == EventType.DRAW:
                self._last_drawn[ev.seat] = ev.tile
                self._refresh_panel(ev.seat)
                self._update_status()
                log.write(
                    f"T{engine.turn_number}  "
                    f"[bold]{_player_label(self._match, ev.seat)}[/bold] drew "
                    f"{_g(ev.tile)} {_gl(ev.tile)}"
                )
            elif ev.type == EventType.DISCARD:
                self._last_drawn[ev.seat] = None
                self._refresh_panel(ev.seat)
                log.write(
                    f"   [dim]{_player_label(self._match, ev.seat)} discarded "
                    f"{_g(ev.tile)} {_gl(ev.tile)}[/dim]"
                )
            elif ev.type == EventType.WIN_SELF_DRAW:
                self._timer.stop()
                self._refresh_all_panels()
                tai = _win_tai_str(engine, ev.seat, ev.tile, self_draw=True)
                zimo_pay = ev.detail.get("zimo_pay", "?")
                winner_label = _player_label(self._match, ev.seat)
                log.write(
                    f"[bold green]🎉 {winner_label} wins by self-draw "
                    f"on {_g(ev.tile)} {_gl(ev.tile)}! — {tai}  "
                    f"(+{zimo_pay * 3 if isinstance(zimo_pay, int) else '?'} ★, each pays {zimo_pay} ★)[/bold green]"
                )
                self.query_one("#status-bar", Label).update(
                    f"[bold green]{winner_label} wins zi mo! — {tai}  each pays {zimo_pay} ★[/bold green]"
                )
                self.set_timer(2.0, self._next_hand)
            elif ev.type == EventType.WIN_DISCARD:
                self._timer.stop()
                shooter = ev.detail.get("shooter")
                self._refresh_all_panels()
                tai = _win_tai_str(engine, ev.seat, ev.tile, self_draw=False)
                shooter_pay = ev.detail.get("shooter_pay", "?")
                winner_label = _player_label(self._match, ev.seat)
                shooter_label = _player_label(self._match, shooter) if shooter in SEAT_NAMES else "?"
                log.write(
                    f"[bold green]🎉 {winner_label} wins!  "
                    f"{shooter_label} dealt {_g(ev.tile)} {_gl(ev.tile)}! — {tai}  "
                    f"({shooter_label} pays {shooter_pay} ★)[/bold green]"
                )
                self.query_one("#status-bar", Label).update(
                    f"[bold green]{winner_label} wins from {shooter_label}! — {tai}  {shooter_pay} ★[/bold green]"
                )
                self.set_timer(2.0, self._next_hand)
            elif ev.type == EventType.BONUS:
                self._refresh_panel(ev.seat)
                log.write(
                    f"   [green]{_player_label(self._match, ev.seat)} draws bonus: "
                    f"{bonus_tile_name(ev.tile)}[/green]"
                )
            elif ev.type == EventType.MELD:
                meld_type = ev.detail.get("meld_type", "meld").capitalize()
                from_seat = ev.detail.get("from", -1)
                from_str = f" from {_player_label(self._match, from_seat)}" if from_seat in SEAT_NAMES else ""
                tiles_str = " ".join(_g(t) for t in ev.detail.get("tiles", [ev.tile]))
                self._refresh_panel(ev.seat)
                log.write(
                    f"   [bold cyan]{_player_label(self._match, ev.seat)} claims {meld_type}{from_str}! "
                    f"{tiles_str}[/bold cyan]"
                )
            elif ev.type == EventType.WALL_EXHAUSTED:
                self._timer.stop()
                log.write("[yellow]Wall exhausted — draw game.[/yellow]")
                self.query_one("#status-bar", Label).update("[yellow]Draw game.[/yellow]")
                self.set_timer(2.0, self._next_hand)

    def _refresh_panel(self, seat: int) -> None:
        panel = self.query_one(f"#panel-{seat}", Static)
        engine = self._match.engine
        player = engine.players[seat]
        is_current = (seat == engine.current_seat and not engine.is_finished)
        arrow = "▶ " if is_current else "  "
        name_color = "bold yellow" if is_current else "bold"
        tiles = player.hand.concealed_tiles_list()
        drawn = self._last_drawn.get(seat)
        discards_str = _discards_markup(player.discards)
        hand_str = _hand_markup(tiles, drawn)
        n_tiles = int(player.hand.concealed.sum())
        s = shanten(player.hand.concealed, len(player.hand.melds))
        if s == -1:
            tai_label = "[bold green]Win![/bold green]"
        elif s == 0 and n_tiles == 13:
            tai_label = f"[yellow]{_waiting_tai_label(player.hand, engine.prevailing_wind)}[/yellow]"
        elif s == 0:
            tai_label = "[yellow]Waiting[/yellow]"
        else:
            tai_label = f"Tiles away: {s}"
        exposed = _exposed_row(player.hand)
        exposed_line = f"\n{exposed}" if exposed else ""
        chips = engine.chips.get(seat, 0)
        text = (
            f"[{name_color}]{arrow}{_player_label(self._match, seat)}[/{name_color}]"
            f"  ({len(tiles)} tiles, {len(player.discards)} discards)  {tai_label}"
            f"  [yellow]★{chips}[/yellow]"
            f"{exposed_line}\n"
            f"{hand_str}\n"
            f"[dim]Discards:[/dim] {discards_str}"
        )
        panel.update(text)

    def _refresh_all_panels(self) -> None:
        for seat in [27, 28, 29, 30]:
            self._refresh_panel(seat)

    def _update_status(self) -> None:
        engine = self._match.engine
        seat = engine.current_seat
        self.query_one("#status-bar", Label).update(
            f"Round {self._match.hand_number}  "
            f"Wall: {engine.wall_remaining}  "
            f"Turn: {engine.turn_number}  "
            f"Current: {_player_label(self._match, seat)}  "
            f"Speed: {self._speed:.1f}s"
            f"\n[bold yellow]Table Wind: {WIND_NAMES[self._match.table_wind]}[/bold yellow]"
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
            self._new_match()
        elif bid == "btn-back":
            self.action_go_back()
        elif bid == "btn-quit":
            self.app.exit()

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
        height: 10;
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
        height: 9;
        padding: 0 2;
        border-bottom: solid $primary;
    }
    #hand-title {
        text-style: bold;
        color: $accent;
    }
    #meld-display {
        height: 2;
        color: $text;
        border-bottom: dashed $surface-darken-2;
        padding-bottom: 0;
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="opp-row"):
            for i in range(3):
                with Vertical(classes="opp-panel", id=f"opp-{i}"):
                    yield Label("", classes="opp-panel-title", id=f"opp-title-{i}")
                    yield Static("", id=f"opp-disc-{i}")
        with Vertical(id="hand-area"):
            yield Label("Your Hand (East)", id="hand-title")
            yield Static("", id="meld-display")
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
            yield Button("New Match", id="btn-new", variant="primary")
            yield Button("Next Hand", id="btn-next", variant="success", disabled=True)
            yield Button("← Back", id="btn-back")
            yield Button("Quit", id="btn-quit", variant="error")
            yield Label("", id="status-bar")
            yield Label("", id="table-wind-label")
        yield Footer()

    def on_mount(self) -> None:
        rec = self.query_one("#rec-table", DataTable)
        rec.add_columns("#", "Discard", "Tiles away", "Accepts", "Danger", "Tai pot", "Utility")
        rec.cursor_type = "none"

        self._drawn_tid: Optional[int] = None
        self._match: Optional[GameMatch] = None
        self._ai_timer: Optional[Timer] = None
        self._new_match()

    @property
    def _my_seat(self) -> int:
        return self._match.human_wind if self._match else int(Wind.EAST)

    @property
    def _engine(self) -> Optional[GameEngine]:
        return self._match.engine if self._match else None

    def _new_match(self) -> None:
        if self._ai_timer is not None:
            self._ai_timer.stop()
        self._match = GameMatch(human_initial_wind=int(Wind.EAST))
        log = self.query_one("#game-log", RichLog)
        log.clear()
        self.query_one("#btn-next", Button).disabled = True
        self._start_hand(f"[bold]New match — you are {WIND_NAMES[self._my_seat]}.[/bold]")

    def _start_hand(self, intro: str = "") -> None:
        if self._ai_timer is not None:
            self._ai_timer.stop()
        events = self._match.start_hand()
        self._drawn_tid = None
        log = self.query_one("#game-log", RichLog)
        if intro:
            log.write(intro)
        self._update_opponent_labels()
        self._update_opponent_panels()
        self._clear_recommendations()
        self.query_one("#discard-input", Input).disabled = True
        self.query_one("#discard-hint", Label).update("")
        self.query_one("#btn-next", Button).disabled = True
        self.query_one("#table-wind-label", Label).update(
            f"[bold yellow]Table Wind: {WIND_NAMES[self._match.table_wind]}[/bold yellow]"
        )
        self._process_events(events)
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
        engine = self._engine
        for ev in events:
            if ev.type == EventType.DEAL:
                pass
            elif ev.type == EventType.DRAW:
                if ev.seat == self._my_seat:
                    self._drawn_tid = ev.tile
                    self._refresh_hand()
                log.write(
                    f"T{engine.turn_number}  "
                    f"[bold]{_player_label(self._match, ev.seat)}[/bold] drew "
                    + (f"{_g(ev.tile)} {_gl(ev.tile)}" if ev.seat == self._my_seat else "[dim]a tile[/dim]")
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
                if ev.seat == self._my_seat:
                    self._drawn_tid = None
                    self._refresh_hand()
                    self.query_one("#discard-hint", Label).update("")
                else:
                    self._update_opponent_panels()
                log.write(
                    f"[dim]   {_player_label(self._match, ev.seat)} discarded "
                    f"{_g(ev.tile)} {_gl(ev.tile)}[/dim]"
                )
            elif ev.type == EventType.WIN_SELF_DRAW:
                if self._ai_timer:
                    self._ai_timer.stop()
                self._refresh_hand()
                self._update_opponent_panels()
                tai = _win_tai_str(engine, ev.seat, ev.tile, self_draw=True)
                zimo_pay = ev.detail.get("zimo_pay", "?")
                chip_str = f"  each pays {zimo_pay} ★"
                if ev.seat == self._my_seat:
                    msg = f"[bold green]🎉 You win by self-draw on {_g(ev.tile)} {_gl(ev.tile)}! — {tai}{chip_str}[/bold green]"
                else:
                    msg = f"[bold red]{_player_label(self._match, ev.seat)} wins zi mo! — {tai}{chip_str}[/bold red]"
                log.write(msg)
                self.query_one("#status-bar", Label).update(msg)
                self.query_one("#discard-input", Input).disabled = True
                self._on_hand_over()
            elif ev.type == EventType.WIN_DISCARD:
                if self._ai_timer:
                    self._ai_timer.stop()
                self._refresh_hand()
                self._update_opponent_panels()
                shooter = ev.detail.get("shooter")
                tai = _win_tai_str(engine, ev.seat, ev.tile, self_draw=False)
                shooter_pay = ev.detail.get("shooter_pay", "?")
                shooter_label = _player_label(self._match, shooter) if shooter in SEAT_NAMES else "?"
                if ev.seat == self._my_seat:
                    msg = (
                        f"[bold green]🎉 You win! {shooter_label} dealt "
                        f"{_g(ev.tile)} {_gl(ev.tile)}! — {tai}  (+{shooter_pay} ★)[/bold green]"
                    )
                elif shooter == self._my_seat:
                    msg = (
                        f"[bold red]You shot! {_player_label(self._match, ev.seat)} wins from your "
                        f"{_g(ev.tile)} {_gl(ev.tile)} discard! — {tai}  (-{shooter_pay} ★)[/bold red]"
                    )
                else:
                    msg = (
                        f"[bold red]{_player_label(self._match, ev.seat)} wins! "
                        f"{shooter_label} dealt {_g(ev.tile)} {_gl(ev.tile)}! — {tai}  "
                        f"({shooter_label} pays {shooter_pay} ★)[/bold red]"
                    )
                log.write(msg)
                self.query_one("#status-bar", Label).update(msg)
                self.query_one("#discard-input", Input).disabled = True
                self._on_hand_over()
            elif ev.type == EventType.BONUS:
                if ev.seat == self._my_seat:
                    self._refresh_hand()
                else:
                    self._update_opponent_panels()
                log.write(
                    f"   [green]{_player_label(self._match, ev.seat)} draws bonus: "
                    f"{bonus_tile_name(ev.tile)}[/green]"
                )
            elif ev.type == EventType.MELD:
                meld_type = ev.detail.get("meld_type", "meld").capitalize()
                from_seat = ev.detail.get("from", -1)
                from_str = f" from {_player_label(self._match, from_seat)}" if from_seat in SEAT_NAMES else ""
                tiles_str = " ".join(f"{_g(t)} {_gl(t)}" for t in ev.detail.get("tiles", [ev.tile]))
                if ev.seat == self._my_seat:
                    self._refresh_hand()
                else:
                    self._update_opponent_panels()
                log.write(
                    f"   [bold cyan]{_player_label(self._match, ev.seat)} claims {meld_type}{from_str}! "
                    f"{tiles_str}[/bold cyan]"
                )
            elif ev.type == EventType.WALL_EXHAUSTED:
                if self._ai_timer:
                    self._ai_timer.stop()
                log.write("[yellow]Wall exhausted — draw game.[/yellow]")
                self.query_one("#status-bar", Label).update("[yellow]Draw game.[/yellow]")
                self.query_one("#discard-input", Input).disabled = True
                self._on_hand_over()

    def _on_hand_over(self) -> None:
        """Called when a hand ends. Preview the next rotation and enable Next Hand."""
        result = self._match.finish_hand()
        log = self.query_one("#game-log", RichLog)
        if result.rotated:
            new_my = WIND_NAMES.get(self._my_seat, "?")
            log.write(f"[cyan]Winds rotate (S→E, E→N, N→W, W→S) — you are now {new_my}.[/cyan]")
        else:
            log.write("[cyan]East wins — no rotation.[/cyan]")
        if self._match.is_complete:
            log.write("[bold green]Match complete![/bold green]")
            sorted_chips = sorted(self._match.chips.items(), key=lambda x: -x[1])
            for wind, chips in sorted_chips:
                log.write(f"  {_player_label(self._match, wind):20s}  ★{chips}")
            self.query_one("#discard-hint", Label).update("[bold]Match over. Press New Match.[/bold]")
        else:
            hint = (
                f"[bold]{self._match.round_label}, Hand {self._match.hand_number} — "
                f"you are {WIND_NAMES.get(self._my_seat, '?')}.[/bold]  Press Next Hand."
            )
            self.query_one("#discard-hint", Label).update(hint)
            self.query_one("#btn-next", Button).disabled = False

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

    def _update_opponent_labels(self) -> None:
        """Update opponent panel titles to reflect current wind assignments."""
        opp_seats = [s for s in [int(Wind.NORTH), int(Wind.SOUTH), int(Wind.WEST),
                                  int(Wind.EAST)] if s != self._my_seat]
        for i, seat in enumerate(opp_seats):
            self.query_one(f"#opp-title-{i}", Label).update(_player_label(self._match, seat))

    def _refresh_hand(self) -> None:
        if self._engine is None:
            return
        player = self._engine.players[self._my_seat]
        tiles = player.hand.concealed_tiles_list()
        s = shanten(player.hand.concealed, len(player.hand.melds))
        n_tiles = int(player.hand.concealed.sum())
        if s == -1:
            shanten_str = "[bold green]win![/bold green]"
        elif s == 0 and n_tiles == 13:
            shanten_str = f"[yellow]{_waiting_tai_label(player.hand, self._engine.prevailing_wind)}[/yellow]"
        elif s == 0:
            shanten_str = "[yellow]waiting![/yellow]"
        else:
            shanten_str = str(s)
        wall = self._engine.wall_remaining
        table_wind = SEAT_NAMES.get(self._engine.prevailing_wind, "?")
        my_chips = self._engine.chips.get(self._my_seat, 0)
        my_player_num = self._match.player_at.get(self._my_seat, "?") if self._match else "?"
        my_wind_name = WIND_NAMES.get(self._my_seat, "?")
        self.query_one("#hand-title", Label).update(
            f"[bold]Player {my_player_num} — Your Hand ({my_wind_name})[/bold]  —  Tiles away: {shanten_str}"
            f"  |  Wall: {wall}  Turn: {self._engine.turn_number}"
            f"  |  [yellow]★{my_chips}[/yellow]"
        )
        self.query_one("#table-wind-label", Label).update(
            f"[bold yellow]Table Wind: {table_wind}[/bold yellow]"
        )
        exposed = _exposed_row(player.hand)
        self.query_one("#meld-display", Static).update(
            exposed if exposed else "[dim]No exposed melds[/dim]"
        )
        self.query_one("#hand-display", Static).update(
            _hand_markup(tiles, self._drawn_tid)
        )

    def _update_opponent_panels(self) -> None:
        if self._engine is None:
            return
        opp_seats = [s for s in [int(Wind.NORTH), int(Wind.SOUTH), int(Wind.WEST),
                                   int(Wind.EAST)] if s != self._my_seat]
        for i, seat in enumerate(opp_seats):
            player = self._engine.players[seat]
            discs = _discards_markup(player.discards)
            exposed = _exposed_row(player.hand)
            opp_chips = self._engine.chips.get(seat, 0)
            lines: list[str] = [f"[yellow]★{opp_chips}[/yellow]"]
            if exposed:
                lines.append(exposed)
            lines.append(f"[dim]Discards:[/dim] {discs if discs else '—'}")
            self.query_one(f"#opp-disc-{i}", Static).update("\n".join(lines))

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
                s_str = "waiting" if r.shanten_after == 0 else ("win" if r.shanten_after == -1 else str(r.shanten_after))
                table.add_row(
                    str(i),
                    f"{tile_glyph(r.tile_id)} {tile_name(r.tile_id).upper()}",
                    s_str,
                    str(r.weighted_acceptance),
                    f"{r.danger_score:.2f}",
                    f"{r.tai_potential:.1f}",
                    f"{r.utility:.3f}",
                )

        self.app.call_from_thread(_update)

    def _clear_recommendations(self) -> None:
        self.query_one("#rec-table", DataTable).clear()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-new":
            self._new_match()
        elif event.button.id == "btn-next":
            self.query_one("#btn-next", Button).disabled = True
            self._start_hand(
                f"[bold]{self._match.round_label}, Hand {self._match.hand_number} "
                f"— you are {WIND_NAMES.get(self._my_seat, '?')}.[/bold]"
            )
        elif event.button.id == "btn-back":
            self.action_go_back()
        elif event.button.id == "btn-quit":
            self.app.exit()

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
