"""Textual TUI for crackedMahjong — Singapore Mahjong Discard Optimizer."""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, Button, DataTable, Label, Select
from textual import work

from cracked.tiles import tile_id, tile_name, new_hand_array, NTILES, Wind
from cracked.hand import HandState
from cracked.game_state import GameState, PlayerView
from cracked.shanten import shanten
from cracked.optimizer import recommend_discard
from cracked.simulator import run_simulation


SEAT_OPTIONS = [("East", "East"), ("South", "South"), ("West", "West"), ("North", "North")]
SEAT_MAP = {"East": Wind.EAST, "South": Wind.SOUTH, "West": Wind.WEST, "North": Wind.NORTH}


class CrackedMahjongApp(App):
    """Singapore Mahjong Discard Optimizer TUI."""

    TITLE = "crackedMahjong"
    SUB_TITLE = "Singapore Mahjong Discard Optimizer"

    CSS = """
    Screen {
        background: $surface;
    }

    #layout {
        height: 1fr;
    }

    #left-panel {
        width: 40;
        padding: 1 2;
        border: round $primary;
        margin: 0 1 0 0;
    }

    #right-panel {
        width: 1fr;
        border: round $primary;
        padding: 1 2;
    }

    .panel-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .field-label {
        color: $text-muted;
        margin-top: 1;
    }

    Select { width: 100%; }
    Input  { width: 100%; }

    #btn-row {
        margin-top: 1;
        height: 3;
    }

    #btn-recommend { margin-right: 1; }

    #status {
        margin-top: 1;
        color: $warning;
        text-style: italic;
        height: 3;
    }

    .section-label {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }

    #hand-display {
        color: $text;
        margin-bottom: 1;
    }

    DataTable {
        height: auto;
        max-height: 11;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("r", "recommend", "Recommend"),
        ("s", "simulate", "Simulate"),
        ("q", "quit", "Quit"),
    ]

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="layout"):
            with Vertical(id="left-panel"):
                yield Label("Setup", classes="panel-title")

                yield Label("Your seat", classes="field-label")
                yield Select(SEAT_OPTIONS, value="East", id="seat-select", allow_blank=False)

                yield Label("Prevailing wind", classes="field-label")
                yield Select(SEAT_OPTIONS, value="East", id="prev-select", allow_blank=False)

                yield Label("Hand — 13 tiles (space-separated)", classes="field-label")
                yield Input(
                    value="b1 b2 b3 c4 c5 c6 d7 d8 d9 ew ew ew rd",
                    id="hand-input",
                )

                yield Label("Draw tile (14th)", classes="field-label")
                yield Input(value="gd", id="draw-input")

                yield Label("Simulations per discard", classes="field-label")
                yield Input(value="200", id="games-input")

                with Horizontal(id="btn-row"):
                    yield Button("Recommend", id="btn-recommend", variant="primary")
                    yield Button("Simulate", id="btn-simulate")
                    yield Button("Quit", id="btn-quit", variant="error")

                yield Label("Press R to recommend, S to simulate", id="status")

            with ScrollableContainer(id="right-panel"):
                yield Label("Hand", classes="section-label")
                yield Label("—", id="hand-display")
                yield Label("Recommendations", classes="section-label")
                yield DataTable(id="rec-table")
                yield Label("Simulation", classes="section-label")
                yield DataTable(id="sim-table")

        yield Footer()

    def on_mount(self) -> None:
        rec = self.query_one("#rec-table", DataTable)
        rec.add_columns("#", "Discard", "Tiles away", "Accepts", "Danger", "Cost", "Utility")
        rec.cursor_type = "none"

        sim = self.query_one("#sim-table", DataTable)
        sim.add_columns("Discard", "Win%", "Shoot%", "Draw%", "E[Gain]")
        sim.cursor_type = "none"

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-recommend":
            self.action_recommend()
        elif event.button.id == "btn-simulate":
            self.action_simulate()
        elif event.button.id == "btn-quit":
            self.exit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_state(self) -> GameState | None:
        seat_val = self.query_one("#seat-select", Select).value
        prev_val = self.query_one("#prev-select", Select).value
        hand_text = self.query_one("#hand-input", Input).value.strip()
        draw_text = self.query_one("#draw-input", Input).value.strip()

        if seat_val is Select.BLANK or prev_val is Select.BLANK:
            self._set_status("Select seat and prevailing wind")
            return None

        my_seat = SEAT_MAP[str(seat_val)]
        prev_wind = SEAT_MAP[str(prev_val)]

        tokens = hand_text.split()
        draw_tokens = draw_text.split()

        if len(tokens) != 13:
            self._set_status(f"Need 13 hand tiles, got {len(tokens)}")
            return None
        if len(draw_tokens) != 1:
            self._set_status("Enter exactly 1 draw tile")
            return None

        try:
            hand_tids = [tile_id(t) for t in tokens]
            draw_tid = tile_id(draw_tokens[0])
        except ValueError as exc:
            self._set_status(str(exc))
            return None

        arr = new_hand_array()
        for t in hand_tids + [draw_tid]:
            arr[t] += 1

        opponents = [PlayerView(seat=int(w)) for w in Wind if int(w) != my_seat]
        return GameState(
            my_hand=HandState(concealed=arr, seat_wind=my_seat),
            my_seat=my_seat,
            prevailing_wind=prev_wind,
            opponents=opponents,
        )

    def _set_status(self, msg: str) -> None:
        self.query_one("#status", Label).update(msg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_recommend(self) -> None:
        state = self._build_state()
        if state is None:
            return

        s = shanten(state.my_hand.concealed, 0)
        tiles_flat = [t for t in range(NTILES) for _ in range(int(state.my_hand.concealed[t]))]
        hand_str = " ".join(tile_name(t) for t in tiles_flat)
        self.query_one("#hand-display", Label).update(f"{hand_str}  |  Tiles away: {s}")

        try:
            recs = recommend_discard(state)
        except Exception as exc:
            self._set_status(f"Error: {exc}")
            return

        rec_table = self.query_one("#rec-table", DataTable)
        rec_table.clear()

        for i, r in enumerate(recs[:8], 1):
            accept_keys = sorted(r.acceptance.keys())[:5]
            accept_str = " ".join(tile_name(t) for t in accept_keys)
            if len(r.acceptance) > 5:
                accept_str += f" +{len(r.acceptance) - 5}"

            if r.shanten_after == -1:
                shanten_str = "win!"
            elif r.shanten_after == 0:
                shanten_str = "waiting"
            else:
                shanten_str = str(r.shanten_after)

            rec_table.add_row(
                str(i),
                tile_name(r.tile_id),
                shanten_str,
                str(r.weighted_acceptance),
                f"{r.danger_score:.2f}",
                f"{r.shooting_cost:.2f}",
                f"{r.utility:.3f}",
            )

        if recs:
            best = recs[0]
            if best.shanten_after == -1:
                self._set_status(f"Win! Discard {tile_name(best.tile_id)}")
            elif best.shanten_after == 0:
                self._set_status(
                    f"Waiting! Best: {tile_name(best.tile_id)} — "
                    f"{best.weighted_acceptance} acceptance tiles"
                )
            else:
                self._set_status(
                    f"Best: {tile_name(best.tile_id)} → "
                    f"tiles away {best.shanten_after}, {best.weighted_acceptance} accepts"
                )

    def action_simulate(self) -> None:
        state = self._build_state()
        if state is None:
            return

        try:
            n_games = max(50, min(2000, int(self.query_one("#games-input", Input).value)))
        except ValueError:
            n_games = 200

        self._run_simulation(state, n_games)

    @work(thread=True)
    def _run_simulation(self, state: GameState, n_games: int) -> None:
        self.call_from_thread(self._set_status, f"Running {n_games} simulations per discard…")

        try:
            results = run_simulation(state, n_games=n_games)
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Simulation error: {exc}")
            return

        def _update() -> None:
            sim_table = self.query_one("#sim-table", DataTable)
            sim_table.clear()
            for sr in results[:8]:
                draw_pct = sr.draw_count / max(sr.n_games, 1) * 100
                sim_table.add_row(
                    tile_name(sr.tile_id),
                    f"{sr.win_rate * 100:.1f}%",
                    f"{sr.shoot_rate * 100:.1f}%",
                    f"{draw_pct:.1f}%",
                    f"{sr.expected_gain:+.2f}",
                )
            if results:
                best = results[0]
                self._set_status(
                    f"Sim best: {tile_name(best.tile_id)} — "
                    f"win {best.win_rate * 100:.1f}%, "
                    f"shoot {best.shoot_rate * 100:.1f}%, "
                    f"E[gain] {best.expected_gain:+.2f}"
                )

        self.call_from_thread(_update)


def main() -> None:
    CrackedMahjongApp().run()


if __name__ == "__main__":
    main()
