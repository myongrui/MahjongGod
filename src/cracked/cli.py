"""CLI for crackedMahjong — Singapore Mahjong discard optimizer."""

from __future__ import annotations

from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.table import Table

from cracked.tiles import (
    Wind, tile_id, tile_name, bonus_tile_id, bonus_tile_name,
    is_animal, new_hand_array,
)
from cracked.hand import HandState, Meld, MeldType
from cracked.game_state import GameState, PlayerView, save_state, load_state
from cracked.shanten import shanten
from cracked.optimizer import recommend_discard, adaptive_alpha
from cracked.opponent_model import model_all_opponents

console = Console()

SEAT_MAP = {
    "east": Wind.EAST, "e": Wind.EAST,
    "south": Wind.SOUTH, "s": Wind.SOUTH,
    "west": Wind.WEST, "w": Wind.WEST,
    "north": Wind.NORTH, "n": Wind.NORTH,
}
SEAT_NAMES = {27: "East", 28: "South", 29: "West", 30: "North"}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _tile_color(tid: int) -> str:
    if tid < 9:  return "green"
    if tid < 18: return "red"
    if tid < 27: return "cyan"
    if tid < 31: return "yellow"
    return "magenta"


def _rt(tid: int) -> str:
    """Rich-formatted tile name."""
    c = _tile_color(tid)
    return f"[{c}]{tile_name(tid)}[/{c}]"


def _render_hand(hand: HandState) -> str:
    parts = [_rt(t) for t in hand.concealed_tiles_list()]
    for m in hand.melds:
        parts.append("(" + " ".join(_rt(t) for t in m.tiles) + ")")
    return " ".join(parts)


def _parse_seat(s: str) -> int:
    key = s.strip().lower()
    if key not in SEAT_MAP:
        raise click.BadParameter(f"Unknown seat '{s}'. Use east/south/west/north.")
    return SEAT_MAP[key]


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """crackedMahjong — Singapore Mahjong discard optimizer."""


# ---------------------------------------------------------------------------
# new-game
# ---------------------------------------------------------------------------

@cli.command("new-game")
@click.option("--seat", default="east", show_default=True,
              help="Your seat wind (east/south/west/north)")
@click.option("--prevailing", default="east", show_default=True,
              help="Prevailing wind (east/south/west/north)")
def new_game(seat: str, prevailing: str):
    """Start a new game session."""
    my_seat = _parse_seat(seat)
    prev_wind = _parse_seat(prevailing)

    all_winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
    opponents = [PlayerView(seat=w) for w in all_winds if w != my_seat]

    state = GameState(
        my_hand=HandState(seat_wind=my_seat),
        my_seat=my_seat,
        prevailing_wind=prev_wind,
        opponents=opponents,
    )
    save_state(state)
    console.print(
        f"[bold]New game started.[/bold]  "
        f"You are [yellow]{SEAT_NAMES[my_seat]}[/yellow]  |  "
        f"Prevailing: [yellow]{SEAT_NAMES[prev_wind]}[/yellow]"
    )


# ---------------------------------------------------------------------------
# hand — set starting 13-tile hand
# ---------------------------------------------------------------------------

@cli.command("hand")
@click.argument("tiles", nargs=-1, required=True)
def set_hand(tiles):
    """Set your starting 13-tile hand.  Example: cracked hand b1 b2 b3 c4 c5 c6 ..."""
    state = load_state()
    try:
        tids = [tile_id(t) for t in tiles]
    except ValueError as e:
        raise click.BadParameter(str(e))

    expected = 13 - 3 * len(state.my_hand.melds)
    if len(tids) != expected:
        raise click.UsageError(
            f"Expected {expected} tiles (you have {len(state.my_hand.melds)} melds), got {len(tids)}."
        )

    arr = new_hand_array()
    for t in tids:
        arr[t] += 1

    state.my_hand.concealed = arr
    save_state(state)

    s = shanten(state.my_hand.concealed, len(state.my_hand.melds))
    console.print(f"Hand set: {_render_hand(state.my_hand)}  |  Shanten: [bold]{s}[/bold]")


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------

@cli.command("draw")
@click.argument("tile")
def draw_tile(tile: str):
    """Record that you drew TILE from the wall."""
    state = load_state()
    try:
        tid = tile_id(tile)
    except ValueError as e:
        raise click.BadParameter(str(e))

    state.my_hand.add_tile(tid)
    state.wall_tiles_remaining = max(0, state.wall_tiles_remaining - 1)
    state.turn_number += 1
    save_state(state)

    s = shanten(state.my_hand.concealed, len(state.my_hand.melds))
    console.print(
        f"Drew: {_rt(tid)}  |  "
        f"Hand ({state.my_hand.total_concealed} tiles): {_render_hand(state.my_hand)}  |  "
        f"Shanten: [bold]{s}[/bold]"
    )


# ---------------------------------------------------------------------------
# discard
# ---------------------------------------------------------------------------

@cli.command("discard")
@click.argument("tile")
@click.option("--by", default="me", show_default=True,
              help="Who discarded: 'me' or a seat name (east/south/west/north)")
def discard_tile(tile: str, by: str):
    """Record a tile discard.  Omit --by for your own discard."""
    state = load_state()
    try:
        tid = tile_id(tile)
    except ValueError as e:
        raise click.BadParameter(str(e))

    if by.lower() in ("me", "self"):
        state.my_hand.remove_tile(tid)
        console.print(f"You discarded: {_rt(tid)}")
    else:
        seat = _parse_seat(by)
        opp = state.opponent_by_seat(seat)
        opp.discards.append(tid)
        state.wall_tiles_remaining = max(0, state.wall_tiles_remaining - 1)
        state.turn_number += 1
        console.print(f"[yellow]{SEAT_NAMES[seat]}[/yellow] discarded: {_rt(tid)}")

    save_state(state)


# ---------------------------------------------------------------------------
# meld
# ---------------------------------------------------------------------------

@cli.command("meld")
@click.argument("meld_type", type=click.Choice(["pong", "kong", "chow"]))
@click.argument("tiles", nargs=-1, required=True)
@click.option("--by", required=True,
              help="Who made the meld: 'me' or a seat name")
@click.option("--concealed", is_flag=True, default=False,
              help="Concealed kong (ankan)")
def record_meld(meld_type: str, tiles, by: str, concealed: bool):
    """Record an exposed meld.  Example: cracked meld pong rd --by south"""
    state = load_state()
    try:
        tids = tuple(tile_id(t) for t in tiles)
    except ValueError as e:
        raise click.BadParameter(str(e))

    meld = Meld(type=MeldType(meld_type), tiles=tids, concealed=concealed)
    tile_display = " ".join(_rt(t) for t in tids)

    if by.lower() in ("me", "self"):
        for t in tids:
            if state.my_hand.concealed[t] > 0:
                state.my_hand.concealed[t] -= 1
        state.my_hand.add_meld(meld)
        console.print(f"You made: {meld_type.upper()} [{tile_display}]")
    else:
        seat = _parse_seat(by)
        opp = state.opponent_by_seat(seat)
        opp.melds.append(meld)
        console.print(f"[yellow]{SEAT_NAMES[seat]}[/yellow] made: {meld_type.upper()} [{tile_display}]")

    save_state(state)


# ---------------------------------------------------------------------------
# flower / animal
# ---------------------------------------------------------------------------

@cli.command("flower")
@click.argument("tile")
@click.option("--by", default="me", show_default=True,
              help="Who revealed it: 'me' or a seat name")
def record_flower(tile: str, by: str):
    """Record a flower, season, or animal tile.  Example: cracked flower f1  or  cracked flower cat"""
    state = load_state()
    try:
        bid = bonus_tile_id(tile)
    except ValueError as e:
        raise click.BadParameter(str(e))

    label = bonus_tile_name(bid)

    if by.lower() in ("me", "self"):
        if is_animal(bid):
            state.my_hand.animals.append(bid)
        else:
            state.my_hand.flowers.append(bid)
        console.print(f"You revealed: {label}")
    else:
        seat = _parse_seat(by)
        opp = state.opponent_by_seat(seat)
        opp.flowers.append(bid)
        console.print(f"[yellow]{SEAT_NAMES[seat]}[/yellow] revealed: {label}")

    save_state(state)


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

@cli.command("recommend")
@click.option("--deep", is_flag=True, default=False,
              help="Run Monte Carlo simulations for more accurate estimates")
@click.option("--games", default=200, show_default=True,
              help="Number of simulations per discard (used with --deep)")
@click.option("--log", "log_file", default=None, show_default=True,
              help="JSONL file to append simulation results for ML training "
                   "(used with --deep, default: data/game_log.jsonl)")
@click.option("--model", "model_path", default=None,
              help="Path to a trained DangerNet .pt model for prediction")
def recommend(deep: bool, games: int, log_file: str | None, model_path: str | None):
    """Show discard recommendations for your current 14-tile hand."""
    state = load_state()
    hand = state.my_hand
    expected = 14 - 3 * len(hand.melds)

    if hand.total_concealed != expected:
        console.print(
            f"[yellow]Need {expected} concealed tiles to recommend, "
            f"currently have {hand.total_concealed}.[/yellow]  "
            f"Run [bold]cracked draw TILE[/bold] first."
        )
        return

    results = recommend_discard(state)
    if not results:
        console.print("[red]No valid discards found.[/red]")
        return

    models = model_all_opponents(state)
    current_s = shanten(hand.concealed, len(hand.melds))
    alpha = adaptive_alpha(state, models, results[0].shanten_after)

    table = Table(
        title=(
            f"Discard Recommendations  "
            f"(shanten: {current_s}  |  α={alpha:.2f}  |  "
            f"wall: {state.wall_tiles_remaining})"
        ),
        box=box.ROUNDED, show_lines=False,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Discard", width=7)
    table.add_column("Shanten", justify="center", width=8)
    table.add_column("Tiles in", justify="center", width=8)
    table.add_column("Danger", justify="center", width=7)
    table.add_column("Cost", justify="center", width=6)
    table.add_column("Accepts", width=38)

    for i, r in enumerate(results[:8], 1):
        accept_tiles = sorted(r.acceptance.keys())
        accept_str = " ".join(_rt(t) for t in accept_tiles[:10])
        if len(accept_tiles) > 10:
            accept_str += f" [dim]+{len(accept_tiles)-10}[/dim]"

        s_color = "green" if r.shanten_after <= 0 else ("yellow" if r.shanten_after == 1 else "white")
        d_color = "red" if r.danger_score > 0.3 else ("yellow" if r.danger_score > 0.1 else "green")
        table.add_row(
            str(i),
            _rt(r.tile_id),
            f"[{s_color}]{r.shanten_after}[/{s_color}]",
            str(r.weighted_acceptance),
            f"[{d_color}]{r.danger_score:.2f}[/{d_color}]",
            f"{r.shooting_cost:.1f}",
            accept_str,
        )

    console.print(table)

    best = results[0]
    if best.shanten_after == -1:
        console.print(f"[bold green]Complete hand![/bold green] Discard {_rt(best.tile_id)} to win.")
    elif best.shanten_after == 0:
        console.print(
            f"[bold green]Tenpai![/bold green] "
            f"Best discard: {_rt(best.tile_id)}  |  "
            f"{best.weighted_acceptance} acceptance tiles  |  "
            f"Danger: {best.danger_score:.2f}"
        )
    else:
        console.print(
            f"Best discard: {_rt(best.tile_id)}  →  "
            f"shanten {best.shanten_after}  |  "
            f"{best.weighted_acceptance} acceptance tiles  |  "
            f"Danger: {best.danger_score:.2f}"
        )

    if model_path is not None:
        _show_model_predictions(state, results, model_path)

    if not deep:
        return

    # ------------------------------------------------------------------
    # Deep mode: Monte Carlo simulation
    # ------------------------------------------------------------------
    from cracked.simulator import run_simulation

    console.print()
    console.print(f"[dim]Running {games} simulations per discard…[/dim]")
    sim_results = run_simulation(state, n_games=games)

    sim_by_tile = {sr.tile_id: sr for sr in sim_results}

    sim_table = Table(
        title=f"Simulation Results  ({games} games each)",
        box=box.ROUNDED, show_lines=False,
    )
    sim_table.add_column("#", style="dim", width=3)
    sim_table.add_column("Discard", width=7)
    sim_table.add_column("Win%", justify="center", width=7)
    sim_table.add_column("Shoot%", justify="center", width=8)
    sim_table.add_column("E[Gain]", justify="center", width=8)

    for i, r in enumerate(results[:8], 1):
        sr = sim_by_tile.get(r.tile_id)
        if sr is None:
            continue
        gain = sr.expected_gain
        gain_color = "green" if gain > 0 else ("red" if gain < -0.5 else "yellow")
        sim_table.add_row(
            str(i),
            _rt(r.tile_id),
            f"{sr.win_rate * 100:.1f}%",
            f"{sr.shoot_rate * 100:.1f}%",
            f"[{gain_color}]{gain:+.2f}[/{gain_color}]",
        )

    console.print(sim_table)

    # Log simulation results for ML training
    _log_sim_results(state, sim_results, log_file)


def _log_sim_results(state, sim_results, log_file_str):
    """Append simulation results to the training log file."""
    from pathlib import Path
    from cracked.training.data import record_simulation, DEFAULT_LOG_FILE
    log_path = Path(log_file_str) if log_file_str else DEFAULT_LOG_FILE
    try:
        for sr in sim_results:
            record_simulation(state, sr, log_file=log_path)
        console.print(f"[dim]Logged {len(sim_results)} examples → {log_path}[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Warning: could not log training data: {exc}[/yellow]")


def _show_model_predictions(state, heuristic_results, model_path_str):
    """Show DangerNet predictions alongside the heuristic table."""
    import numpy as np
    from pathlib import Path

    try:
        import torch
        from cracked.training.model import load_model
        from cracked.training.features import extract_features
    except ImportError:
        console.print("[yellow]--model requires PyTorch: pip install 'cracked[ml]'[/yellow]")
        return

    path = Path(model_path_str)
    if not path.exists():
        console.print(f"[red]Model not found: {path}[/red]")
        return

    model = load_model(path)

    rows = []
    for r in heuristic_results[:8]:
        feat = extract_features(state, r.tile_id)
        with torch.no_grad():
            pred = model(torch.tensor(feat).unsqueeze(0)).squeeze(0).numpy()
        rows.append((r.tile_id, float(pred[0]), float(pred[1]), float(pred[2])))

    rows.sort(key=lambda x: -x[3])  # sort by predicted expected_gain

    model_table = Table(
        title=f"DangerNet Predictions  ({path.name})",
        box=box.ROUNDED, show_lines=False,
    )
    model_table.add_column("#", style="dim", width=3)
    model_table.add_column("Discard", width=7)
    model_table.add_column("Win%", justify="center", width=7)
    model_table.add_column("Shoot%", justify="center", width=8)
    model_table.add_column("E[Gain]", justify="center", width=8)

    for i, (tid, win_r, shoot_r, gain) in enumerate(rows, 1):
        gain_color = "green" if gain > 0 else ("red" if gain < -0.5 else "yellow")
        model_table.add_row(
            str(i),
            _rt(tid),
            f"{win_r * 100:.1f}%",
            f"{shoot_r * 100:.1f}%",
            f"[{gain_color}]{gain:+.2f}[/{gain_color}]",
        )

    console.print()
    console.print(model_table)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command("status")
def status():
    """Show the full current game state."""
    state = load_state()
    hand = state.my_hand
    s = shanten(hand.concealed, len(hand.melds))

    console.rule("[bold]crackedMahjong[/bold]")
    console.print(
        f"You: [yellow]{SEAT_NAMES[state.my_seat]}[/yellow]  |  "
        f"Prevailing: [yellow]{SEAT_NAMES[state.prevailing_wind]}[/yellow]  |  "
        f"Wall: {state.wall_tiles_remaining} tiles  |  "
        f"Turn: {state.turn_number}"
    )
    console.print()

    console.print(f"[bold]Your hand[/bold]  (shanten: [bold]{s}[/bold])")
    console.print(f"  {_render_hand(hand)}")
    if hand.flowers:
        console.print(f"  Flowers/Seasons: " + "  ".join(bonus_tile_name(f) for f in hand.flowers))
    if hand.animals:
        console.print(f"  Animals: " + "  ".join(bonus_tile_name(a) for a in hand.animals))
    console.print()

    for opp in state.opponents:
        console.print(f"[bold yellow]{SEAT_NAMES[opp.seat]}[/bold yellow]"
                      f"  ({len(opp.discards)} discards, {len(opp.melds)} melds)")
        if opp.discards:
            console.print("  Discards: " + " ".join(_rt(t) for t in opp.discards))
        if opp.melds:
            meld_parts = []
            for m in opp.melds:
                meld_parts.append(m.type.value.upper() + "[" + " ".join(_rt(t) for t in m.tiles) + "]")
            console.print("  Melds: " + "  ".join(meld_parts))
        if opp.flowers:
            console.print("  Bonus: " + "  ".join(bonus_tile_name(f) for f in opp.flowers))
        console.print()
