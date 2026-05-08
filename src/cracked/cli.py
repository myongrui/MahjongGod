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
from cracked.shanten import shanten, best_discards

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
def recommend():
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

    current_s = shanten(hand.concealed, len(hand.melds))
    unknown = state.unknown_tiles()
    results = best_discards(hand.concealed, unknown, len(hand.melds))

    if not results:
        console.print("[red]No valid discards found.[/red]")
        return

    table = Table(
        title=f"Discard Recommendations  (current shanten: {current_s})",
        box=box.ROUNDED, show_lines=False,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Discard", width=7)
    table.add_column("Shanten", justify="center", width=8)
    table.add_column("Tiles in", justify="center", width=8)
    table.add_column("Accepts", width=45)

    for i, r in enumerate(results[:8], 1):
        tid = r["tile_id"]
        s_after = r["shanten_after"]
        acc = r["acceptance"]
        weighted = r["weighted_acceptance"]

        accept_tiles = sorted(acc.keys())
        accept_str = " ".join(_rt(t) for t in accept_tiles[:12])
        if len(accept_tiles) > 12:
            accept_str += f" [dim]+{len(accept_tiles)-12}[/dim]"

        s_color = "green" if s_after <= 0 else ("yellow" if s_after == 1 else "white")
        table.add_row(
            str(i),
            _rt(tid),
            f"[{s_color}]{s_after}[/{s_color}]",
            str(weighted),
            accept_str,
        )

    console.print(table)

    best = results[0]
    if best["shanten_after"] == -1:
        console.print(f"[bold green]Complete hand![/bold green] Discard {_rt(best['tile_id'])} to win.")
    elif best["shanten_after"] == 0:
        console.print(
            f"[bold green]Tenpai![/bold green] "
            f"Best discard: {_rt(best['tile_id'])}  |  "
            f"{best['weighted_acceptance']} acceptance tiles"
        )
    else:
        console.print(
            f"Best discard: {_rt(best['tile_id'])}  →  "
            f"shanten {best['shanten_after']} with {best['weighted_acceptance']} acceptance tiles"
        )


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
