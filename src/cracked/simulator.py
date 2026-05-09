"""
Monte Carlo game simulator for Singapore Mahjong.

Simulates complete games forward from the current state to estimate
win and shoot probabilities for each candidate discard.

V1 simplifications (to be refined in later phases):
  - No meld claiming during simulation (pong/chow from discards not modelled)
  - Simplified tai estimation for payment calculation
  - All players use the same pure-shanten discard heuristic
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cracked.tiles import NTILES, is_suited, is_honor, suit_of, DRAGON_START, DRAGON_END
from cracked.game_state import GameState
from cracked.shanten import shanten


# ---------------------------------------------------------------------------
# Lightweight simulation structures
# ---------------------------------------------------------------------------

@dataclass
class SimHand:
    """Minimal hand state used inside a simulation."""
    concealed: np.ndarray   # 34-element int8 tile-count array
    n_melds: int            # number of complete exposed sets already held
    seat: int               # Wind constant (27–30)

    def is_winner(self) -> bool:
        return shanten(self.concealed, self.n_melds) == -1

    def can_win_from(self, tid: int) -> bool:
        """Would drawing tid complete this hand?"""
        self.concealed[tid] += 1
        win = self.is_winner()
        self.concealed[tid] -= 1
        return win


@dataclass
class GameResult:
    """Outcome of one simulated game from our perspective."""
    winner_seat: Optional[int]   # None = wall exhausted with no winner
    is_self_draw: bool
    shooter_seat: Optional[int]  # None for self-draw or no-winner
    tai: int                     # estimated tai of the winning hand
    my_net: float                # our net gain/loss in base-payment units


@dataclass
class SimulationResult:
    """Aggregated statistics from N simulated games for one candidate discard."""
    tile_id: int
    n_games: int
    win_count: int       # games where we won
    shoot_count: int     # games where we dealt into an opponent's win
    draw_count: int      # games that ended with an empty wall (no winner)
    total_net: float     # sum of net gains across all games

    @property
    def win_rate(self) -> float:
        return self.win_count / max(self.n_games, 1)

    @property
    def shoot_rate(self) -> float:
        return self.shoot_count / max(self.n_games, 1)

    @property
    def expected_gain(self) -> float:
        """Average net gain per game in base-payment units."""
        return self.total_net / max(self.n_games, 1)


# ---------------------------------------------------------------------------
# AI heuristic
# ---------------------------------------------------------------------------

def heuristic_discard(hand: SimHand) -> int:  # public alias
    return _heuristic_discard(hand)


def _heuristic_discard(hand: SimHand) -> int:
    """
    Discard the tile that minimises shanten.
    Break ties by preferring isolated honor tiles (hardest to improve).
    """
    best_tid = -1
    best_shanten = 99
    best_is_honor = False

    for tid in range(NTILES):
        if hand.concealed[tid] == 0:
            continue
        hand.concealed[tid] -= 1
        s = shanten(hand.concealed, hand.n_melds)
        hand.concealed[tid] += 1

        isolated_honor = is_honor(tid) and hand.concealed[tid] == 1
        if (s < best_shanten
                or (s == best_shanten and isolated_honor and not best_is_honor)):
            best_shanten = s
            best_tid = tid
            best_is_honor = isolated_honor

    return best_tid


# ---------------------------------------------------------------------------
# Payment estimation (fast approximation for simulation)
# ---------------------------------------------------------------------------

def _estimate_tai(concealed: np.ndarray, n_melds: int) -> int:
    """Rough tai estimate from visible hand structure."""
    tiles = [tid for tid in range(NTILES) for _ in range(int(concealed[tid]))]
    tai = 1  # baseline

    suits = {suit_of(t) for t in tiles if is_suited(t)}
    honors = any(is_honor(t) for t in tiles)

    if not honors and len(suits) == 1:
        tai += 3   # full flush
    elif honors and len(suits) <= 1:
        tai += 1   # half flush or honor-heavy

    for t in range(DRAGON_START, DRAGON_END):
        if concealed[t] >= 3:
            tai += 1

    return max(tai, 1)


def _payment(tai: int) -> float:
    """Base-unit payment for a given tai level (doubles each tai)."""
    return pow(2.0, tai - 1.0)


# ---------------------------------------------------------------------------
# Hand dealing
# ---------------------------------------------------------------------------

def _deal_hands(
    unknown: np.ndarray,
    opp_concealed_counts: list[int],
    rng: random.Random,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Randomly distribute unknown tiles to opponents and a wall.

    Returns (list of concealed arrays per opponent, flat wall tile list).
    """
    # Flatten unknown tiles to a shuffled list
    pool: list[int] = []
    for tid in range(NTILES):
        pool.extend([tid] * int(unknown[tid]))
    rng.shuffle(pool)

    opp_hands: list[np.ndarray] = []
    idx = 0
    for count in opp_concealed_counts:
        arr = np.zeros(NTILES, dtype=np.int8)
        take = min(count, len(pool) - idx)
        for t in pool[idx: idx + take]:
            arr[t] += 1
        idx += take
        opp_hands.append(arr)

    wall = pool[idx:]
    return opp_hands, wall


# ---------------------------------------------------------------------------
# Single game simulation
# ---------------------------------------------------------------------------

def _play_one_game(
    my_hand: SimHand,
    opp_hands: list[SimHand],
    wall: list[int],
    my_seat: int,
) -> GameResult:
    """
    Simulate one complete game turn-by-turn until someone wins or wall empties.

    Players are ordered by seat wind (East first).
    Pong/chow claiming is not modelled in V1 — players only win by self-draw
    or from another player's discard.
    """
    all_seats = sorted([my_seat] + [h.seat for h in opp_hands])
    hands: dict[int, SimHand] = {my_seat: my_hand}
    for h in opp_hands:
        hands[h.seat] = h

    wall_idx = 0
    max_rounds = 40  # Singapore games rarely exceed 30 rounds

    for _ in range(max_rounds):
        for seat in all_seats:
            if wall_idx >= len(wall):
                return GameResult(None, False, None, 0, 0.0)

            h = hands[seat]
            drawn = wall[wall_idx]
            wall_idx += 1
            h.concealed[drawn] += 1

            # Self-draw win check
            if h.is_winner():
                tai = _estimate_tai(h.concealed, h.n_melds)
                pay = _payment(tai)
                if seat == my_seat:
                    return GameResult(seat, True, None, tai, pay * 3.0)
                else:
                    return GameResult(seat, True, None, tai, -pay)

            # Discard
            discard = _heuristic_discard(h)
            h.concealed[discard] -= 1

            # Ron (discard) win check — all other players
            for claimer_seat in all_seats:
                if claimer_seat == seat:
                    continue
                claimer = hands[claimer_seat]
                if claimer.can_win_from(discard):
                    tai = _estimate_tai(claimer.concealed, claimer.n_melds)
                    pay = _payment(tai)
                    if seat == my_seat:
                        # We shot: shooter pays all (3 losers)
                        return GameResult(claimer_seat, False, seat, tai, -pay * 3.0)
                    elif claimer_seat == my_seat:
                        # We won: shooter pays us
                        return GameResult(my_seat, False, seat, tai, pay * 3.0)
                    else:
                        return GameResult(claimer_seat, False, seat, tai, 0.0)

    return GameResult(None, False, None, 0, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_from_state(
    state: GameState,
    candidate_discard: int,
    n_games: int = 200,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Simulate N games after committing to discarding candidate_discard.

    Returns aggregated win/shoot statistics.
    """
    rng = random.Random(seed)

    # Apply our discard
    my_concealed = state.my_hand.concealed.copy()
    if my_concealed[candidate_discard] <= 0:
        raise ValueError(f"Tile {candidate_discard} not in concealed hand")
    my_concealed[candidate_discard] -= 1
    my_n_melds = len(state.my_hand.melds)

    # Estimate how many concealed tiles each opponent holds
    opp_counts = [max(1, 13 - 3 * len(opp.melds)) for opp in state.opponents]

    # Unknown tiles after our discard (the discarded tile is now visible)
    unknown = state.unknown_tiles().copy()
    if unknown[candidate_discard] > 0:
        unknown[candidate_discard] -= 1

    win_count = shoot_count = draw_count = 0
    total_net = 0.0

    for _ in range(n_games):
        opp_arrays, wall = _deal_hands(unknown, opp_counts, rng)

        my_hand = SimHand(my_concealed.copy(), my_n_melds, state.my_seat)
        opp_hands = [
            SimHand(arr.copy(), len(state.opponents[i].melds), state.opponents[i].seat)
            for i, arr in enumerate(opp_arrays)
        ]

        result = _play_one_game(my_hand, opp_hands, wall, state.my_seat)

        total_net += result.my_net
        if result.winner_seat == state.my_seat:
            win_count += 1
        elif result.shooter_seat == state.my_seat:
            shoot_count += 1
        elif result.winner_seat is None:
            draw_count += 1

    return SimulationResult(
        tile_id=candidate_discard,
        n_games=n_games,
        win_count=win_count,
        shoot_count=shoot_count,
        draw_count=draw_count,
        total_net=total_net,
    )


def run_simulation(
    state: GameState,
    n_games: int = 200,
    seed: Optional[int] = None,
) -> list[SimulationResult]:
    """
    Run simulations for every valid discard from the current 14-tile hand.

    Returns results sorted by expected_gain descending.
    """
    hand = state.my_hand
    n_melds = len(hand.melds)
    expected = 14 - 3 * n_melds
    if hand.total_concealed != expected:
        raise ValueError(
            f"Expected {expected} concealed tiles, got {hand.total_concealed}"
        )

    results: list[SimulationResult] = []
    for tid in range(NTILES):
        if hand.concealed[tid] == 0:
            continue
        # Use different seed per tile so simulations are independent but reproducible
        tile_seed = None if seed is None else seed + tid
        sr = simulate_from_state(state, tid, n_games=n_games, seed=tile_seed)
        results.append(sr)

    results.sort(key=lambda r: -r.expected_gain)
    return results
