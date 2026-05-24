"""
Monte Carlo game simulator for Singapore Mahjong.

Simulates complete games forward from the current state to estimate
win and shoot probabilities for each candidate discard.

Remaining simplifications vs the full engine:
  - Tai is estimated (suit/dragon heuristic) rather than using the full scoring engine
  - Agent never claims pong/kong/chow on opponent discards (only chooses what to discard)
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
from cracked.scoring import chip_payment

_MIN_TAI = 1  # minimum tai required to declare a win


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


def _wants_pong_sim(hand: SimHand, tile: int) -> bool:
    """True if ponging maintains or improves best post-pong shanten."""
    if hand.concealed[tile] < 2:
        return False
    current_s = shanten(hand.concealed, hand.n_melds)
    if current_s == -1:
        return False
    hand.concealed[tile] -= 2
    hand.n_melds += 1
    eye = np.eye(NTILES, dtype=np.int8)
    best_s = min(
        (shanten(hand.concealed - eye[t], hand.n_melds)
         for t in range(NTILES) if hand.concealed[t] > 0),
        default=current_s,
    )
    hand.concealed[tile] += 2
    hand.n_melds -= 1
    return best_s <= current_s


def _wants_kong_sim(hand: SimHand, tile: int) -> bool:
    """True if konging doesn't significantly worsen shanten (+1 allowed; replacement compensates)."""
    if hand.concealed[tile] < 3:
        return False
    current_s = shanten(hand.concealed, hand.n_melds)
    if current_s == -1:
        return False
    hand.concealed[tile] -= 3
    hand.n_melds += 1
    eye = np.eye(NTILES, dtype=np.int8)
    best_s = min(
        (shanten(hand.concealed - eye[t], hand.n_melds)
         for t in range(NTILES) if hand.concealed[t] > 0),
        default=current_s,
    )
    hand.concealed[tile] += 3
    hand.n_melds -= 1
    return best_s <= current_s + 1


def _pick_best_chow_sim(hand: SimHand, tile: int) -> Optional[tuple[int, int, int]]:
    """Best chow option that strictly improves shanten, or None."""
    if tile >= 27:
        return None
    suit_start = (tile // 9) * 9
    rank = tile % 9
    current_s = shanten(hand.concealed, hand.n_melds)
    best_option: Optional[tuple[int, int, int]] = None
    best_s = current_s  # chow requires strict improvement
    eye = np.eye(NTILES, dtype=np.int8)
    for low in (rank - 2, rank - 1, rank):
        if low < 0 or low + 2 > 8:
            continue
        t1, t2, t3 = suit_start + low, suit_start + low + 1, suit_start + low + 2
        if not all(hand.concealed[t] > 0 for t in (t1, t2, t3) if t != tile):
            continue
        test = hand.concealed.copy()
        for t in (t1, t2, t3):
            if t != tile:
                test[t] -= 1
        test_melds = hand.n_melds + 1
        min_s = min(
            (shanten(test - eye[t], test_melds) for t in range(NTILES) if test[t] > 0),
            default=current_s,
        )
        if min_s < best_s:
            best_s = min_s
            best_option = (t1, t2, t3)
    return best_option


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
    Claims: pong/kong (clockwise priority) and chow (left player only).
    Dead wall: stops at 15 tiles remaining, matching Singapore rules.
    """
    all_seats = sorted([my_seat] + [h.seat for h in opp_hands])
    hands: dict[int, SimHand] = {my_seat: my_hand}
    for h in opp_hands:
        hands[h.seat] = h

    n = len(all_seats)
    wall_idx = 0
    wall_remaining = len(wall)
    seat_idx = 0
    max_turns = 40 * n * 2

    def _ron_result(shooter: int, tile: int) -> Optional[GameResult]:
        for cs in all_seats:
            if cs == shooter:
                continue
            if hands[cs].can_win_from(tile):
                tai = _estimate_tai(hands[cs].concealed, hands[cs].n_melds)
                if tai < _MIN_TAI:
                    continue
                shooter_pay, _ = chip_payment(tai)
                if shooter == my_seat:
                    return GameResult(cs, False, shooter, tai, -shooter_pay)
                elif cs == my_seat:
                    return GameResult(my_seat, False, shooter, tai, shooter_pay)
                else:
                    return GameResult(cs, False, shooter, tai, 0.0)
        return None

    for _ in range(max_turns):
        if wall_remaining <= 15 or wall_idx >= len(wall):
            return GameResult(None, False, None, 0, 0.0)

        seat = all_seats[seat_idx]
        drawn = wall[wall_idx]
        wall_idx += 1
        wall_remaining -= 1
        h = hands[seat]
        h.concealed[drawn] += 1

        # Self-draw win check
        if h.is_winner():
            tai = _estimate_tai(h.concealed, h.n_melds)
            if tai >= _MIN_TAI:
                _, zimo_pay = chip_payment(tai)
                net = zimo_pay * 3.0 if seat == my_seat else -zimo_pay
                return GameResult(seat, True, None, tai, net)

        # Discard
        discard = _heuristic_discard(h)
        h.concealed[discard] -= 1

        # Ron check
        ron = _ron_result(seat, discard)
        if ron is not None:
            return ron

        # Claims: kong/pong (clockwise priority) then chow (left player only)
        claimed = False
        for offset in range(1, n):
            cs = all_seats[(seat_idx + offset) % n]
            ch = hands[cs]
            if ch.concealed[discard] >= 3 and _wants_kong_sim(ch, discard):
                ch.concealed[discard] -= 3
                ch.n_melds += 1
                rep = None
                while wall_idx < len(wall) and wall_remaining > 15:
                    rtid = wall[wall_idx]
                    wall_idx += 1
                    wall_remaining -= 1
                    if rtid < 34:
                        rep = rtid
                        break
                if rep is None:
                    return GameResult(None, False, None, 0, 0.0)
                ch.concealed[rep] += 1
                if ch.is_winner():
                    tai = _estimate_tai(ch.concealed, ch.n_melds)
                    if tai >= _MIN_TAI:
                        _, zimo_pay = chip_payment(tai)
                        net = zimo_pay * 3.0 if cs == my_seat else -zimo_pay
                        return GameResult(cs, True, None, tai, net)
                kong_disc = _heuristic_discard(ch)
                ch.concealed[kong_disc] -= 1
                ron = _ron_result(cs, kong_disc)
                if ron is not None:
                    return ron
                seat_idx = (seat_idx + offset + 1) % n
                claimed = True
                break
            elif ch.concealed[discard] >= 2 and _wants_pong_sim(ch, discard):
                ch.concealed[discard] -= 2
                ch.n_melds += 1
                pong_disc = _heuristic_discard(ch)
                ch.concealed[pong_disc] -= 1
                ron = _ron_result(cs, pong_disc)
                if ron is not None:
                    return ron
                seat_idx = (seat_idx + offset + 1) % n
                claimed = True
                break

        if not claimed:
            left_cs = all_seats[(seat_idx + 1) % n]
            chow = _pick_best_chow_sim(hands[left_cs], discard)
            if chow is not None:
                ch = hands[left_cs]
                for t in chow:
                    if t != discard:
                        ch.concealed[t] -= 1
                ch.n_melds += 1
                chow_disc = _heuristic_discard(ch)
                ch.concealed[chow_disc] -= 1
                ron = _ron_result(left_cs, chow_disc)
                if ron is not None:
                    return ron
                seat_idx = (seat_idx + 2) % n
                claimed = True

        if not claimed:
            seat_idx = (seat_idx + 1) % n

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
