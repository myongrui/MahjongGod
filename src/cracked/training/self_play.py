"""
Self-play RL training for Singapore Mahjong.

Trains an ActorCritic policy network using REINFORCE with baseline
(a simplified, single-clip PPO step).  The learning agent plays as East
against three heuristic opponents from simulator.py.

Usage:
    python -m cracked.training.self_play --episodes 5000 --out models/policy.pt

Requires PyTorch: pip install 'cracked[ml]'
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from cracked.tiles import NTILES, Wind
from cracked.hand import HandState
from cracked.game_state import GameState, PlayerView
from cracked.simulator import (
    SimHand, _heuristic_discard, _estimate_tai, _MIN_TAI,
    _wants_pong_sim, _wants_kong_sim, _pick_best_chow_sim,
)
from cracked.scoring import chip_payment
from cracked.optimizer import hand_tai_potential
from cracked.tiles_away import tiles_away
from cracked.danger import expected_shooting_cost
from cracked.opponent_model import model_all_opponents
from cracked.training.features import N_STATE_FEATURES, extract_state_features

_WIND_ORDER = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]

HIDDEN_SIZE = 256
N_LAYERS = 2
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_ROUNDS = 40  # game turn cap (same as simulator)
SHAPING_SCALE = 1.0  # potential-based reward shaping weight
DEFENSE_WEIGHT = 0.02  # immediate penalty per step scaled by expected shooting cost


def _compute_potential(
    concealed: np.ndarray,
    n_melds: int,
    seat_wind: int,
    prevailing_wind: int,
    unknown_tiles: np.ndarray | None = None,
) -> float:
    """
    State potential for PBRS (potential-based reward shaping).

    Combines tiles_away proximity and tai ceiling so the agent receives a
    dense learning signal at every step rather than only at game end.
    """
    s = tiles_away(concealed, n_melds)
    tiles_away_val = (7 - s) / 8.0
    tai_pot = hand_tai_potential(concealed, [], seat_wind, prevailing_wind)
    tai_val = min(tai_pot / 6.0, 1.0)
    return 1.5 * tiles_away_val + 1.0 * tai_val


def oracle_threat(opp_sims: list) -> float:
    """
    Privileged perfect-information threat in [0, 1]: how close the closest
    opponent is to a complete hand, read from their actual concealed tiles.

    Used for Suphx-style oracle guiding: early in training the agent gets a
    perfect-information caution signal (opponents are dangerous *right now*),
    which is annealed to zero so the final policy relies only on observable
    state. A waiting opponent (tiles_away 0) returns 1.0; further away decays.
    """
    best = 99
    for h in opp_sims:
        s = tiles_away(h.concealed, h.n_melds)
        if s < best:
            best = s
    if best >= 99:
        return 0.0
    return 1.0 / (1.0 + max(best, 0))


def _wants_pong(hand: SimHand, tile: int) -> bool:
    """True if claiming a pong of tile maintains or improves best post-pong tiles_away."""
    if hand.concealed[tile] < 2:
        return False
    s_before = tiles_away(hand.concealed, hand.n_melds)
    hand.concealed[tile] -= 2
    hand.n_melds += 1
    best_s = 99
    for t in range(NTILES):
        if hand.concealed[t] == 0:
            continue
        hand.concealed[t] -= 1
        s = tiles_away(hand.concealed, hand.n_melds)
        hand.concealed[t] += 1
        if s < best_s:
            best_s = s
    hand.concealed[tile] += 2
    hand.n_melds -= 1
    return best_s <= s_before


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for self-play. Run: pip install 'cracked[ml]'"
        ) from exc


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

def ActorCritic(
    n_features: int = N_STATE_FEATURES,
    hidden: int = HIDDEN_SIZE,
    n_layers: int = N_LAYERS,
):
    """
    Factory: shared trunk → policy head (34-dim logits) + value head (scalar).

    Returns a torch.nn.Module. Requires torch.
    """
    torch = _require_torch()
    import torch.nn as nn

    class _ResBlock(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(size, size), nn.LayerNorm(size), nn.ReLU(),
                nn.Linear(size, size), nn.LayerNorm(size),
            )
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class _ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                *[_ResBlock(hidden) for _ in range(n_layers)],
            )
            self.policy_head = nn.Linear(hidden, NTILES)   # 34 tile logits
            self.value_head = nn.Linear(hidden, 1)          # scalar baseline

        def forward(self, x):
            h = self.trunk(x)
            return self.policy_head(h), self.value_head(h)

        def policy_logits(self, x):
            return self.policy_head(self.trunk(x))

    return _ActorCritic()


def save_policy(model, path: Path, optimizer=None) -> None:
    """Save ActorCritic weights, hyperparameters, and optionally optimizer state."""
    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trunk_linear = next(m for m in model.trunk.modules()
                        if hasattr(m, "in_features"))
    ckpt = {
        "state_dict": model.state_dict(),
        "n_features": trunk_linear.in_features,
        "hidden": hidden_size(model),
        "n_layers": sum(1 for m in model.trunk.modules()
                        if type(m).__name__ == "_ResBlock"),
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    torch.save(ckpt, path)


def hidden_size(model) -> int:
    return model.policy_head.in_features


def load_policy(path: Path):
    """Load an ActorCritic from a saved checkpoint."""
    torch = _require_torch()
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ActorCritic(
        n_features=ckpt["n_features"],
        hidden=ckpt["hidden"],
        n_layers=ckpt["n_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Episode data structures
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One decision point for the learning agent during an episode."""
    state_feat: np.ndarray   # shape (N_STATE_FEATURES,)
    action: int              # tile index discarded
    old_log_prob: float      # log prob under collection policy
    value_pred: float        # value estimate at this state
    potential: float = 0.0          # state potential φ(s) for PBRS reward shaping
    discard_reward: float = 0.0     # immediate defense penalty: −DEFENSE_WEIGHT × expected_cost
    progress_reward: float = 0.0    # tiles_away-progress reward: +tiles_away_reward per tiles_away step gained


@dataclass
class Episode:
    """Trajectory collected from one simulated game."""
    steps: list[Step] = field(default_factory=list)
    terminal_reward: float = 0.0    # my_net from game result
    final_potential: float = 0.0    # φ(s_T) at terminal state


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------

def _deal_fresh_game(rng: random.Random):
    """
    Deal a complete 4-player starting position with all 148 tiles (136 standard + 12 bonus).

    Returns (my_concealed, opp_concealeds, wall) where:
      my_concealed   — int8 array (34,) for the learning agent
      opp_concealeds — list of 3 int8 arrays, one per opponent
      wall           — remaining tiles (including bonus tiles IDs 34-45)
    Bonus tiles encountered during dealing are set aside without replacement,
    mirroring the engine behaviour. Bonus tiles in the wall are handled during play.
    """
    pool = [tid for tid in range(NTILES) for _ in range(4)] + list(range(34, 46))
    rng.shuffle(pool)
    hands = []
    idx = 0
    for _ in range(4):
        arr = np.zeros(NTILES, dtype=np.int8)
        dealt = 0
        while dealt < 13 and idx < len(pool):
            tid = pool[idx]
            idx += 1
            if tid >= 34:  # bonus tile during deal — set aside, continue dealing
                continue
            arr[tid] += 1
            dealt += 1
        hands.append(arr)
    return hands[0], hands[1:], pool[idx:]


def _build_game_state(
    my_concealed: np.ndarray,
    my_seat: int,
    prevailing_wind: int,
    wall_remaining: int,
    turn: int,
    opp_discards: list[list[int]],
    opp_seats: list[int],
) -> GameState:
    """Construct a GameState snapshot for feature extraction."""
    opponents = [
        PlayerView(seat=seat, discards=list(opp_discards[i]))
        for i, seat in enumerate(opp_seats)
    ]
    return GameState(
        my_hand=HandState(
            concealed=my_concealed.copy(),
            seat_wind=my_seat,
        ),
        my_seat=my_seat,
        prevailing_wind=prevailing_wind,
        opponents=opponents,
        wall_tiles_remaining=wall_remaining,
        turn_number=turn,
    )


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episode(
    actor_critic,
    rng: random.Random,
    my_seat: int = Wind.EAST,
    prevailing_wind: int = Wind.EAST,
    shaping_scale: float = SHAPING_SCALE,
    defense_weight: float = DEFENSE_WEIGHT,
    tiles_away_reward: float = 0.0,
    waiting_bonus: float = 0.0,
    reward_scale: float = 1.0,
    oracle_coef: float = 0.0,
) -> Episode:
    """
    Play one game: learning agent (actor_critic) vs three heuristic AI opponents.

    AI opponents will claim pongs when it maintains or improves their tiles_away.
    The agent does not auto-pong — it only decides what to discard on its draw.
    Wall stops at 15 tiles remaining, matching the real game dead-wall rule.
    """
    torch = _require_torch()
    import torch.nn.functional as F

    torch.manual_seed(rng.randint(0, 2**31 - 1))
    my_concealed, opp_concealeds, wall = _deal_fresh_game(rng)

    _WINDS = [int(Wind.EAST), int(Wind.SOUTH), int(Wind.WEST), int(Wind.NORTH)]
    opp_seats = [s for s in _WINDS if s != my_seat]

    my_sim = SimHand(my_concealed.copy(), 0, my_seat)
    opp_sims = [SimHand(arr.copy(), 0, seat) for arr, seat in zip(opp_concealeds, opp_seats)]
    all_seats = sorted([my_seat] + opp_seats)
    sim_by_seat: dict[int, SimHand] = {my_seat: my_sim}
    for h in opp_sims:
        sim_by_seat[h.seat] = h

    my_hand = my_concealed.copy()
    opp_discards: list[list[int]] = [[], [], []]
    opp_seat_to_idx = {seat: i for i, seat in enumerate(opp_seats)}
    prev_tiles_away_val = tiles_away(my_hand, 0)  # tracks 13-tile tiles_away between agent turns
    waiting_reached = False  # fires waiting_bonus only once per game

    episode = Episode()
    wall_idx = 0
    wall_remaining = len(wall)
    turn = 0
    n = len(all_seats)

    def _finalize() -> Episode:
        state_snap = _build_game_state(
            my_hand, my_seat, prevailing_wind, wall_remaining, turn, opp_discards, opp_seats,
        )
        episode.final_potential = _compute_potential(
            my_hand, my_sim.n_melds, my_seat, prevailing_wind, state_snap.unknown_tiles(),
        ) * reward_scale
        return episode

    def _check_ron(discarder: int, discard: int) -> bool:
        """Discard-win check after a discard. Sets terminal_reward and returns True if game over."""
        for claimer_seat in all_seats:
            if claimer_seat == discarder:
                continue
            if sim_by_seat[claimer_seat].can_win_from(tile):
                tai = _estimate_tai(sim_by_seat[claimer_seat].concealed, sim_by_seat[claimer_seat].n_melds)
                if tai < _MIN_TAI:
                    continue
                shooter_pay, _ = chip_payment(tai)
                if discarder == my_seat:
                    episode.terminal_reward = -shooter_pay * reward_scale
                elif claimer_seat == my_seat:
                    episode.terminal_reward = shooter_pay * reward_scale
                return True
        return False

    def _run_agent_discard() -> int:
        """Run RL policy to pick a discard. Mutates my_hand, my_sim, episode. Returns tile id."""
        nonlocal prev_shanten_val, tenpai_reached

        state_snap = _build_game_state(
            my_hand, my_seat, prevailing_wind, wall_remaining, turn, opp_discards, opp_seats,
        )
        state_feat = extract_state_features(state_snap)
        feat_t = torch.tensor(state_feat, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, value_t = actor_critic(feat_t)
        logits = logits.squeeze(0)
        value = value_t.squeeze(0).item()

        valid = torch.zeros(NTILES, dtype=torch.bool)
        for t in range(NTILES):
            if my_hand[t] > 0:
                valid[t] = True
        logits[~valid] = -1e9

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        action = torch.multinomial(probs, 1).item()
        old_log_prob = log_probs[action].item()

        discard_tile = action
        my_hand[discard_tile] -= 1
        my_sim.concealed[discard_tile] -= 1

            s_after = tiles_away(my_hand, 0)
            pr = tiles_away_reward * float(max(0, prev_tiles_away_val - s_after))
            if not waiting_reached and s_after == 0:
                waiting_reached = True
                pr += waiting_bonus
            pr *= reward_scale
            prev_tiles_away_val = s_after

            pot = _compute_potential(my_hand, 0, my_seat, prevailing_wind)
            opp_models = model_all_opponents(state_snap)
            cost = expected_shooting_cost(discard, state_snap, opp_models)
            dr = -defense_weight * cost * reward_scale
            # Oracle guiding: privileged perfect-information caution signal,
            # annealed to zero over training by the caller (oracle_coef → 0).
            if oracle_coef > 0.0:
                dr += -oracle_coef * oracle_threat(opp_sims) * reward_scale
            episode.steps.append(Step(state_feat, action, old_log_prob, value, pot, dr, pr))
        else:
            discard = _heuristic_discard(h)
            h.concealed[discard] -= 1
            opp_discards[opp_seat_to_idx[seat]].append(discard)

        # Discard-win check
        if _check_ron(seat, discard):
            return _finalize()

        # Claims: kong/pong (clockwise priority) then chow (left player only).
        # Agent uses same heuristics as opponents; RL policy runs for post-claim discard.
        seat_in_order = all_seats.index(seat)
        claimed = False

        for offset in range(1, n):
            cs = all_seats[(seat_in_order + offset) % n]
            ch = sim_by_seat[cs]

            if ch.concealed[discard] >= 3 and _wants_kong_sim(ch, discard):
                ch.concealed[discard] -= 3
                ch.n_melds += 1
                if cs == my_seat:
                    my_hand[discard] -= 3
                rep = None
                while wall_idx < len(wall):
                    if wall_remaining <= 15:
                        return _finalize()
                    rtid = wall[wall_idx]
                    wall_idx += 1
                    wall_remaining -= 1
                    if rtid < 34:
                        rep = rtid
                        break
                if rep is None:
                    return _finalize()
                ch.concealed[rep] += 1
                if cs == my_seat:
                    my_hand[rep] += 1
                if ch.is_winner():
                    tai = _estimate_tai(ch.concealed, ch.n_melds)
                    if tai >= _MIN_TAI:
                        _, zimo_pay = chip_payment(tai)
                        net = zimo_pay * 3.0 if cs == my_seat else -zimo_pay
                        episode.terminal_reward = net * reward_scale
                        return _finalize()
                if cs == my_seat:
                    kong_disc = _run_agent_discard()
                else:
                    kong_disc = _heuristic_discard(ch)
                    ch.concealed[kong_disc] -= 1
                    opp_discards[opp_seat_to_idx[cs]].append(kong_disc)
                if _check_ron(cs, kong_disc):
                    return _finalize()
                seat_cursor = all_seats.index(cs) + 1
                claimed = True
                break

            elif ch.concealed[discard] >= 2 and _wants_pong_sim(ch, discard):
                ch.concealed[discard] -= 2
                ch.n_melds += 1
                if cs == my_seat:
                    my_hand[discard] -= 2
                    pong_disc = _run_agent_discard()
                else:
                    pong_disc = _heuristic_discard(ch)
                    ch.concealed[pong_disc] -= 1
                    opp_discards[opp_seat_to_idx[cs]].append(pong_disc)
                if _check_ron(cs, pong_disc):
                    return _finalize()
                seat_cursor = all_seats.index(cs) + 1
                claimed = True
                break

        if not claimed:
            left_cs = all_seats[(seat_in_order + 1) % n]
            ch = sim_by_seat[left_cs]
            chow = _pick_best_chow_sim(ch, discard)
            if chow is not None:
                for t in chow:
                    if t != discard:
                        ch.concealed[t] -= 1
                        if left_cs == my_seat:
                            my_hand[t] -= 1
                ch.n_melds += 1
                if left_cs == my_seat:
                    chow_disc = _run_agent_discard()
                else:
                    chow_disc = _heuristic_discard(ch)
                    ch.concealed[chow_disc] -= 1
                    opp_discards[opp_seat_to_idx[left_cs]].append(chow_disc)
                if _check_ron(left_cs, chow_disc):
                    return _finalize()
                seat_cursor = all_seats.index(left_cs) + 1
                claimed = True

        if not claimed:
            seat_cursor += 1

    return _finalize()


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    actor_critic,
    optimizer,
    episodes: list[Episode],
    clip_eps: float = CLIP_EPS,
    vf_coef: float = VF_COEF,
    ent_coef: float = ENT_COEF,
    ppo_epochs: int = 4,
    shaping_scale: float = SHAPING_SCALE,
    gamma: float = 1.0,
) -> dict:
    """
    One PPO update pass over a batch of episodes.

    Computes discounted returns backwards from the terminal reward.  At each
    step the shaped reward combines three signals:
      1. discard_reward: immediate defense penalty.
      2. progress_reward: per-step tiles_away-improvement bonus.
      3. PBRS: γ·φ(s_{t+1}) − φ(s_t) — telescopes to the end-of-episode
         potential gain, weighted by shaping_scale.

    With gamma=1.0 and progress_reward=0 the result is identical to the
    original formulation, so the default is fully backwards-compatible.

    Returns a dict with scalar loss metrics.
    """
    torch = _require_torch()
    import torch.nn.functional as F

    all_states, all_actions, all_old_lp, all_returns = [], [], [], []
    for ep in episodes:
        if not ep.steps:
            continue
        n = len(ep.steps)
        returns = [0.0] * n
        G = ep.terminal_reward
        for i in range(n - 1, -1, -1):
            step = ep.steps[i]
            next_pot = ep.final_potential if i == n - 1 else ep.steps[i + 1].potential
            pbrs = shaping_scale * (gamma * next_pot - step.potential)
            G = step.discard_reward + step.progress_reward + pbrs + gamma * G
            returns[i] = G
        for i, step in enumerate(ep.steps):
            all_states.append(step.state_feat)
            all_actions.append(step.action)
            all_old_lp.append(step.old_log_prob)
            all_returns.append(returns[i])

    if not all_states:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    states_t = torch.tensor(np.stack(all_states), dtype=torch.float32)
    actions_t = torch.tensor(all_actions, dtype=torch.long)
    old_lp_t = torch.tensor(all_old_lp, dtype=torch.float32)
    returns_t = torch.tensor(all_returns, dtype=torch.float32)

    total_pl = total_vl = total_ent = 0.0
    for _ in range(ppo_epochs):
        logits, values = actor_critic(states_t)
        values = values.squeeze(-1)
        log_probs_all = F.log_softmax(logits, dim=-1)
        new_lp = log_probs_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Normalise advantages (not returns) so the sign of terminal reward is preserved
        # for the value head, while the policy gradient still gets stable gradients.
        advantages = returns_t - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = (new_lp - old_lp_t).exp()
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        value_loss = F.mse_loss(values, returns_t)
        entropy = -(F.softmax(logits, dim=-1) * log_probs_all).sum(-1).mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
        optimizer.step()

        total_pl += policy_loss.item()
        total_vl += value_loss.item()
        total_ent += entropy.item()

    return {
        "policy_loss": total_pl / ppo_epochs,
        "value_loss": total_vl / ppo_epochs,
        "entropy": total_ent / ppo_epochs,
    }


# ---------------------------------------------------------------------------
# Tournament evaluation
# ---------------------------------------------------------------------------

def evaluate_vs_heuristic(
    actor_critic,
    n_games: int = 100,
    seed: int = 0,
    my_seat: int = Wind.EAST,
    prevailing_wind: int = Wind.EAST,
) -> dict:
    """
    Play n_games of learning agent vs three heuristic opponents.

    Returns {win_rate, shoot_rate, draw_rate, mean_net}.
    """
    rng = random.Random(seed)
    _WINDS = [int(Wind.EAST), int(Wind.SOUTH), int(Wind.WEST), int(Wind.NORTH)]
    win = shoot = draw = 0
    total_net = 0.0

    for i in range(n_games):
        ep_seat = _WINDS[i % 4]  # rotate evenly across all seats
        ep = collect_episode(actor_critic, rng, ep_seat, prevailing_wind)
        r = ep.terminal_reward
        total_net += r
        if r > 0:
            win += 1
        elif r < 0:
            shoot += 1
        else:
            draw += 1

    return {
        "win_rate": win / n_games,
        "shoot_rate": shoot / n_games,
        "draw_rate": draw / n_games,
        "mean_net": total_net / n_games,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_self_play(
    n_episodes: int = 5000,
    model_path: Path = Path("models/policy.pt"),
    episodes_per_update: int = 32,
    lr: float = 3e-4,
    seed: int = 0,
    verbose: bool = True,
    eval_every: int = 500,
    eval_games: int = 200,
    shaping_scale: float = SHAPING_SCALE,
    defense_weight: float = DEFENSE_WEIGHT,
    resume: bool = False,
    gamma: float = 1.0,
    tiles_away_reward: float = 0.0,
    waiting_bonus: float = 0.0,
    reward_scale: float = 1.0,
    ent_coef: float = ENT_COEF,
    oracle_coef: float = 0.0,
) -> dict:
    """
    Train ActorCritic via self-play against heuristic opponents.

    Uses potential-based reward shaping (PBRS) derived from tiles_away
    proximity and tai-potential to give the agent dense learning signal.

    oracle_coef > 0 enables Suphx-style oracle guiding: a privileged
    perfect-information caution signal that is linearly annealed from oracle_coef
    to 0 across training, so the final policy depends only on observable state.

    Saves the best checkpoint (by mean_net in evaluation) to model_path.
    Pass resume=True to continue from an existing checkpoint at model_path.
    Raises ImportError if torch is not installed.
    """
    torch = _require_torch()

    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if resume and Path(model_path).exists():
        ckpt = torch.load(Path(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if verbose:
            print(f"Resumed from {model_path}")

    best_mean_net = float("-inf")
    last_stats: dict = {}
    episode_buffer: list[Episode] = []

    if verbose:
        print(
            f"Training for {n_episodes} episodes, "
            f"batch={episodes_per_update}, shaping={shaping_scale}, "
            f"defense={defense_weight}, gamma={gamma}, "
            f"tiles_away_reward={tiles_away_reward}, waiting_bonus={waiting_bonus}, "
            f"reward_scale={reward_scale}, ent_coef={ent_coef}"
        )

    _WINDS = [int(Wind.EAST), int(Wind.SOUTH), int(Wind.WEST), int(Wind.NORTH)]

    for ep_num in range(1, n_episodes + 1):
        ep_seat = rng.choice(_WINDS)  # train from all seats equally
        # Anneal oracle guiding linearly from oracle_coef → 0 over training.
        cur_oracle = oracle_coef * max(0.0, 1.0 - (ep_num - 1) / max(n_episodes - 1, 1))
        episode_buffer.append(
            collect_episode(model, rng, my_seat=ep_seat, shaping_scale=shaping_scale,
                            defense_weight=defense_weight, tiles_away_reward=tiles_away_reward,
                            waiting_bonus=waiting_bonus, reward_scale=reward_scale,
                            oracle_coef=cur_oracle)
        )

        if len(episode_buffer) >= episodes_per_update:
            metrics = ppo_update(
                model, optimizer, episode_buffer, shaping_scale=shaping_scale,
                gamma=gamma, ent_coef=ent_coef,
            )
            episode_buffer.clear()

            if verbose and ep_num % 200 == 0:
                print(
                    f"Ep {ep_num:5d}  "
                    f"pl={metrics['policy_loss']:+.4f}  "
                    f"vl={metrics['value_loss']:.4f}  "
                    f"ent={metrics['entropy']:.4f}"
                )

        if ep_num % eval_every == 0:
            model.eval()
            stats = evaluate_vs_heuristic(model, n_games=eval_games, seed=ep_num)
            model.train()
            last_stats = stats
            if verbose:
                print(
                    f"  eval @ {ep_num:5d}: "
                    f"win={stats['win_rate']:.3f}  "
                    f"shoot={stats['shoot_rate']:.3f}  "
                    f"net={stats['mean_net']:+.3f}"
                )
            if stats["mean_net"] > best_mean_net:
                best_mean_net = stats["mean_net"]
                save_policy(model, model_path, optimizer)
                if verbose:
                    print(f"  -> new best ({best_mean_net:+.3f}), saved to {model_path}")

    if verbose:
        print(f"Done. Best mean_net: {best_mean_net:+.3f}")
    return {
        "best_mean_net": best_mean_net,
        "final_win_rate": last_stats.get("win_rate", 0.0),
        "final_shoot_rate": last_stats.get("shoot_rate", 0.0),
        "final_draw_rate": last_stats.get("draw_rate", 0.0),
    }


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Self-play RL training.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=Path("models/policy.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--shaping-scale", type=float, default=SHAPING_SCALE,
                        help="PBRS reward shaping weight (0 = off)")
    parser.add_argument("--defense-weight", type=float, default=DEFENSE_WEIGHT,
                        help="Defense penalty weight per step (0 = off)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor for returns (1.0 = no discounting)")
    parser.add_argument("--tiles-away-reward", type=float, default=0.0,
                        help="Per-step reward for each tiles_away improvement")
    parser.add_argument("--waiting-bonus", type=float, default=0.0,
                        help="One-time reward the first time the agent reaches waiting in a game")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Scale all rewards by this factor (e.g. 1/48 normalises to [-1,+1])")
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF,
                        help="Entropy coefficient in PPO loss")
    parser.add_argument("--oracle-coef", type=float, default=0.0,
                        help="Oracle-guiding strength, annealed to 0 over training (0 = off)")
    parser.add_argument("--resume", action="store_true",
                        help="Continue training from existing checkpoint at --out path")
    args = parser.parse_args()
    train_self_play(
        n_episodes=args.episodes,
        model_path=args.out,
        episodes_per_update=args.batch,
        lr=args.lr,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        shaping_scale=args.shaping_scale,
        defense_weight=args.defense_weight,
        gamma=args.gamma,
        tiles_away_reward=args.tiles_away_reward,
        waiting_bonus=args.waiting_bonus,
        reward_scale=args.reward_scale,
        ent_coef=args.ent_coef,
        oracle_coef=args.oracle_coef,
        resume=args.resume,
    )


if __name__ == "__main__":
    _cli()
