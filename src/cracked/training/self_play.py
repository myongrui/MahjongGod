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
    SimHand, _heuristic_discard, _estimate_tai, _payment,
)
from cracked.training.features import N_STATE_FEATURES, extract_state_features

_WIND_ORDER = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
_OPP_SEATS = [Wind.SOUTH, Wind.WEST, Wind.NORTH]

HIDDEN_SIZE = 256
N_LAYERS = 2
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_ROUNDS = 40  # game turn cap (same as simulator)


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


def save_policy(model, path: Path) -> None:
    """Save ActorCritic weights and hyperparameters."""
    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trunk_linear = next(m for m in model.trunk.modules()
                        if hasattr(m, "in_features"))
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": trunk_linear.in_features,
        "hidden": hidden_size(model),
        "n_layers": sum(1 for m in model.trunk.modules()
                        if type(m).__name__ == "_ResBlock"),
    }, path)


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


@dataclass
class Episode:
    """Trajectory collected from one simulated game."""
    steps: list[Step] = field(default_factory=list)
    terminal_reward: float = 0.0   # my_net from game result


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------

def _deal_fresh_game(rng: random.Random):
    """
    Deal a complete 4-player starting position.

    Returns (my_concealed, opp_concealeds, wall) where:
      my_concealed   — int8 array (34,) for the learning agent (East)
      opp_concealeds — list of 3 int8 arrays, one per opponent
      wall           — list of tile IDs (84 tiles, drawn in order)
    """
    pool = [tid for tid in range(NTILES) for _ in range(4)]
    rng.shuffle(pool)
    hands = []
    for i in range(4):
        arr = np.zeros(NTILES, dtype=np.int8)
        for t in pool[i * 13: (i + 1) * 13]:
            arr[t] += 1
        hands.append(arr)
    return hands[0], hands[1:], pool[52:]


def _build_game_state(
    my_concealed: np.ndarray,
    my_seat: int,
    prevailing_wind: int,
    wall_remaining: int,
    turn: int,
    opp_discards: list[list[int]],
) -> GameState:
    """Construct a GameState snapshot for feature extraction."""
    opponents = [
        PlayerView(seat=seat, discards=list(opp_discards[i]))
        for i, seat in enumerate(_OPP_SEATS)
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
) -> Episode:
    """
    Play one game: learning agent (actor_critic) vs three heuristic opponents.

    The learning agent plays as my_seat (default East).
    Returns an Episode with one Step per discard decision made by our agent.
    """
    torch = _require_torch()
    import torch.nn.functional as F

    # Derive torch seed from rng so all randomness is controlled by one seed.
    torch.manual_seed(rng.randint(0, 2**31 - 1))
    my_concealed, opp_concealeds, wall = _deal_fresh_game(rng)

    # SimHand objects hold the ground-truth tile counts (for win checking)
    my_sim = SimHand(my_concealed.copy(), 0, my_seat)
    opp_sims = [
        SimHand(arr.copy(), 0, seat)
        for arr, seat in zip(opp_concealeds, _OPP_SEATS)
    ]
    all_seats = sorted([my_seat] + list(_OPP_SEATS))
    sim_by_seat = {my_seat: my_sim}
    for h in opp_sims:
        sim_by_seat[h.seat] = h

    # Tracked from our perspective (for feature extraction)
    my_hand = my_concealed.copy()
    opp_discards: list[list[int]] = [[], [], []]   # indexed by opp order in _OPP_SEATS
    opp_seat_to_idx = {seat: i for i, seat in enumerate(_OPP_SEATS)}

    episode = Episode()
    wall_idx = 0
    wall_remaining = len(wall)
    turn = 0

    for _ in range(MAX_ROUNDS):
        for seat in all_seats:
            if wall_idx >= len(wall):
                return episode  # wall exhausted — draw

            drawn = wall[wall_idx]
            wall_idx += 1
            wall_remaining -= 1
            turn += 1
            sim_by_seat[seat].concealed[drawn] += 1
            if seat == my_seat:
                my_hand[drawn] += 1

            # Self-draw win check
            h = sim_by_seat[seat]
            if h.is_winner():
                tai = _estimate_tai(h.concealed, h.n_melds)
                pay = _payment(tai)
                if seat == my_seat:
                    episode.terminal_reward = pay * 3.0
                else:
                    episode.terminal_reward = -pay
                return episode

            # Choose discard
            if seat == my_seat:
                state_snap = _build_game_state(
                    my_hand, my_seat, prevailing_wind,
                    wall_remaining, turn, opp_discards,
                )
                state_feat = extract_state_features(state_snap)
                feat_t = torch.tensor(state_feat, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    logits, value_t = actor_critic(feat_t)

                logits = logits.squeeze(0)
                value = value_t.squeeze(0).item()

                # Mask tiles not in hand
                valid = torch.zeros(NTILES, dtype=torch.bool)
                for t in range(NTILES):
                    if my_hand[t] > 0:
                        valid[t] = True
                logits[~valid] = -1e9

                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                action = torch.multinomial(probs, 1).item()
                old_log_prob = log_probs[action].item()

                episode.steps.append(Step(state_feat, action, old_log_prob, value))
                discard = action
                my_hand[discard] -= 1
            else:
                discard = _heuristic_discard(h)

            h.concealed[discard] -= 1
            if seat != my_seat:
                opp_discards[opp_seat_to_idx[seat]].append(discard)

            # Ron win check (any other player claims the discard)
            for claimer_seat in all_seats:
                if claimer_seat == seat:
                    continue
                if sim_by_seat[claimer_seat].can_win_from(discard):
                    tai = _estimate_tai(
                        sim_by_seat[claimer_seat].concealed,
                        sim_by_seat[claimer_seat].n_melds,
                    )
                    pay = _payment(tai)
                    if seat == my_seat:
                        episode.terminal_reward = -pay * 3.0
                    elif claimer_seat == my_seat:
                        episode.terminal_reward = pay * 3.0
                    return episode

    return episode  # max rounds reached — draw


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
) -> dict:
    """
    One PPO update pass over a batch of episodes.

    Returns a dict with scalar loss metrics.
    """
    torch = _require_torch()
    import torch.nn.functional as F

    # Flatten all steps across episodes; each step gets the same terminal reward
    all_states, all_actions, all_old_lp, all_returns = [], [], [], []
    for ep in episodes:
        if not ep.steps:
            continue
        G = ep.terminal_reward
        for step in ep.steps:
            all_states.append(step.state_feat)
            all_actions.append(step.action)
            all_old_lp.append(step.old_log_prob)
            all_returns.append(G)

    if not all_states:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    states_t = torch.tensor(np.stack(all_states), dtype=torch.float32)
    actions_t = torch.tensor(all_actions, dtype=torch.long)
    old_lp_t = torch.tensor(all_old_lp, dtype=torch.float32)
    returns_t = torch.tensor(all_returns, dtype=torch.float32)

    # Normalise returns for more stable gradients
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    total_pl = total_vl = total_ent = 0.0
    for _ in range(ppo_epochs):
        logits, values = actor_critic(states_t)
        values = values.squeeze(-1)
        log_probs_all = F.log_softmax(logits, dim=-1)
        new_lp = log_probs_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        advantages = (returns_t - values.detach())
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
    win = shoot = draw = 0
    total_net = 0.0

    for _ in range(n_games):
        ep = collect_episode(actor_critic, rng, my_seat, prevailing_wind)
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
) -> None:
    """
    Train ActorCritic via self-play against heuristic opponents.

    Saves the best checkpoint (by mean_net in evaluation) to model_path.
    Raises ImportError if torch is not installed.
    """
    torch = _require_torch()

    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_mean_net = float("-inf")
    episode_buffer: list[Episode] = []

    if verbose:
        print(f"Training for {n_episodes} episodes, batch={episodes_per_update}")

    for ep_num in range(1, n_episodes + 1):
        episode_buffer.append(collect_episode(model, rng))

        if len(episode_buffer) >= episodes_per_update:
            metrics = ppo_update(model, optimizer, episode_buffer)
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
            if verbose:
                print(
                    f"  eval @ {ep_num:5d}: "
                    f"win={stats['win_rate']:.3f}  "
                    f"shoot={stats['shoot_rate']:.3f}  "
                    f"net={stats['mean_net']:+.3f}"
                )
            if stats["mean_net"] > best_mean_net:
                best_mean_net = stats["mean_net"]
                save_policy(model, model_path)
                if verbose:
                    print(f"  → new best ({best_mean_net:+.3f}), saved to {model_path}")

    if verbose:
        print(f"Done. Best mean_net: {best_mean_net:+.3f}")


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
    args = parser.parse_args()
    train_self_play(
        n_episodes=args.episodes,
        model_path=args.out,
        episodes_per_update=args.batch,
        lr=args.lr,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
    )


if __name__ == "__main__":
    _cli()
