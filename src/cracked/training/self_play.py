"""
Self-play RL training for Singapore Mahjong.

Trains an ActorCritic policy network with PPO. Each episode is a full match
played inside the real game engine: the learning agent occupies one rotating
seat and the other three are the fixed-weight HeuristicPolicy. The reward is
the agent's true net chip change over the match (real scoring), so the policy
optimizes chip count across a whole game.

Usage:
    python -m cracked.training.self_play --episodes 5000 --out models/policy.pt

Requires PyTorch: pip install 'cracked[ml]'
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from cracked.tiles import NTILES, Wind
from cracked.game_state import GameState
from cracked.scoring import STARTING_CHIPS
from cracked.optimizer import hand_tai_potential
from cracked.tiles_away import tiles_away, acceptance_count
from cracked.danger import expected_shooting_cost
from cracked.opponent_model import model_all_opponents
from cracked.policy import HeuristicPolicy
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

    Combines tiles_away proximity, tai ceiling, and tile acceptance count
    so the agent receives a dense learning signal at every step.
    """
    s = tiles_away(concealed, n_melds)
    tiles_away_val = (7 - s) / 8.0
    tai_pot = hand_tai_potential(concealed, [], seat_wind, prevailing_wind)
    tai_val = min(tai_pot / 6.0, 1.0)

    acc_val = 0.0
    if unknown_tiles is not None and s >= 0:
        acc = acceptance_count(concealed, unknown_tiles, n_melds)
        acc_val = min(sum(acc.values()) / 30.0, 1.0)

    return 1.2 * tiles_away_val + 0.8 * tai_val + 0.5 * acc_val


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
    is_hand_end: bool = False       # last agent step of a hand — PBRS does not telescope across the hand boundary


@dataclass
class Episode:
    """Trajectory collected from one game/match."""
    steps: list[Step] = field(default_factory=list)
    terminal_reward: float = 0.0    # net chip change over the match
    final_potential: float = 0.0    # φ(s_T) at terminal state


# ---------------------------------------------------------------------------
# In-engine collection: the agent plays a full match inside the real engine
# ---------------------------------------------------------------------------

def _torch_policy_fn(actor_critic):
    """Adapt a torch ActorCritic into a feat→(logits, value) numpy callable."""
    torch = _require_torch()

    def fn(feat: np.ndarray):
        with torch.no_grad():
            logits, value = actor_critic(
                torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            )
        return logits.squeeze(0).numpy(), float(value.squeeze(0))

    return fn


class RecordingPolicy:
    """Learning agent's seat policy.

    Samples discards from an injected ``policy_fn`` (feat → logits, value) and
    records each decision for PPO. The fn is plain NumPy in/out so collection is
    independent of torch (the torch network is wrapped by ``_torch_policy_fn``).
    Claims use a fixed heuristic in this phase (claim-learning is a later step).
    """

    def __init__(
        self,
        policy_fn,
        rng: random.Random,
        *,
        defense_weight: float = DEFENSE_WEIGHT,
        tiles_away_reward: float = 0.0,
        waiting_bonus: float = 0.0,
        reward_scale: float = 1.0,
        greedy: bool = False,
        claim_policy=None,
    ):
        self.policy_fn = policy_fn
        self.rng = rng
        self.defense_weight = defense_weight
        self.tiles_away_reward = tiles_away_reward
        self.waiting_bonus = waiting_bonus
        self.reward_scale = reward_scale
        self.greedy = greedy
        self.claim_policy = claim_policy or HeuristicPolicy()
        self.steps: list[Step] = []
        self._prev_s: Optional[int] = None
        self._waiting_reached = False

    def reset_hand(self) -> None:
        """Reset per-hand shaping state at the start of each new hand."""
        self._prev_s = None
        self._waiting_reached = False

    def choose_discard(self, view: GameState) -> int:
        concealed = view.my_hand.concealed
        n_melds = len(view.my_hand.melds)

        feat = extract_state_features(view)
        logits, value = self.policy_fn(feat)
        logits = np.asarray(logits, dtype=np.float64)

        masked = np.where(concealed > 0, logits, -np.inf)
        masked -= masked.max()
        probs = np.exp(masked)
        probs /= probs.sum()
        if self.greedy:
            action = int(np.argmax(probs))
        else:
            action = int(self.rng.choices(range(NTILES), weights=probs.tolist())[0])
        log_prob = float(np.log(probs[action] + 1e-12))

        # Per-step shaping signals (mirror the previous per-hand formulation).
        post = concealed.copy()
        post[action] -= 1
        s_after = tiles_away(post, n_melds)
        base_prev = self._prev_s if self._prev_s is not None else s_after
        progress = self.tiles_away_reward * float(max(0, base_prev - s_after))
        if not self._waiting_reached and s_after == 0:
            self._waiting_reached = True
            progress += self.waiting_bonus
        progress *= self.reward_scale
        self._prev_s = s_after

        potential = _compute_potential(
            post, n_melds, view.my_seat, view.prevailing_wind, view.unknown_tiles()
        ) * self.reward_scale
        models = model_all_opponents(view)
        cost = expected_shooting_cost(action, view, models)
        discard_reward = -self.defense_weight * cost * self.reward_scale

        self.steps.append(Step(
            state_feat=feat,
            action=action,
            old_log_prob=log_prob,
            value_pred=float(value),
            potential=potential,
            discard_reward=discard_reward,
            progress_reward=progress,
        ))
        return action

    def wants_pong(self, view: GameState, tile: int) -> bool:
        return self.claim_policy.wants_pong(view, tile)

    def wants_kong(self, view: GameState, tile: int) -> bool:
        return self.claim_policy.wants_kong(view, tile)

    def choose_chow(self, view: GameState, tile: int, options):
        return self.claim_policy.choose_chow(view, tile, options)


def collect_match_episode(
    policy_fn,
    rng: random.Random,
    *,
    n_rounds: int = 4,
    agent_wind: int = Wind.EAST,
    defense_weight: float = DEFENSE_WEIGHT,
    tiles_away_reward: float = 0.0,
    waiting_bonus: float = 0.0,
    reward_scale: float = 1.0,
) -> Episode:
    """Play one full match inside the real engine and return the agent's Episode.

    The agent occupies one seat (``RecordingPolicy``); the other three are the
    fixed-weight ``HeuristicPolicy``. Real scoring and chip rotation come from
    GameMatch, so the terminal reward is the agent's true net chip change over
    the whole match. PBRS shaping is kept within each hand (steps carry an
    is_hand_end marker so it does not telescope across hand boundaries).
    """
    from cracked.match import GameMatch

    agent = RecordingPolicy(
        policy_fn, rng,
        defense_weight=defense_weight,
        tiles_away_reward=tiles_away_reward,
        waiting_bonus=waiting_bonus,
        reward_scale=reward_scale,
    )
    match = GameMatch(
        n_rounds=n_rounds,
        human_initial_wind=int(agent_wind),
        agent_policy=agent,
        seed=rng.randint(0, 2**31 - 1),
    )

    hand_guard = 0
    max_hands = 4 * n_rounds * 8  # generous cap (rotations + extra East hands)
    while not match.is_complete and hand_guard < max_hands:
        match.start_hand()
        agent.reset_hand()
        n_before = len(agent.steps)
        engine = match.engine
        step_guard = 0
        while not engine.is_finished and step_guard < 4000:
            engine.step()
            step_guard += 1
        match.finish_hand()
        if len(agent.steps) > n_before:
            agent.steps[-1].is_hand_end = True  # close PBRS telescoping for this hand
        hand_guard += 1

    ep = Episode()
    ep.steps = agent.steps
    ep.terminal_reward = (match.agent_chips - STARTING_CHIPS) * reward_scale
    ep.final_potential = 0.0  # match is over; last step is a hand boundary
    return ep




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
            if step.is_hand_end:
                # PBRS stays within a hand; the hand-structure potential is
                # meaningless across the reset to a fresh hand.
                pbrs = 0.0
            else:
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

def play_eval_match(policy_fn, rng: random.Random, *, n_rounds: int, agent_wind: int) -> dict:
    """Play one full match with a greedy agent; return per-hand outcome counts.

    Returns {net, won, shot, drew, hands} where net is the agent's chip change
    over the match and won/shot/drew count the agent's per-hand outcomes
    (won = it completed the hand, shot = it discarded another player's winning
    tile, drew = wall-exhausted hand).
    """
    from cracked.match import GameMatch
    from cracked.engine import EventType

    agent = RecordingPolicy(policy_fn, rng, greedy=True)
    match = GameMatch(
        n_rounds=n_rounds, human_initial_wind=int(agent_wind),
        agent_policy=agent, seed=rng.randint(0, 2**31 - 1),
    )
    won = shot = drew = hands = 0
    hand_guard = 0
    max_hands = 4 * n_rounds * 8
    while not match.is_complete and hand_guard < max_hands:
        match.start_hand()
        agent.reset_hand()
        aw = match.agent_wind
        engine = match.engine
        outcome = None
        step_guard = 0
        while not engine.is_finished and step_guard < 4000:
            for ev in engine.step():
                if ev.type == EventType.WIN_SELF_DRAW and ev.seat == aw:
                    outcome = "won"
                elif ev.type == EventType.WIN_DISCARD:
                    if ev.seat == aw:
                        outcome = "won"
                    elif ev.detail.get("shooter") == aw:
                        outcome = "shot"
                elif ev.type == EventType.WALL_EXHAUSTED:
                    outcome = "drew"
            step_guard += 1
        match.finish_hand()
        hands += 1
        if outcome == "won":
            won += 1
        elif outcome == "shot":
            shot += 1
        elif outcome == "drew":
            drew += 1
        hand_guard += 1

    return {"net": match.agent_chips - STARTING_CHIPS,
            "won": won, "shot": shot, "drew": drew, "hands": hands}


def evaluate_vs_heuristic(
    actor_critic,
    n_games: int = 20,
    seed: int = 0,
    my_seat: int = Wind.EAST,
    prevailing_wind: int = Wind.EAST,
    n_rounds: int = 4,
) -> dict:
    """
    Play n_games full matches of the greedy agent vs three heuristic opponents.

    mean_net is the average chip change per match (the training objective);
    win/shoot/draw rates are per-hand diagnostics. The agent rotates across all
    starting seats. Requires torch (the policy network).
    """
    policy_fn = _torch_policy_fn(actor_critic)
    rng = random.Random(seed)
    _WINDS = [int(Wind.EAST), int(Wind.SOUTH), int(Wind.WEST), int(Wind.NORTH)]
    total_net = 0.0
    won = shot = drew = hands = 0

    for i in range(n_games):
        r = play_eval_match(policy_fn, rng, n_rounds=n_rounds, agent_wind=_WINDS[i % 4])
        total_net += r["net"]
        won += r["won"]; shot += r["shot"]; drew += r["drew"]; hands += max(r["hands"], 0)

    denom = max(hands, 1)
    return {
        "win_rate": won / denom,
        "shoot_rate": shot / denom,
        "draw_rate": drew / denom,
        "mean_net": total_net / max(n_games, 1),
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
    eval_games: int = 20,
    shaping_scale: float = SHAPING_SCALE,
    defense_weight: float = DEFENSE_WEIGHT,
    resume: bool = False,
    gamma: float = 1.0,
    tiles_away_reward: float = 0.0,
    waiting_bonus: float = 0.0,
    reward_scale: float = 1.0,
    ent_coef: float = ENT_COEF,
    n_rounds: int = 4,
) -> dict:
    """
    Train ActorCritic by playing full matches inside the real game engine
    against three fixed-weight heuristic opponents (HeuristicPolicy).

    Each episode is one complete GameMatch; the reward is the agent's true net
    chip change over the match (real scoring), so the policy optimizes chip
    count over a whole game. Potential-based reward shaping (PBRS, kept within
    each hand) plus optional defense/progress terms give a dense signal.

    Saves the best checkpoint (by mean_net in evaluation) to model_path.
    Pass resume=True to continue from an existing checkpoint at model_path.
    Raises ImportError if torch is not installed.
    """
    torch = _require_torch()

    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    policy_fn = _torch_policy_fn(model)  # closes over model; reflects live weights

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
        ep_seat = rng.choice(_WINDS)  # agent starts from all seats equally
        episode_buffer.append(
            collect_match_episode(
                policy_fn, rng, n_rounds=n_rounds, agent_wind=ep_seat,
                defense_weight=defense_weight, tiles_away_reward=tiles_away_reward,
                waiting_bonus=waiting_bonus, reward_scale=reward_scale,
            )
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
            stats = evaluate_vs_heuristic(model, n_games=eval_games, seed=ep_num, n_rounds=n_rounds)
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
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of full matches to train on")
    parser.add_argument("--batch", type=int, default=32,
                        help="Matches collected per PPO update")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=Path("models/policy.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Matches per evaluation")
    parser.add_argument("--n-rounds", type=int, default=4,
                        help="Table-wind rounds per match (4 = full E/S/W/N game)")
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
        n_rounds=args.n_rounds,
        resume=args.resume,
    )


if __name__ == "__main__":
    _cli()
