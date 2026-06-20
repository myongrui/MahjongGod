"""
Tests for the self-play RL module.

The in-engine collection harness is net-agnostic (the policy is injected as a
plain feat→(logits, value) callable), so its logic is tested with a NumPy fake
net and needs no torch. Tests that exercise the actual ActorCritic / PPO are
guarded with pytest.importorskip("torch").
"""

import random

import numpy as np
import pytest

from cracked.tiles import Wind
from cracked.training.features import N_STATE_FEATURES, extract_state_features
from cracked.tiles import tiles_from_names
from cracked.hand import HandState
from cracked.game_state import GameState, PlayerView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state() -> GameState:
    all_winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
    return GameState(
        my_hand=HandState(seat_wind=Wind.EAST),
        my_seat=Wind.EAST,
        prevailing_wind=Wind.EAST,
        opponents=[PlayerView(seat=w) for w in all_winds if w != Wind.EAST],
    )


def _full_state() -> GameState:
    state = _make_state()
    state.my_hand.concealed = tiles_from_names(
        ["b1","b2","b3","c1","c2","c3","d1","d2","d3","ew","ew","ew","rd","gd"]
    )
    return state


def _fake_net(seed: int = 0):
    """A NumPy stand-in for the policy network: feat → (34 logits, value)."""
    r = np.random.RandomState(seed)

    def fn(feat):
        assert feat.shape == (N_STATE_FEATURES,)
        return r.randn(34).astype(np.float32), float(r.randn())

    return fn


# ---------------------------------------------------------------------------
# extract_state_features (no torch needed)
# ---------------------------------------------------------------------------

def test_state_feature_length():
    assert extract_state_features(_full_state()).shape == (N_STATE_FEATURES,)


def test_state_feature_dtype():
    assert extract_state_features(_full_state()).dtype == np.float32


def test_state_feature_bounded():
    feat = extract_state_features(_full_state())
    assert feat.min() >= -1e-6
    assert feat.max() <= 1.0 + 1e-6


def test_state_features_differ_by_turn():
    s1 = _full_state()
    s2 = _full_state()
    s2.turn_number = 20
    assert not np.array_equal(extract_state_features(s1), extract_state_features(s2))


def test_n_state_features_constant():
    from cracked.training.features import _STATE_BLOCK_SIZE, _OPP_BLOCK_SIZE
    assert N_STATE_FEATURES == _STATE_BLOCK_SIZE + 3 * _OPP_BLOCK_SIZE


# ---------------------------------------------------------------------------
# In-engine collection harness (net-agnostic, no torch)
# ---------------------------------------------------------------------------

def test_collect_match_episode_returns_episode():
    from cracked.training.self_play import collect_match_episode, Episode
    ep = collect_match_episode(_fake_net(0), random.Random(1), n_rounds=1)
    assert isinstance(ep, Episode)
    assert len(ep.steps) > 0
    assert np.isfinite(ep.terminal_reward)


def test_collect_match_episode_steps_valid():
    from cracked.training.self_play import collect_match_episode
    ep = collect_match_episode(_fake_net(2), random.Random(3), n_rounds=1)
    for step in ep.steps:
        assert 0 <= step.action < 34
        assert step.state_feat.shape == (N_STATE_FEATURES,)
        assert np.isfinite(step.potential)
        assert np.isfinite(step.discard_reward)


def test_collect_match_episode_marks_hand_ends():
    from cracked.training.self_play import collect_match_episode
    ep = collect_match_episode(_fake_net(4), random.Random(5), n_rounds=1)
    # A match spans several hands; each hand the agent acts in ends with a marker.
    assert sum(s.is_hand_end for s in ep.steps) >= 1


def test_collect_match_episode_reproducible():
    from cracked.training.self_play import collect_match_episode
    ep1 = collect_match_episode(_fake_net(7), random.Random(9), n_rounds=1)
    ep2 = collect_match_episode(_fake_net(7), random.Random(9), n_rounds=1)
    assert len(ep1.steps) == len(ep2.steps)
    assert ep1.terminal_reward == ep2.terminal_reward


def test_collect_match_episode_reward_scale():
    from cracked.training.self_play import collect_match_episode
    unit = collect_match_episode(_fake_net(1), random.Random(2), n_rounds=1, reward_scale=1.0)
    half = collect_match_episode(_fake_net(1), random.Random(2), n_rounds=1, reward_scale=0.5)
    assert half.terminal_reward == pytest.approx(unit.terminal_reward * 0.5)


def test_play_eval_match_outcomes_bounded():
    from cracked.training.self_play import play_eval_match
    r = play_eval_match(_fake_net(3), random.Random(8), n_rounds=1, agent_wind=int(Wind.EAST))
    assert r["hands"] > 0
    assert r["won"] + r["shot"] + r["drew"] <= r["hands"]
    assert np.isfinite(r["net"])


def test_recording_policy_reset_hand():
    from cracked.training.self_play import RecordingPolicy
    p = RecordingPolicy(_fake_net(0), random.Random(0))
    p._prev_s = 3
    p._waiting_reached = True
    p.reset_hand()
    assert p._prev_s is None and p._waiting_reached is False


# ---------------------------------------------------------------------------
# ActorCritic (requires torch)
# ---------------------------------------------------------------------------

def test_actor_critic_output_shapes():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, N_STATE_FEATURES
    model = ActorCritic()
    model.eval()
    with torch.no_grad():
        logits, value = model(torch.zeros(4, N_STATE_FEATURES))
    assert logits.shape == (4, 34)
    assert value.shape == (4, 1)


def test_actor_critic_logits_and_value_finite():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, N_STATE_FEATURES
    model = ActorCritic()
    with torch.no_grad():
        logits, value = model(torch.randn(1, N_STATE_FEATURES))
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_actor_critic_save_load(tmp_path):
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, save_policy, load_policy
    model = ActorCritic()
    path = tmp_path / "policy.pt"
    save_policy(model, path)
    loaded = load_policy(path)
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"


def test_torch_policy_fn_shapes():
    pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, _torch_policy_fn
    fn = _torch_policy_fn(ActorCritic())
    logits, value = fn(np.zeros(N_STATE_FEATURES, dtype=np.float32))
    assert logits.shape == (34,)
    assert isinstance(value, float)


# ---------------------------------------------------------------------------
# collect / ppo / evaluate with the real network (requires torch)
# ---------------------------------------------------------------------------

def test_collect_match_episode_with_torch_net():
    pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, _torch_policy_fn, collect_match_episode
    ep = collect_match_episode(_torch_policy_fn(ActorCritic()), random.Random(0), n_rounds=1)
    assert len(ep.steps) > 0
    assert all(0 <= s.action < 34 for s in ep.steps)


def test_ppo_update_returns_metrics():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import (
        ActorCritic, _torch_policy_fn, collect_match_episode, ppo_update,
    )
    model = ActorCritic()
    fn = _torch_policy_fn(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    episodes = [collect_match_episode(fn, random.Random(i), n_rounds=1) for i in range(2)]
    metrics = ppo_update(model, optimizer, episodes)
    assert "policy_loss" in metrics and "value_loss" in metrics and "entropy" in metrics


def test_ppo_update_empty_episodes_no_crash():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, Episode, ppo_update
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = ppo_update(model, optimizer, [Episode()])
    assert metrics["policy_loss"] == 0.0


def test_evaluate_rates_bounded():
    pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, evaluate_vs_heuristic
    stats = evaluate_vs_heuristic(ActorCritic(), n_games=2, seed=0, n_rounds=1)
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert 0.0 <= stats["shoot_rate"] <= 1.0
    assert 0.0 <= stats["draw_rate"] <= 1.0
    assert np.isfinite(stats["mean_net"])


def test_evaluate_reproducible():
    pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, evaluate_vs_heuristic
    model = ActorCritic()
    s1 = evaluate_vs_heuristic(model, n_games=2, seed=7, n_rounds=1)
    s2 = evaluate_vs_heuristic(model, n_games=2, seed=7, n_rounds=1)
    assert s1["mean_net"] == s2["mean_net"]
    assert s1["win_rate"] == s2["win_rate"]
