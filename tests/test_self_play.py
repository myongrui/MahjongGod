"""
Tests for the self-play RL module.

All tests that need PyTorch are guarded with pytest.importorskip("torch")
so they skip gracefully when the ml extras are not installed.
"""

import random

import numpy as np
import pytest

from cracked.tiles import Wind, tile_id
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


# ---------------------------------------------------------------------------
# extract_state_features (no torch needed)
# ---------------------------------------------------------------------------

def test_state_feature_length():
    state = _full_state()
    feat = extract_state_features(state)
    assert feat.shape == (N_STATE_FEATURES,)


def test_state_feature_dtype():
    state = _full_state()
    feat = extract_state_features(state)
    assert feat.dtype == np.float32


def test_state_feature_bounded():
    state = _full_state()
    feat = extract_state_features(state)
    assert feat.min() >= -1e-6
    assert feat.max() <= 1.0 + 1e-6


def test_state_features_differ_by_turn():
    s1 = _full_state()
    s2 = _full_state()
    s2.turn_number = 20
    f1 = extract_state_features(s1)
    f2 = extract_state_features(s2)
    assert not np.array_equal(f1, f2)


def test_n_state_features_constant():
    from cracked.training.features import _STATE_BLOCK_SIZE, _OPP_BLOCK_SIZE
    assert N_STATE_FEATURES == _STATE_BLOCK_SIZE + 3 * _OPP_BLOCK_SIZE


# ---------------------------------------------------------------------------
# _deal_fresh_game (no torch needed)
# ---------------------------------------------------------------------------

def test_deal_fresh_game_tile_count():
    from cracked.training.self_play import _deal_fresh_game
    rng = random.Random(0)
    my_c, opp_cs, wall = _deal_fresh_game(rng)
    # Each player receives exactly 13 standard tiles; bonus tiles (IDs 34+) may appear in wall
    assert my_c.sum() == 13
    for arr in opp_cs:
        assert arr.sum() == 13
    # All 136 standard tiles are accounted for across hands and wall
    standard_in_wall = sum(1 for t in wall if t < 34)
    assert my_c.sum() + sum(a.sum() for a in opp_cs) + standard_in_wall == 136


def test_deal_fresh_game_no_tile_exceeds_four():
    from cracked.training.self_play import _deal_fresh_game
    rng = random.Random(1)
    my_c, opp_cs, wall = _deal_fresh_game(rng)
    assert my_c.max() <= 4
    for arr in opp_cs:
        assert arr.max() <= 4


def test_deal_fresh_game_reproducible():
    from cracked.training.self_play import _deal_fresh_game
    my1, _, w1 = _deal_fresh_game(random.Random(42))
    my2, _, w2 = _deal_fresh_game(random.Random(42))
    assert np.array_equal(my1, my2)
    assert w1 == w2


def test_deal_fresh_game_different_seeds():
    from cracked.training.self_play import _deal_fresh_game
    _, _, w1 = _deal_fresh_game(random.Random(1))
    _, _, w2 = _deal_fresh_game(random.Random(2))
    assert w1 != w2


# ---------------------------------------------------------------------------
# ActorCritic (requires torch)
# ---------------------------------------------------------------------------

def test_actor_critic_output_shapes():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, N_STATE_FEATURES
    model = ActorCritic()
    model.eval()
    x = torch.zeros(4, N_STATE_FEATURES)
    with torch.no_grad():
        logits, value = model(x)
    assert logits.shape == (4, 34)
    assert value.shape == (4, 1)


def test_actor_critic_policy_logits_finite():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, N_STATE_FEATURES
    model = ActorCritic()
    x = torch.randn(1, N_STATE_FEATURES)
    with torch.no_grad():
        logits, _ = model(x)
    assert torch.isfinite(logits).all()


def test_actor_critic_value_finite():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, N_STATE_FEATURES
    model = ActorCritic()
    x = torch.randn(1, N_STATE_FEATURES)
    with torch.no_grad():
        _, value = model(x)
    assert torch.isfinite(value).all()


def test_actor_critic_save_load(tmp_path):
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, save_policy, load_policy
    model = ActorCritic()
    path = tmp_path / "policy.pt"
    save_policy(model, path)
    loaded = load_policy(path)
    # Weights must match
    for (n1, p1), (n2, p2) in zip(
        model.state_dict().items(), loaded.state_dict().items()
    ):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"


# ---------------------------------------------------------------------------
# collect_episode (requires torch)
# ---------------------------------------------------------------------------

def test_collect_episode_returns_episode():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, collect_episode, Episode
    model = ActorCritic()
    rng = random.Random(0)
    ep = collect_episode(model, rng)
    assert isinstance(ep, Episode)


def test_collect_episode_reward_finite():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, collect_episode
    model = ActorCritic()
    rng = random.Random(1)
    ep = collect_episode(model, rng)
    assert np.isfinite(ep.terminal_reward)


def test_collect_episode_steps_have_valid_actions():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, collect_episode
    model = ActorCritic()
    rng = random.Random(2)
    ep = collect_episode(model, rng)
    for step in ep.steps:
        assert 0 <= step.action < 34


def test_collect_episode_state_feat_shape():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, collect_episode
    model = ActorCritic()
    rng = random.Random(3)
    ep = collect_episode(model, rng)
    for step in ep.steps:
        assert step.state_feat.shape == (N_STATE_FEATURES,)


def test_collect_episode_reproducible():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, collect_episode
    model = ActorCritic()
    ep1 = collect_episode(model, random.Random(42))
    ep2 = collect_episode(model, random.Random(42))
    assert ep1.terminal_reward == ep2.terminal_reward
    assert len(ep1.steps) == len(ep2.steps)


# ---------------------------------------------------------------------------
# ppo_update (requires torch)
# ---------------------------------------------------------------------------

def test_ppo_update_returns_metrics():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, collect_episode, ppo_update
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    episodes = [collect_episode(model, random.Random(i)) for i in range(4)]
    metrics = ppo_update(model, optimizer, episodes)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_ppo_update_empty_episodes_no_crash():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, Episode, ppo_update
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = ppo_update(model, optimizer, [Episode()])
    assert metrics["policy_loss"] == 0.0


# ---------------------------------------------------------------------------
# evaluate_vs_heuristic (requires torch)
# ---------------------------------------------------------------------------

def test_evaluate_rates_bounded():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, evaluate_vs_heuristic
    model = ActorCritic()
    stats = evaluate_vs_heuristic(model, n_games=10, seed=0)
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert 0.0 <= stats["shoot_rate"] <= 1.0
    assert 0.0 <= stats["draw_rate"] <= 1.0
    assert stats["win_rate"] + stats["shoot_rate"] + stats["draw_rate"] == pytest.approx(1.0)


def test_evaluate_reproducible():
    torch = pytest.importorskip("torch")
    from cracked.training.self_play import ActorCritic, evaluate_vs_heuristic
    model = ActorCritic()
    s1 = evaluate_vs_heuristic(model, n_games=10, seed=7)
    s2 = evaluate_vs_heuristic(model, n_games=10, seed=7)
    assert s1["win_rate"] == s2["win_rate"]
    assert s1["mean_net"] == s2["mean_net"]
