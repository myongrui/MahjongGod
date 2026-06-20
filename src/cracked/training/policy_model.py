"""RL model as an engine seat policy.

ModelPolicy lets a trained ActorCritic network drive a seat in the real game
engine, the same way HeuristicPolicy does. It implements the Policy interface
from cracked.policy.

Discards are chosen by the policy network (greedy over valid tiles). Claim
decisions are currently delegated to a heuristic policy as a *placeholder* — the
network's action space does not yet cover pong/kong/chow. Teaching the model to
claim is a later step (the action space is extended there).

Requires PyTorch (lazy-imported), so this module is kept out of the engine's
import path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from cracked.tiles import NTILES
from cracked.game_state import GameState
from cracked.policy import Policy, HeuristicPolicy
from cracked.training.features import extract_state_features


class ModelPolicy:
    """Drive a seat with a trained ActorCritic; heuristic claims for now."""

    def __init__(self, actor_critic, claim_policy: Optional[Policy] = None):
        self.net = actor_critic
        self.claim_policy: Policy = claim_policy or HeuristicPolicy()

    def choose_discard(self, view: GameState) -> Optional[int]:
        import torch

        feat = extract_state_features(view)
        with torch.no_grad():
            logits, _ = self.net(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))
        logits = logits.squeeze(0).numpy()

        concealed = view.my_hand.concealed
        masked = np.where(concealed > 0, logits, -np.inf)
        return int(np.argmax(masked))

    def wants_pong(self, view: GameState, tile: int) -> bool:
        return self.claim_policy.wants_pong(view, tile)

    def wants_kong(self, view: GameState, tile: int) -> bool:
        return self.claim_policy.wants_kong(view, tile)

    def choose_chow(
        self, view: GameState, tile: int, options: list[tuple[int, int, int]]
    ) -> Optional[tuple[int, int, int]]:
        return self.claim_policy.choose_chow(view, tile, options)
