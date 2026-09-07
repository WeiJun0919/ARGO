"""
Replay buffer for exact (state, action, decision-summary, performance) tuples.

Collected during Stage 1 & 2, consumed during Stage 3 training.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Transition:
    """A single exact-evaluation observation for Eq. (11)."""
    state: np.ndarray   # (state_dim,)
    action_id: int      # scalar action index
    action_onehot: np.ndarray  # (action_dim,)
    decision_summary: np.ndarray
    target_score: float


class ProxyRewardBuffer:
    """
    Append-only experience buffer for proxy model training.

    Stores transitions collected during all stages.
    Provides stratified mini-batch sampling (balanced across action types).
    """

    def __init__(self, action_dim: int, state_dim: int, decision_dim: int = 8, max_size: int = 5000):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.decision_dim = int(decision_dim)
        self.max_size = max_size
        self._buffer: deque[Transition] = deque(maxlen=max_size)

    def add(self, state: np.ndarray, action_id: int, action_onehot: np.ndarray, decision_summary: np.ndarray, target_score: float):
        """Append a new transition."""
        self._buffer.append(
            Transition(
                state=state.astype("float32"),
                action_id=action_id,
                action_onehot=action_onehot.astype("float32"),
                decision_summary=self._pad_to_fixed(np.asarray(decision_summary, dtype="float32"), self.decision_dim),
                target_score=float(target_score),
            )
        )

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def rewards(self) -> np.ndarray:
        """Backward-compatible alias for post-operation performance targets."""
        return np.array([t.target_score for t in self._buffer], dtype="float32")

    def _pad_to_fixed(self, arr: np.ndarray, dim: int) -> np.ndarray:
        """Pad a 1-D array to `dim` by appending zeros on the right."""
        if arr.shape[0] >= dim:
            return arr[:dim]
        return np.pad(arr, (0, dim - arr.shape[0]), mode="constant", constant_values=0.0)

    def _pad_actions(self, action_list: list[np.ndarray]) -> np.ndarray:
        """Pad all action_onehot arrays to the fixed action_dim and stack them."""
        return np.stack([self._pad_to_fixed(a, self.action_dim) for a in action_list])

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> tuple:
        """
        Stratified random sample: roughly equal representation per action type.

        Returns
        -------
        states   : (batch, state_dim)
        actions  : (batch, action_dim)
        rewards  : (batch,)
        """
        n = len(self._buffer)
        if n == 0:
            raise ValueError("Buffer is empty; cannot sample.")
        batch_size = min(batch_size, n)

        rng = rng or np.random.default_rng()

        # Group by action_id
        by_action: dict[int, list[int]] = {}
        for idx, t in enumerate(self._buffer):
            by_action.setdefault(t.action_id, []).append(idx)

        n_actions = len(by_action)
        per_action = max(1, batch_size // n_actions)
        indices = []
        remaining = batch_size

        for aid in sorted(by_action):
            pool = by_action[aid]
            rng.shuffle(pool)
            take = min(per_action, len(pool), remaining)
            indices.extend(pool[:take])
            remaining -= take
            if remaining <= 0:
                break

        # Fill leftovers randomly
        while len(indices) < batch_size:
            idx = rng.integers(n)
            if idx not in indices:
                indices.append(idx)

        indices = np.array(indices, dtype=int)
        rng.shuffle(indices)

        # Pad states and actions to fixed dimensions before stacking
        batch_states = [self._buffer[i].state for i in indices]
        states = np.stack([self._pad_to_fixed(s, self.state_dim) for s in batch_states])
        actions = self._pad_actions([self._buffer[i].action_onehot for i in indices])
        decisions = np.stack([self._buffer[i].decision_summary for i in indices])
        rewards = self.rewards[indices]

        return states, actions, decisions, rewards

    def get_recent(self, n: int):
        """Return the last n transitions."""
        n = min(n, len(self._buffer))
        buf = list(self._buffer)[-n:]
        states = np.stack([self._pad_to_fixed(t.state, self.state_dim) for t in buf])
        actions = self._pad_actions([t.action_onehot for t in buf])
        decisions = np.stack([t.decision_summary for t in buf])
        rewards = np.array([t.target_score for t in buf], dtype="float32")
        return states, actions, decisions, rewards
