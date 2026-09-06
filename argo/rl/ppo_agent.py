from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except ImportError as exc:  # pragma: no cover - exercised only on incomplete installs
    raise ImportError("ARGO's high-level PPO policy requires PyTorch") from exc


class _GatedActorCritic(nn.Module):
    """Actor-critic with the adaptive operation-state gate from Eq. (4)."""

    def __init__(self, state_dim, action_dim, layout, hidden_dim=128):
        super().__init__()
        self.layout = layout
        g0, g1 = layout["global"]
        o0, o1 = layout["operation"]
        self.g0, self.g1, self.o0, self.o1 = g0, g1, o0, o1
        self.op_count = int(layout["operation_count"])
        self.op_feature_dim = int(layout["operation_feature_dim"])
        self.gate = nn.Sequential(
            nn.Linear((g1 - g0) + (o1 - o0), 64), nn.Tanh(),
            nn.Linear(64, self.op_count), nn.Sigmoid(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def gated_state(self, state):
        global_state = state[:, self.g0:self.g1]
        operation_state = state[:, self.o0:self.o1]
        alpha = self.gate(torch.cat([global_state, operation_state], dim=-1))
        operation_groups = operation_state.reshape(-1, self.op_count, self.op_feature_dim)
        gated_operation = (operation_groups * alpha.unsqueeze(-1)).reshape(-1, self.o1 - self.o0)
        gated = torch.cat([
            state[:, :self.o0], gated_operation, state[:, self.o1:]
        ], dim=-1)
        return gated, alpha

    def forward(self, state):
        gated, alpha = self.gated_state(state)
        hidden = self.backbone(gated)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), alpha


@dataclass
class _Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class PPOAgent:
    """Clipped PPO actor-critic used for high-level operation selection.

    The model is initialized lazily because the class-count portion of the state
    is dataset dependent.  The gating network, actor, and critic share one PPO
    optimizer, which lets operation relevance and scheduling co-adapt in Stage III.
    """

    def __init__(
        self,
        action_names,
        state_encoder=None,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=6,
        minibatch_size=32,
        rollout_size=16,
        max_grad_norm=0.5,
        device="auto",
        random_state=42,
        **_ignored,
    ):
        self.action_names = list(action_names)
        self.state_encoder = state_encoder
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.rollout_size = int(rollout_size)
        self.max_grad_norm = float(max_grad_norm)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.rng = np.random.default_rng(random_state)
        torch.manual_seed(int(random_state))
        self.model = None
        self.optimizer = None
        self.buffer = []
        self.training_enabled = True
        self.update_count = 0
        self.last_losses = {}

    def _ensure_model(self, state):
        if self.model is not None:
            return
        state_dim = int(np.asarray(state).size)
        if self.state_encoder is not None:
            layout = self.state_encoder.layout()
        else:
            layout = {
                "global": (0, state_dim), "operation": (state_dim, state_dim),
                "history": (state_dim, state_dim), "temporal": (state_dim, state_dim),
                "operation_count": 1, "operation_feature_dim": 0,
            }
        if layout["operation"][0] == layout["operation"][1]:
            # Backward-compatible no-gate layout for custom encoders.
            layout = dict(layout)
            layout.update({"global": (0, state_dim), "operation": (state_dim, state_dim)})
            self.model = _PlainActorCritic(state_dim, len(self.action_names), self.hidden_dim).to(self.device)
        else:
            self.model = _GatedActorCritic(state_dim, len(self.action_names), layout, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _tensor(self, state):
        arr = np.asarray(state, dtype="float32").reshape(1, -1)
        return torch.from_numpy(arr).to(self.device)

    def select_action(self, state, deterministic=False):
        self._ensure_model(state)
        with torch.no_grad():
            logits, _, _ = self.model(self._tensor(state))
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return int(action.item())

    def action_name(self, action_id):
        return self.action_names[int(action_id)]

    def record_transition(self, state, action_id, reward, next_state, done=False):
        if not self.training_enabled:
            return
        self._ensure_model(state)
        with torch.no_grad():
            logits, value, _ = self.model(self._tensor(state))
            dist = Categorical(logits=logits)
            action_t = torch.tensor([int(action_id)], device=self.device)
            log_prob = float(dist.log_prob(action_t).item())
        self.buffer.append(_Transition(
            np.asarray(state, dtype="float32").copy(), int(action_id), float(reward),
            np.asarray(next_state, dtype="float32").copy(), bool(done), log_prob,
            float(value.item()),
        ))
        if len(self.buffer) >= self.rollout_size:
            self.update_policy(force=True)

    def update(self, action_id, reward, state=None, next_state=None, done=False, **_kwargs):
        """Compatibility wrapper; new code should call ``record_transition``."""
        if state is not None and next_state is not None:
            self.record_transition(state, action_id, reward, next_state, done=done)

    def _advantages_and_returns(self):
        advantages = np.zeros(len(self.buffer), dtype="float32")
        gae = 0.0
        for i in range(len(self.buffer) - 1, -1, -1):
            tr = self.buffer[i]
            with torch.no_grad():
                _, next_value, _ = self.model(self._tensor(tr.next_state))
            nonterminal = 0.0 if tr.done else 1.0
            delta = tr.reward + self.gamma * float(next_value.item()) * nonterminal - tr.value
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[i] = gae
        values = np.array([tr.value for tr in self.buffer], dtype="float32")
        return advantages, advantages + values

    def update_policy(self, force=False):
        if not self.buffer or (not force and len(self.buffer) < self.rollout_size):
            return self.last_losses
        states = np.stack([tr.state for tr in self.buffer])
        actions = np.array([tr.action for tr in self.buffer], dtype="int64")
        old_log_probs = np.array([tr.log_prob for tr in self.buffer], dtype="float32")
        advantages, returns = self._advantages_and_returns()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        old_log_t = torch.from_numpy(old_log_probs).float().to(self.device)
        adv_t = torch.from_numpy(advantages).float().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)
        n = len(states)
        loss_acc = {"policy": [], "value": [], "entropy": []}

        for _ in range(self.update_epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx = torch.as_tensor(order[start:start + self.minibatch_size], device=self.device)
                logits, values, _ = self.model(states_t[idx])
                dist = Categorical(logits=logits)
                new_log = dist.log_prob(actions_t[idx])
                ratio = torch.exp(new_log - old_log_t[idx])
                unclipped = ratio * adv_t[idx]
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (returns_t[idx] - values).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                loss_acc["policy"].append(float(policy_loss.item()))
                loss_acc["value"].append(float(value_loss.item()))
                loss_acc["entropy"].append(float(entropy.item()))

        self.last_losses = {key: float(np.mean(value)) for key, value in loss_acc.items()}
        self.update_count += 1
        self.buffer.clear()
        return self.last_losses

    def action_probabilities(self, state):
        self._ensure_model(state)
        with torch.no_grad():
            logits, _, _ = self.model(self._tensor(state))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return {name: float(probs[i]) for i, name in enumerate(self.action_names)}

    def gating_weights(self, state):
        self._ensure_model(state)
        with torch.no_grad():
            _, _, alpha = self.model(self._tensor(state))
        values = alpha.cpu().numpy()[0]
        names = getattr(self.state_encoder, "OPERATION_ORDER", tuple(range(len(values))))
        return {str(name): float(values[i]) for i, name in enumerate(names)}


class _PlainActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        hidden = self.backbone(state)
        alpha = torch.ones((len(state), 1), device=state.device)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), alpha
