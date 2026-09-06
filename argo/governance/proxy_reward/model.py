"""
Heteroscedastic proxy from Eq. (10)-(11): (state, operation, rho(Delta D))
maps to post-operation validation performance and uncertainty.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ProxyRewardMLP(nn.Module):
    """
    Two-layer MLP that maps (state, action) -> (mu, log_sigma2).

    mu        : predicted reward mean
    log_sigma2: log of predicted variance (unbounded; clamped for stability)
    """

    def __init__(self, state_dim: int, action_dim: int, decision_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.decision_dim = int(decision_dim)
        input_dim = state_dim + action_dim + self.decision_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # (mu, log_sigma2)
        )

        # Orthogonal init for better gradient flow at start
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor, decision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mu        : (batch,) reward mean
        log_var   : (batch,) log(variance), clamped to [-5, 2]
        """
        x = torch.cat([state, action, decision], dim=-1)
        out = self.net(x)
        mu = out[:, 0]
        log_var = out[:, 1].clamp(-5.0, 2.0)
        return mu, log_var

    def nll_loss(self, state: torch.Tensor, action: torch.Tensor, decision: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss under Gaussian assumption:

            NLL = 0.5 * (y - mu)^2 / sigma^2 + 0.5 * log(sigma^2)

        Low uncertainty (small sigma2) on well-covered samples;
        high uncertainty on sparse/ambiguous (state, action) pairs.
        """
        mu, log_var = self.forward(state, action, decision)
        sigma2 = log_var.exp()
        nll = 0.5 * (target_score - mu) ** 2 / sigma2 + 0.5 * log_var
        return nll.mean()

    def predict(self, state_np: np.ndarray, action_np: np.ndarray, decision_np: np.ndarray) -> tuple[float, float]:
        """
        Deterministic forward pass (no sampling). Returns (mu, sigma2).
        """
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state_np).float().to(device)
            action_t = torch.from_numpy(action_np).float().to(device)
            decision_t = torch.from_numpy(decision_np).float().to(device)
            mu, log_var = self.forward(state_t, action_t, decision_t)
            mu = mu.cpu().item()
            sigma2 = log_var.exp().cpu().item()
        return mu, sigma2

    @property
    def device(self):
        return next(self.parameters()).device
