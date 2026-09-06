"""
Ensemble of M ProxyRewardMLP networks.

Each member is initialised with a different random seed, providing
epistemic uncertainty via disagreement (variance across μ predictions)
in addition to the aleatoric uncertainty (mean of individual σ²).
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from argo.governance.proxy_reward.model import ProxyRewardMLP


@dataclass
class EnsemblePrediction:
    """Output of an ensemble forward pass."""
    mu: float           # mean of member means
    uncertainty: float   # mean(var) + var(means) — total uncertainty
    mu_per_member: list[float]   # list of M means
    sigma2_per_member: list[float]  # list of M variances


class ProxyRewardEnsemble:
    """
    Ensemble of M ProxyRewardMLP networks.

    Inference:
        mu_ensemble = mean(mu_1, ..., mu_M)
        uncertainty = mean(sigma2_i) + var(mu_i)

    The uncertainty is the sum of:
      - Aleatoric (mean of each network's own variance estimate)
      - Epistemic (variance across the M means, capturing model disagreement)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        decision_dim: int = 8,
        hidden_dim: int = 128,
        n_members: int = 3,
        seeds: list[int] | None = None,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.decision_dim = int(decision_dim)
        self.hidden_dim = hidden_dim
        self.n_members = n_members
        self._device = device
        self._members: list[ProxyRewardMLP] = []
        base_seeds = list(seeds or [42, 123, 777])
        while len(base_seeds) < self.n_members:
            base_seeds.append(base_seeds[-1] + 104729)
        self._seeds = base_seeds

        self._build_members()

    def _build_members(self):
        for i in range(self.n_members):
            torch.manual_seed(self._seeds[i])
            np.random.seed(self._seeds[i])

            member = ProxyRewardMLP(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                decision_dim=self.decision_dim,
                hidden_dim=self.hidden_dim,
            )
            device = self._resolve_device()
            member = member.to(device)
            self._members.append(member)

    def _resolve_device(self) -> torch.device:
        if self._device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device)

    @property
    def device(self) -> torch.device:
        return self._members[0].device if self._members else self._resolve_device()

    def train(self, mode: bool = True):
        for m in self._members:
            m.train(mode)
        return self

    def eval(self):
        return self.train(mode=False)

    def named_parameters(self):
        """Flatten all member parameters for a single optimizer."""
        params = []
        for m in self._members:
            params.extend(list(m.parameters()))
        return params

    def parameters(self):
        return self.named_parameters()

    def predict(self, state_np: np.ndarray, action_np: np.ndarray, decision_np: np.ndarray) -> EnsemblePrediction:
        """
        Forward pass through all M members and aggregate.

        Parameters
        ----------
        state_np  : (state_dim,) or (batch, state_dim)
        action_np : (action_dim,) or (batch, action_dim)

        Returns
        -------
        EnsemblePrediction with mu and uncertainty
        """
        if state_np.ndim == 1:
            state_np = state_np[np.newaxis, :]
            action_np = action_np[np.newaxis, :]
            decision_np = decision_np[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        state_t = torch.from_numpy(state_np).float().to(self.device)
        action_t = torch.from_numpy(action_np).float().to(self.device)
        decision_t = torch.from_numpy(decision_np).float().to(self.device)

        mu_list, sigma2_list = [], []
        self.eval()
        with torch.no_grad():
            for member in self._members:
                mu, log_var = member.forward(state_t, action_t, decision_t)
                sigma2 = log_var.exp()
                mu_list.append(mu.cpu().numpy())
                sigma2_list.append(sigma2.cpu().numpy())

        mu_arr = np.array(mu_list)          # (M, batch)
        sigma2_arr = np.array(sigma2_list)  # (M, batch)

        mu_ensemble = mu_arr.mean(axis=0)   # (batch,)
        mean_sigma2 = sigma2_arr.mean(axis=0)
        var_mu = mu_arr.var(axis=0)
        uncertainty = mean_sigma2 + var_mu  # aleatoric + epistemic

        if squeeze:
            mu_ensemble = float(mu_ensemble[0])
            uncertainty = float(uncertainty[0])
            mu_list = [float(m[0]) for m in mu_list]
            sigma2_list = [float(s[0]) for s in sigma2_list]
        else:
            mu_list = mu_arr.tolist()
            sigma2_list = sigma2_arr.tolist()

        return EnsemblePrediction(
            mu=float(mu_ensemble) if squeeze else mu_ensemble.tolist(),
            uncertainty=uncertainty if squeeze else uncertainty.tolist(),
            mu_per_member=mu_list,
            sigma2_per_member=sigma2_list,
        )

    def predict_uncertainty(self, state_np: np.ndarray, action_np: np.ndarray, decision_np: np.ndarray) -> float:
        """Convenience: return just the uncertainty scalar."""
        pred = self.predict(state_np, action_np, decision_np)
        return pred.uncertainty if isinstance(pred.uncertainty, float) else float(pred.uncertainty[0])
