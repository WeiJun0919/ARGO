"""
ProxyRewardTrainer - manages ensemble training, threshold calibration, and timing.
"""

import time

import numpy as np
import torch

from governance.proxy_reward.ensemble import ProxyRewardEnsemble
from governance.proxy_reward.buffer import ProxyRewardBuffer


class CalibrationResult:
    """Result of uncertainty threshold calibration."""

    def __init__(self, threshold, precision, recall_proxy, total_samples):
        self.threshold = threshold
        self.precision = precision
        self.recall_proxy = recall_proxy
        self.total_samples = total_samples


class ProxyRewardTrainer:
    """
    Orchestrates proxy reward model lifecycle.

    Stage 1 & 2:
        - Every real-evaluated step collects (state, action, real_reward).
        - No proxy model is used to skip retraining before Stage 3.

    Stage 3:
        - Calibrate an uncertainty threshold on accumulated real rewards.
        - Use low-uncertainty proxy reward predictions to skip retraining.
        - Add only real downstream-evaluated rewards to the supervised buffer.
        - Track proxy training/calibration/inference wall time so experiments can
          check whether proxy overhead offsets the skipped downstream fits.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_names: list,
        decision_dim: int = 8,
        hidden_dim: int = 128,
        n_members: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        update_every: int = 5,
        calibration_tolerance: float = 0.02,
        min_calibration_samples: int = 10,
        recalibrate_every: int = 0,
        device: str = "auto",
        log_timing: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_names = action_names
        self.decision_dim = int(decision_dim)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.update_every = int(update_every)
        self.calibration_tolerance = calibration_tolerance
        self.min_calibration_samples = int(min_calibration_samples)
        self.recalibrate_every = int(recalibrate_every)
        self.log_timing = bool(log_timing)

        self.buffer = ProxyRewardBuffer(action_dim=action_dim, state_dim=state_dim, decision_dim=self.decision_dim)
        self.ensemble = ProxyRewardEnsemble(
            state_dim=state_dim,
            action_dim=action_dim,
            decision_dim=self.decision_dim,
            hidden_dim=hidden_dim,
            n_members=n_members,
            device=device,
        )
        self.optimizer = torch.optim.AdamW(
            list(self.ensemble.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.is_active = False
        self.threshold = None
        self._step_counter = 0
        self._rng = np.random.default_rng(42)
        self.total_proxy_used = 0
        self.total_real_used = 0
        self._last_calibrated_size = 0
        self._calibration_count = 0

        # Timing counters. Non-overlapping total proxy overhead is estimated as:
        # setup + full calibration wall time + periodic training outside
        # calibration + proxy inference. Calibration wall time already includes
        # calibration training, so do not add calibration_train_time again there.
        self.setup_seconds = 0.0
        self.total_training_seconds = 0.0
        self.calibration_train_seconds = 0.0
        self.periodic_train_seconds = 0.0
        self.other_train_seconds = 0.0
        self.calibration_total_seconds = 0.0
        self.calibration_eval_seconds = 0.0
        self.proxy_decision_seconds = 0.0
        self.proxy_prediction_seconds = 0.0
        self.training_update_calls = 0
        self.calibration_update_calls = 0
        self.periodic_update_calls = 0
        self.other_update_calls = 0
        self.training_epochs_requested = 0
        self.training_steps_with_loss = 0
        self.proxy_decision_calls = 0
        self.proxy_prediction_calls = 0
        self.last_update_seconds = 0.0
        self.last_calibration_seconds = 0.0

    def set_init_wall_time(self, seconds: float):
        """Record external proxy import/initialization time measured by caller."""
        self.setup_seconds = float(seconds)

    def add_transition(self, state, action_id, action_onehot, decision_summary, exact_score):
        """Accumulate an exact post-operation validation observation."""
        self.buffer.add(state, action_id, action_onehot, decision_summary, exact_score)

    def train_step(self):
        """Run one gradient step on the current buffer. Returns training loss."""
        n = len(self.buffer)
        if n < 4:
            return float("nan")

        batch_size = min(64, n)
        states, actions, decisions, target_scores = self.buffer.sample(batch_size, rng=self._rng)
        self.ensemble.train()
        self.optimizer.zero_grad()

        states_t = torch.from_numpy(states).float().to(self.ensemble.device)
        actions_t = torch.from_numpy(actions).float().to(self.ensemble.device)
        decisions_t = torch.from_numpy(decisions).float().to(self.ensemble.device)
        targets_t = torch.from_numpy(target_scores).float().to(self.ensemble.device)

        total_loss = 0.0
        for member in self.ensemble._members:
            loss = member.nll_loss(states_t, actions_t, decisions_t, targets_t)
            total_loss += loss

        total_loss.backward()
        self.optimizer.step()
        return float(total_loss.item())

    def update_model(self, n_epochs: int = 5, reason: str = "manual"):
        """Run n_epochs gradient steps on the buffer and record wall-clock time."""
        t0 = time.perf_counter()
        losses = []
        for _ in range(n_epochs):
            loss = self.train_step()
            if not np.isnan(loss):
                losses.append(loss)
        elapsed = time.perf_counter() - t0

        self.last_update_seconds = elapsed
        self.total_training_seconds += elapsed
        self.training_update_calls += 1
        self.training_epochs_requested += int(n_epochs)
        self.training_steps_with_loss += len(losses)

        if reason == "calibration":
            self.calibration_train_seconds += elapsed
            self.calibration_update_calls += 1
        elif reason == "periodic":
            self.periodic_train_seconds += elapsed
            self.periodic_update_calls += 1
        else:
            self.other_train_seconds += elapsed
            self.other_update_calls += 1

        if self.log_timing:
            print(
                f"[ProxyReward][TIME] train_update reason={reason} "
                f"epochs={n_epochs} valid_steps={len(losses)} "
                f"time={elapsed:.3f}s train_total={self.total_training_seconds:.3f}s"
            )
        return losses

    def calibrate_threshold(self, n_calibration_samples: int = 50):
        """Calibrate the uncertainty threshold on a held-out subset of the buffer."""
        t_cal0 = time.perf_counter()
        n = len(self.buffer)
        min_required = self.min_calibration_samples
        if n < min_required:
            elapsed = time.perf_counter() - t_cal0
            self.calibration_total_seconds += elapsed
            self.last_calibration_seconds = elapsed
            print(
                f"[ProxyReward] Buffer too small ({n}/{min_required}) for calibration; "
                f"using conservative threshold=0."
            )
            if self.log_timing:
                print(f"[ProxyReward][TIME] calibration skipped time={elapsed:.3f}s")
            self.threshold = 0.0
            return None

        cal_size = min(n_calibration_samples, n)
        states, actions, decisions, targets = self.buffer.sample(cal_size, rng=self._rng)

        split = int(cal_size * 0.8)
        train_states, val_states = states[:split], states[split:]
        train_actions, val_actions = actions[:split], actions[split:]
        train_decisions, val_decisions = decisions[:split], decisions[split:]
        train_rewards, val_rewards = targets[:split], targets[split:]
        # Current behavior trains from the full replay buffer; keep these names
        # explicit to make the calibration split obvious in logs/code review.
        _ = (train_states, train_actions, train_decisions, train_rewards)

        train_before = self.total_training_seconds
        self.update_model(n_epochs=10, reason="calibration")
        train_elapsed = self.total_training_seconds - train_before

        preds = []
        uncertainties = []
        t_eval0 = time.perf_counter()
        for i in range(len(val_states)):
            p = self.ensemble.predict(val_states[i], val_actions[i], val_decisions[i])
            preds.append(p.mu)
            uncertainties.append(p.uncertainty)
        eval_elapsed = time.perf_counter() - t_eval0
        self.calibration_eval_seconds += eval_elapsed

        preds = np.array(preds)
        uncertainties = np.array(uncertainties)
        errors = np.abs(preds - val_rewards)

        tol = self.calibration_tolerance
        best_threshold = float("inf")
        best_f1 = 0.0

        for frac in np.linspace(0.1, 1.0, 50):
            t = float(np.percentile(uncertainties, frac * 100))
            mask_low = uncertainties <= t
            if mask_low.sum() < 2:
                continue
            accurate = errors[mask_low] <= tol
            precision = float(accurate.mean())
            recall_proxy = float(mask_low.mean())
            if precision + recall_proxy > 0:
                f1 = 2 * precision * recall_proxy / (precision + recall_proxy)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

        if best_threshold == float("inf"):
            best_threshold = float(np.median(uncertainties))

        mask_low = uncertainties <= best_threshold
        precision_val = float((errors[mask_low] <= tol).mean()) if mask_low.sum() > 0 else 0.0

        self.threshold = best_threshold
        self._last_calibrated_size = n
        self._calibration_count += 1
        result = CalibrationResult(
            threshold=best_threshold,
            precision=precision_val,
            recall_proxy=float(mask_low.mean()),
            total_samples=len(val_rewards),
        )

        elapsed = time.perf_counter() - t_cal0
        self.calibration_total_seconds += elapsed
        self.last_calibration_seconds = elapsed
        threshold_elapsed = max(0.0, elapsed - train_elapsed - eval_elapsed)

        print(
            f"[ProxyReward] Calibrated | threshold={best_threshold:.4f} "
            f"| precision={precision_val:.3f} | proxy_coverage={mask_low.mean():.3f} "
            f"| n_cal={len(val_rewards)}"
        )
        if self.log_timing:
            print(
                f"[ProxyReward][TIME] calibration total={elapsed:.3f}s "
                f"train={train_elapsed:.3f}s eval={eval_elapsed:.3f}s "
                f"threshold_search={threshold_elapsed:.3f}s "
                f"overhead_total={self.estimated_overhead_seconds():.3f}s "
                f"calibration_count={self._calibration_count}"
            )
        return result

    def has_usable_threshold(self):
        return (
            self.is_active
            and self.threshold is not None
            and self.threshold > 0
            and len(self.buffer) >= self.min_calibration_samples
        )

    def maybe_recalibrate(self, force: bool = False):
        """Recalibrate once enough new real-reward data has arrived."""
        n = len(self.buffer)
        if n < self.min_calibration_samples:
            if force:
                return self.calibrate_threshold()
            return None

        need = (
            force
            or self.threshold is None
            or self.threshold <= 0
            or (
                self.recalibrate_every > 0
                and n - self._last_calibrated_size >= self.recalibrate_every
            )
        )
        if not need:
            return None
        return self.calibrate_threshold()

    def should_use_proxy(self, state, action_onehot, decision_summary):
        """Return True if uncertainty < threshold (use proxy, skip retrain)."""
        if not self.has_usable_threshold():
            return False
        t0 = time.perf_counter()
        unc = self.ensemble.predict_uncertainty(state, action_onehot, decision_summary)
        elapsed = time.perf_counter() - t0
        self.proxy_decision_seconds += elapsed
        self.proxy_decision_calls += 1
        return unc < self.threshold

    def predict_post_performance(self, state, action_onehot, decision_summary):
        """Return the proxy mean post-operation validation performance."""
        t0 = time.perf_counter()
        pred = self.ensemble.predict(state, action_onehot, decision_summary)
        elapsed = time.perf_counter() - t0
        self.proxy_prediction_seconds += elapsed
        self.proxy_prediction_calls += 1
        if isinstance(pred.mu, float):
            return pred.mu
        return float(pred.mu[0])

    def on_step_completed(self, state, action_id, action_onehot, decision_summary, exact_score, used_proxy: bool):
        """Called after each env step. Handles buffer update and periodic refresh.

        Proxy-used steps do not have a true downstream-evaluated reward, so they
        are counted for coverage but are not fed back into the supervised proxy
        buffer. This avoids self-training the proxy on its own predictions.
        """
        self._step_counter += 1

        if used_proxy:
            if self.is_active:
                self.total_proxy_used += 1
            return

        if exact_score is None:
            return
        self.add_transition(state, action_id, action_onehot, decision_summary, exact_score)

        if not self.is_active:
            return

        self.total_real_used += 1

        if self.update_every > 0 and self.total_real_used % self.update_every == 0:
            losses = self.update_model(n_epochs=3, reason="periodic")
            if losses:
                active_total = self.total_proxy_used + self.total_real_used
                print(
                    f"[ProxyReward] Updated | real_steps={self.total_real_used} "
                    f"| loss={losses[-1]:.4f} | "
                    f"proxy_ratio={self.total_proxy_used}/{max(1, active_total)} "
                    f"| train_time={self.last_update_seconds:.3f}s "
                    f"| periodic_train_total={self.periodic_train_seconds:.3f}s "
                    f"| overhead_total={self.estimated_overhead_seconds():.3f}s"
                )

    def activate(self):
        """Switch to active mode (called when Stage 3 begins)."""
        self.is_active = True
        print("[ProxyReward] Activated for Stage 3.")

    def estimated_overhead_seconds(self) -> float:
        """Best-effort non-overlapping proxy overhead estimate."""
        return float(
            self.setup_seconds
            + self.calibration_total_seconds
            + self.periodic_train_seconds
            + self.other_train_seconds
            + self.proxy_decision_seconds
            + self.proxy_prediction_seconds
        )

    def timing_summary(self):
        """Return proxy timing numbers useful for wall-clock attribution."""
        return {
            "estimated_proxy_overhead_sec": round(float(self.estimated_overhead_seconds()), 6),
            "setup_sec": round(float(self.setup_seconds), 6),
            "total_training_sec": round(float(self.total_training_seconds), 6),
            "calibration_train_sec": round(float(self.calibration_train_seconds), 6),
            "periodic_train_sec": round(float(self.periodic_train_seconds), 6),
            "other_train_sec": round(float(self.other_train_seconds), 6),
            "calibration_total_sec": round(float(self.calibration_total_seconds), 6),
            "calibration_eval_sec": round(float(self.calibration_eval_seconds), 6),
            "proxy_decision_sec": round(float(self.proxy_decision_seconds), 6),
            "proxy_prediction_sec": round(float(self.proxy_prediction_seconds), 6),
            "training_update_calls": int(self.training_update_calls),
            "calibration_update_calls": int(self.calibration_update_calls),
            "periodic_update_calls": int(self.periodic_update_calls),
            "other_update_calls": int(self.other_update_calls),
            "training_epochs_requested": int(self.training_epochs_requested),
            "training_steps_with_loss": int(self.training_steps_with_loss),
            "proxy_decision_calls": int(self.proxy_decision_calls),
            "proxy_prediction_calls": int(self.proxy_prediction_calls),
            "last_update_sec": round(float(self.last_update_seconds), 6),
            "last_calibration_sec": round(float(self.last_calibration_seconds), 6),
        }

    def print_timing_summary(self, prefix: str = "[ProxyReward][TIME]"):
        """Print a compact timing summary for proxy overhead analysis."""
        t = self.timing_summary()
        print(
            f"{prefix} setup={t['setup_sec']:.3f}s | "
            f"calibration_total={t['calibration_total_sec']:.3f}s "
            f"(cal_train={t['calibration_train_sec']:.3f}s, "
            f"cal_eval={t['calibration_eval_sec']:.3f}s) | "
            f"periodic_train={t['periodic_train_sec']:.3f}s "
            f"(calls={t['periodic_update_calls']}, epochs_requested={t['training_epochs_requested']}) | "
            f"proxy_decision={t['proxy_decision_sec']:.3f}s/{t['proxy_decision_calls']} | "
            f"proxy_predict={t['proxy_prediction_sec']:.3f}s/{t['proxy_prediction_calls']} | "
            f"estimated_proxy_overhead={t['estimated_proxy_overhead_sec']:.3f}s"
        )

    def summary(self):
        return {
            "buffer_size": len(self.buffer),
            "is_active": self.is_active,
            "threshold": self.threshold,
            "total_proxy_used": self.total_proxy_used,
            "total_real_used": self.total_real_used,
            "total_steps": self._step_counter,
            "min_calibration_samples": self.min_calibration_samples,
            "last_calibrated_size": self._last_calibrated_size,
            "calibration_count": self._calibration_count,
            "timing": self.timing_summary(),
        }
