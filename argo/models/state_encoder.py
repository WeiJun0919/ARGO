from collections import deque

import numpy as np

from argo.data.diagnostics import compute_diagnostics


class StateEncoder:
    """Encode the paper's global, operation, history, and temporal state.

    The learnable gate lives in :class:`PPOAgent`, so it is optimized directly
    by the actor/critic loss as required by Eq. (4).
    """

    OPERATION_ORDER = ("remove", "augment", "repair", "relabel")
    OP_FEATURE_DIM = 4
    GLOBAL_BASE_DIM = 12
    TEMPORAL_DIM = 4

    def __init__(self, action_dim, history_len=5, max_steps=100):
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)
        self.max_steps = max(1, int(max_steps))
        self.history = deque(maxlen=self.history_len)
        self.usage_counts = np.zeros(self.action_dim, dtype="float32")
        self.step_count = 0
        self._state_dim = None
        self._n_classes = None
        self._previous_snapshot = None
        self._last_reward = 0.0

    @staticmethod
    def _safe_mean(values, default=0.0):
        values = np.asarray(values, dtype=float)
        return float(values.mean()) if values.size else float(default)

    @staticmethod
    def _normalise(values):
        values = np.asarray(values, dtype="float32")
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

    @staticmethod
    def _entropy(class_distribution):
        p = np.asarray(class_distribution, dtype=float)
        p = p[p > 0]
        if len(p) <= 1:
            return 0.0
        return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))

    @staticmethod
    def _feature_quality(X):
        try:
            if hasattr(X, "toarray"):
                X = X.toarray()
            arr = np.asarray(X)
            if arr.ndim > 2:
                arr = arr.reshape(len(arr), -1)
            arr = arr.astype(float, copy=False)
            missing = float((~np.isfinite(arr)).mean())
            finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            med = np.median(finite, axis=0)
            mad = np.median(np.abs(finite - med), axis=0) + 1e-8
            z = np.abs((finite - med) / mad)
            noisy_cells = float((z > 6.0).mean())
            inconsistency = float(np.tanh(np.mean(np.minimum(z, 10.0)) / 5.0))
            density = float(1.0 - np.mean(np.isclose(finite, 0.0)))
            return missing, noisy_cells, inconsistency, np.clip(density, 0.0, 1.0)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def state_dim(self, dataset, model=None):
        n_classes = int(getattr(dataset, "n_classes", 0) or 0)
        if n_classes <= 0:
            n_classes = int(np.max(dataset.y_train) + 1)
        expected = (
            self.GLOBAL_BASE_DIM
            + n_classes
            + len(self.OPERATION_ORDER) * self.OP_FEATURE_DIM
            + self.history_len * self.action_dim
            + self.action_dim
            + 3
            + self.TEMPORAL_DIM
        )
        if self._state_dim is not None and expected != self._state_dim:
            raise ValueError("The number of classes changed during a governance trajectory")
        self._n_classes = n_classes
        self._state_dim = expected
        return expected

    def layout(self, dataset=None, model=None):
        if self._state_dim is None:
            if dataset is None:
                raise ValueError("dataset is required before the state layout is known")
            self.state_dim(dataset, model)
        global_dim = self.GLOBAL_BASE_DIM + self._n_classes
        op_start = global_dim
        op_end = op_start + len(self.OPERATION_ORDER) * self.OP_FEATURE_DIM
        history_end = self._state_dim - self.TEMPORAL_DIM
        return {
            "state_dim": self._state_dim,
            "global": (0, global_dim),
            "operation": (op_start, op_end),
            "history": (op_end, history_end),
            "temporal": (history_end, self._state_dim),
            "operation_count": len(self.OPERATION_ORDER),
            "operation_feature_dim": self.OP_FEATURE_DIM,
        }

    def reset(self):
        self.history.clear()
        self.usage_counts.fill(0.0)
        self.step_count = 0
        self._previous_snapshot = None
        self._last_reward = 0.0

    def add_action(self, action_id, reward=0.0):
        one_hot = np.zeros(self.action_dim, dtype="float32")
        if 0 <= int(action_id) < self.action_dim:
            one_hot[int(action_id)] = 1.0
            self.usage_counts[int(action_id)] += 1.0
        self.history.append(one_hot)
        self.step_count += 1
        self._last_reward = float(np.tanh(float(reward)))

    def encode(self, dataset, model, validation_score_override=None):
        self.state_dim(dataset, model)
        diag = compute_diagnostics(dataset.X_train, dataset.y_train, model)
        train_score = float(model.score(dataset.X_train, dataset.y_train))
        val_score = (
            float(validation_score_override)
            if validation_score_override is not None
            else float(model.score(dataset.X_val, dataset.y_val))
        )
        losses = np.asarray(model.losses(dataset.X_train, dataset.y_train), dtype=float)
        proba = np.asarray(model.predict_proba(dataset.X_train), dtype=float)
        confidence = proba.max(axis=1)
        pred = proba.argmax(axis=1)
        disagreement = (pred != np.asarray(dataset.y_train)).astype(float)
        class_dist = np.asarray(diag.class_distribution, dtype="float32")
        missing, feature_noise, inconsistency, density = self._feature_quality(dataset.X_train)

        loss_mean = self._safe_mean(losses)
        loss_std = float(losses.std()) if losses.size else 0.0
        uncertainty = 1.0 - confidence
        high_loss = float((losses > np.quantile(losses, 0.75)).mean()) if losses.size else 0.0
        suspicious = np.maximum(diag.label_noise_scores, diag.outlier_scores)
        suspicious_mask = suspicious >= np.quantile(suspicious, 0.75) if len(suspicious) else np.zeros(0, bool)
        if suspicious_mask.any():
            suspicious_dist = np.bincount(
                np.asarray(dataset.y_train)[suspicious_mask], minlength=self._n_classes
            ).astype(float)
            suspicious_dist /= max(1.0, suspicious_dist.sum())
            suspicious_imbalance = float(suspicious_dist.max() - suspicious_dist.min())
        else:
            suspicious_imbalance = 0.0

        global_state = np.concatenate([
            np.array([
                np.tanh(np.log1p(len(dataset.y_train)) / 12.0),
                float(diag.imbalance_score), self._entropy(class_dist), density,
                train_score, val_score, np.clip(train_score - val_score, -1.0, 1.0),
                np.tanh(loss_mean), np.tanh(loss_std), self._safe_mean(confidence),
                float(confidence.std()) if confidence.size else 0.0,
                float(diag.estimated_label_noise_ratio + diag.estimated_outlier_ratio) / 2.0,
            ], dtype="float32"),
            class_dist,
        ])

        if uncertainty.size:
            minority = int(np.argmin(class_dist))
            minority_uncertainty = self._safe_mean(uncertainty[np.asarray(dataset.y_train) == minority])
        else:
            minority_uncertainty = 0.0
        operation_state = np.concatenate([
            [diag.estimated_outlier_ratio, high_loss, self._safe_mean(suspicious), suspicious_imbalance],
            [diag.imbalance_score, 1.0 - density, self._safe_mean(uncertainty), minority_uncertainty],
            [feature_noise, inconsistency, missing, diag.estimated_outlier_ratio],
            [diag.estimated_label_noise_ratio, self._safe_mean(disagreement), high_loss, 1.0 - self._safe_mean(confidence)],
        ]).astype("float32")

        padded = [np.zeros(self.action_dim, dtype="float32") for _ in range(self.history_len - len(self.history))]
        padded.extend(list(self.history))
        recent = np.concatenate(padded) if padded else np.zeros(0, dtype="float32")
        usage = self.usage_counts / max(1.0, float(self.step_count))
        progress = min(1.0, self.step_count / float(self.max_steps))
        history_state = np.concatenate([
            recent, usage.astype("float32"),
            np.array([progress, 1.0 - progress, self._last_reward], dtype="float32"),
        ])

        snapshot = np.array([
            val_score, np.tanh(loss_mean),
            float(diag.estimated_label_noise_ratio + diag.estimated_outlier_ratio) / 2.0,
            float(diag.imbalance_score),
        ], dtype="float32")
        temporal = np.zeros(self.TEMPORAL_DIM, dtype="float32") if self._previous_snapshot is None else np.clip(snapshot - self._previous_snapshot, -1.0, 1.0)
        self._previous_snapshot = snapshot

        state = np.concatenate([global_state, operation_state, history_state, temporal])
        if len(state) != self._state_dim:
            raise RuntimeError(f"state dimension mismatch: got {len(state)}, expected {self._state_dim}")
        return self._normalise(state)
