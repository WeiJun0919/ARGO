import numpy as np

from argo.data.diagnostics import compute_diagnostics
from argo.selectors.base_selector import BaseSelector
from argo.selectors.ide_module import (
    PseudoLabelPair,
    run_stage1_ilde_warmup,
    warmup_selector_from_stage1_cache,
)


class NeuralSelectorBase(BaseSelector):
    """
    Base selector with IDE/ILDE Stage 1 warmup support.

    The warmup() method now runs the full Stage 1 pipeline:
      1. Fit model on noisy data
      2. ILDE Gram-matrix noise transition matrix estimation
      3. High-confidence clean sample identification
      4. Synthetic noise injection → pseudo-label pairs
      5. Supervised selector warmup on pseudo pairs
    """

    def __init__(self):
        self.last_loss = 0.0
        self.last_feedback_loss = None
        self.last_reward_update = None
        self.ilde_result = None
        self._pair_stats = {}
        self._warmup_scale = 1.0
        self._warmup_bias = 0.0
        self._policy_weights = None
        self._policy_bias = 0.0
        self._reward_baseline = 0.0
        self._policy_lr = 0.05
        self._last_policy_batch = None
        self._correction_bias = None
        self._last_correction_batch = None

    def warmup(self, dataset, model, clean_y_train=None, epochs=6):
        """
        ILDE-based noise-aware warmup.

        This remains a backward-compatible single-selector entrypoint.  The
        trainer-level fast path uses warmup_from_stage1_cache() so that ED2/ILDE
        detection is computed once and shared across selectors.
        """
        result = run_stage1_ilde_warmup(
            dataset=dataset,
            model=model,
            clean_y_train=clean_y_train,
            epochs=epochs,
            selector=self,
            verbose=False,
            modality=getattr(dataset, "modality", None),
        )
        self.ilde_result = result
        self.last_loss = result.get("pair_loss_final", 1.0)
        # Store the refined clean indices from the final warmup epoch so that
        # differential training can be activated with high-confidence clean data.
        self._clean_indices = result.get("_final_clean_indices")
        return result

    def warmup_from_stage1_cache(self, stage1_cache, dataset, model, epochs=6, verbose=False):
        """Warm up this selector from a shared Stage-1 ILDE/ED2 cache."""
        result = warmup_selector_from_stage1_cache(
            selector=self,
            stage1_cache=stage1_cache,
            dataset=dataset,
            model=model,
            inner_epochs=2,
            verbose=verbose,
        )
        self.ilde_result = result
        self.last_loss = result.get("pair_loss_final", 1.0)
        self._clean_indices = result.get("_final_clean_indices")
        return result

    def get_clean_indices(self):
        """Return high-confidence clean indices identified during warmup, or None."""
        return getattr(self, "_clean_indices", None)

    def warmup_with_pairs(self, pairs, X_all, y_all, model, epochs=2,
                          batch_size=64, lr=0.001, verbose=False):
        """Stage-I BCE warm-up on controlled corruptions (Eq. 6)."""
        n = len(pairs)
        if n == 0:
            return {"n_pairs": 0, "mean_loss": 0.0, "final_pair_acc": 0.0}

        y_clean_arr = np.array([p.y_clean for p in pairs])
        y_synth_arr = np.array([p.y_synthetic for p in pairs])
        noise_types = np.array([p.noise_type for p in pairs])
        severities = np.array([p.severity for p in pairs])

        if isinstance(self, NeuralRelabelSelector):
            target = (noise_types == "label_flip").astype(float)
        elif isinstance(self, NeuralRepairSelector):
            target = np.isin(noise_types, ["feature_corrupt", "outlier"]).astype(float)
        elif isinstance(self, NeuralAugmentSelector):
            # Corrupted examples are negative augmentation candidates.
            target = np.zeros(n, dtype=float)
        else:
            target = np.ones(n, dtype=float)
        pair_weights = np.maximum(severities, 0.1)

        X_flat = np.array([
            p.x.flatten() if p.x.ndim > 1 else p.x
            for p in pairs
        ])

        raw = np.asarray(self._score_batch(X_flat, y_synth_arr, model), dtype=float)
        raw = np.clip(raw, 0.0, 1.0)
        if isinstance(self, NeuralRepairSelector):
            utility = raw
        elif isinstance(self, NeuralAugmentSelector):
            utility = 1.0 - raw
        else:
            utility = 1.0 - raw

        losses, accuracies = [], []
        step_lr = max(0.01, float(lr) * 50.0)
        x = utility - 0.5
        for epoch in range(max(1, int(epochs))):
            logits = self._warmup_scale * x + self._warmup_bias
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            error = (prob - target) * pair_weights
            self._warmup_scale -= step_lr * float(np.mean(error * x))
            self._warmup_bias -= step_lr * float(np.mean(error))
            loss = -np.average(
                target * np.log(prob + 1e-8) + (1.0 - target) * np.log(1.0 - prob + 1e-8),
                weights=pair_weights,
            )
            acc = float(((prob >= 0.5).astype(float) == target).mean())
            losses.append(float(loss))
            accuracies.append(acc)
            if verbose:
                print(f"    [selector warmup epoch {epoch + 1}/{epochs}] loss={loss:.4f} acc={acc:.3f}")

        self.last_loss = float(np.mean(losses[-3:])) if len(losses) >= 3 else float(np.mean(losses))
        if isinstance(self, NeuralRelabelSelector):
            n_classes = int(max(y_clean_arr.max(), y_synth_arr.max()) + 1)
            counts = np.ones((n_classes, n_classes), dtype="float32")
            for noisy_label, clean_label, weight in zip(y_synth_arr, y_clean_arr, pair_weights):
                counts[int(noisy_label), int(clean_label)] += float(weight)
            transition = counts / counts.sum(axis=1, keepdims=True)
            self._correction_bias = np.log(transition + 1e-8)
        self._pair_stats = {
            "n_pairs": n,
            "mean_loss": self.last_loss,
            "final_pair_acc": float(np.mean(accuracies[-max(1, len(accuracies) // 4):])) if accuracies else 0.0,
            "noise_type_counts": {
                "label_flip": int((noise_types == "label_flip").sum()),
                "feature_corrupt": int((noise_types == "feature_corrupt").sum()),
                "outlier": int((noise_types == "outlier").sum()),
            },
        }
        return self._pair_stats

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    @staticmethod
    def _normalise_feature_columns(features):
        features = np.asarray(features, dtype="float32")
        if features.ndim == 1:
            features = features[:, None]
        lo = np.nanmin(features, axis=0)
        hi = np.nanmax(features, axis=0)
        return np.nan_to_num((features - lo) / (hi - lo + 1e-8), nan=0.0)

    def _rank_with_policy(self, candidate_indices, features, heuristic, budget):
        """Rank candidates and cache Bernoulli decisions for Eq. (8)."""
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        if len(candidate_indices) == 0:
            return candidate_indices
        features = self._normalise_feature_columns(features)
        heuristic = np.clip(np.asarray(heuristic, dtype=float), 1e-4, 1.0 - 1e-4)
        if self._policy_weights is None or len(self._policy_weights) != features.shape[1]:
            self._policy_weights = np.zeros(features.shape[1], dtype="float32")
        base_logit = np.log(heuristic / (1.0 - heuristic))
        logits = (
            base_logit
            + self._warmup_scale * (heuristic - 0.5)
            + self._warmup_bias
            + features @ self._policy_weights
            + self._policy_bias
        )
        probability = self._sigmoid(logits)
        k = min(max(1, int(budget)), len(candidate_indices))
        local = np.argpartition(probability, -k)[-k:]
        local = local[np.argsort(probability[local])[::-1]]
        selected = np.zeros(len(candidate_indices), dtype="float32")
        selected[local] = 1.0

        # Keep selected examples plus a bounded set of negatives for policy updates.
        negative = np.where(selected == 0)[0]
        if len(negative) > max(4 * k, 64):
            negative = np.random.default_rng(42).choice(negative, size=max(4 * k, 64), replace=False)
        keep = np.concatenate([local, negative])
        self._last_policy_batch = {
            "candidate_indices": candidate_indices[keep],
            "features": features[keep],
            "decisions": selected[keep],
            "probabilities": probability[keep],
        }
        return candidate_indices[local]

    def _score_batch(self, X_batch, y_synth_batch, model):
        """
        Compute per-sample noise suspicion score for a batch.

        Subclasses override this to implement selector-specific scoring logic.
        Higher returned score → more suspicious (needs correction).

        Base implementation: confidence-based scoring.
        """
        try:
            proba = model.predict_proba(X_batch)
            pred = proba.argmax(axis=1)
            conf = proba.max(axis=1)
            score = np.where(pred != y_synth_batch, conf, 1 - conf)
            return 1.0 - score
        except Exception:
            return np.full(len(X_batch), 0.5)

    def update_with_feedback(self, dataset, model, indices, modifications, feedback):
        if feedback is None:
            return
        self.last_feedback_loss = 1.0 - feedback.acceptance_rate()
        batch = self._last_policy_batch
        if batch is None:
            return
        accepted = set(map(int, getattr(feedback, "accepted_indices", []) or []))
        rejected = set(map(int, getattr(feedback, "rejected_indices", []) or []))
        if not accepted and not rejected:
            return
        mask = np.array([int(i) in accepted or int(i) in rejected for i in batch["candidate_indices"]])
        if not mask.any():
            return
        target = np.array([1.0 if int(i) in accepted else 0.0 for i in batch["candidate_indices"][mask]])
        error = target - batch["probabilities"][mask]
        self._policy_weights += self._policy_lr * np.mean(error[:, None] * batch["features"][mask], axis=0)
        self._policy_bias += self._policy_lr * float(error.mean())
        if self._correction_bias is not None and getattr(feedback, "corrected_labels", None):
            for index, clean_label in feedback.corrected_labels.items():
                if 0 <= int(index) < len(dataset.y_train):
                    observed = int(dataset.y_train[int(index)])
                    self._correction_bias[observed] *= 0.99
                    self._correction_bias[observed, int(clean_label)] += self._policy_lr

    def update_with_reward(self, reward, accepted=True):
        self.last_reward_update = float(reward)
        batch = self._last_policy_batch
        if batch is None:
            return
        advantage = float(reward) - self._reward_baseline
        self._reward_baseline = 0.9 * self._reward_baseline + 0.1 * float(reward)
        score = batch["decisions"] - batch["probabilities"]
        gradient = np.mean(score[:, None] * batch["features"], axis=0)
        self._policy_weights += self._policy_lr * advantage * gradient
        self._policy_bias += self._policy_lr * advantage * float(score.mean())
        corr = self._last_correction_batch
        if corr is not None and self._correction_bias is not None:
            for observed, chosen, probability in zip(corr["observed"], corr["chosen"], corr["probabilities"]):
                grad = -np.asarray(probability, dtype="float32")
                grad[int(chosen)] += 1.0
                self._correction_bias[int(observed)] += self._policy_lr * advantage * grad

    def stats(self):
        base = {
            "last_loss": self.last_loss,
            "last_feedback_loss": self.last_feedback_loss,
            "last_reward_update": self.last_reward_update,
            "reward_baseline": float(self._reward_baseline),
            "policy_weight_norm": (
                0.0 if self._policy_weights is None else float(np.linalg.norm(self._policy_weights))
            ),
            "warmup_scale": float(self._warmup_scale),
            "warmup_bias": float(self._warmup_bias),
        }
        if self.ilde_result:
            base["ilde_pair_loss"] = self.ilde_result.get("pair_loss_final")
            base["ilde_n_pairs_mean"] = self.ilde_result.get("n_pairs_mean", 0)
            base["ilde_clean_samples_mean"] = self.ilde_result.get("n_clean_samples_mean", 0)
        base["pair_stats"] = self._pair_stats
        return base


class NeuralRemoveSelector(NeuralSelectorBase):
    """Remove selector: identifies samples that should be deleted."""

    def _score_batch(self, X_batch, y_synth_batch, model):
        """Remove-specific scoring: emphasizes loss + confidence disagreement.

        Following §4.1.3 weighting: s_i^remove = 0.60 * c_i^wrong + 0.40 * l_i
        """
        n = len(X_batch)
        proba = model.predict_proba(X_batch)
        pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        c_wrong = np.where(pred != y_synth_batch, conf, 0.0)

        try:
            losses = model.losses(X_batch, y_synth_batch)
            l_i = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
        except Exception:
            l_i = np.ones(n) * 0.5

        score = 0.60 * c_wrong + 0.40 * l_i
        return 1.0 - score

    def select(self, dataset, model, budget):
        diag = compute_diagnostics(dataset.X_train, dataset.y_train, model)
        losses = model.losses(dataset.X_train, dataset.y_train)
        loss_score = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
        score = 0.6 * diag.label_noise_scores + 0.4 * loss_score
        features = np.column_stack([
            diag.label_noise_scores, diag.outlier_scores, loss_score,
            np.full(len(losses), diag.estimated_label_noise_ratio),
        ])
        idx = self._rank_with_policy(np.arange(len(losses)), features, score, budget)
        return idx, {}


class NeuralRelabelSelector(NeuralSelectorBase):
    """Relabel selector: identifies samples whose labels should be corrected."""

    def _score_batch(self, X_batch, y_synth_batch, model):
        """Relabel-specific scoring: combines confidence, loss, and ensemble disagreement.

        Following §4.1.3 weighting:
          s_i^relabel = 0.35 * c_i^wrong + 0.30 * l_i + 0.35 * d_i
        """
        n = len(X_batch)
        proba = model.predict_proba(X_batch)
        pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        c_wrong = np.where(pred != y_synth_batch, conf, 0.0)

        try:
            losses = model.losses(X_batch, y_synth_batch)
            l_i = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
        except Exception:
            l_i = np.ones(n) * 0.5

        try:
            if hasattr(model, "predict_proba_ensemble"):
                _, disagreement = model.predict_proba_ensemble(X_batch, n_ensemble=5)
            else:
                disagreement = np.zeros(n)
        except Exception:
            disagreement = np.zeros(n)

        score = 0.35 * c_wrong + 0.30 * l_i + 0.35 * disagreement
        return 1.0 - score

    def select(self, dataset, model, budget):
        proba = model.predict_proba(dataset.X_train)
        pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        losses = model.losses(dataset.X_train, dataset.y_train)
        loss_score = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)

        if hasattr(model, "predict_proba_ensemble"):
            _, disagreement = model.predict_proba_ensemble(dataset.X_train, n_ensemble=5)
        else:
            disagreement = np.zeros(len(dataset.y_train))

        mismatch = np.where(pred != dataset.y_train, conf, 0.0)
        score = np.clip(0.35 * mismatch + 0.30 * loss_score + 0.35 * disagreement, 0.0, 1.0)
        features = np.column_stack([mismatch, loss_score, conf, disagreement])
        idx = self._rank_with_policy(np.arange(len(pred)), features, score, budget)
        observed = np.asarray(dataset.y_train, dtype=int)[idx]
        if self._correction_bias is not None and self._correction_bias.shape[0] >= proba.shape[1]:
            logits = np.log(np.clip(proba[idx], 1e-8, 1.0)) + self._correction_bias[observed, :proba.shape[1]]
            logits -= logits.max(axis=1, keepdims=True)
            correction_proba = np.exp(logits)
            correction_proba /= correction_proba.sum(axis=1, keepdims=True) + 1e-8
            new_labels = correction_proba.argmax(axis=1)
            self._last_correction_batch = {
                "observed": observed.copy(), "chosen": new_labels.copy(),
                "probabilities": correction_proba.copy(),
            }
        else:
            new_labels = pred[idx]
            self._last_correction_batch = None
        return idx, {"new_labels": new_labels}


class NeuralAugmentSelector(NeuralSelectorBase):
    """Augment selector: identifies clean samples to synthesize variants from."""

    def __init__(self, noise_scale=0.01):
        super().__init__()
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(42)

    def _score_batch(self, X_batch, y_synth_batch, model):
        """Augment-specific scoring: prefer clean, high-confidence samples."""
        n = len(X_batch)
        proba = model.predict_proba(X_batch)
        pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        agree = np.where(pred == y_synth_batch, conf, 1 - conf)

        try:
            losses = model.losses(X_batch, y_synth_batch)
            l_i = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
            clean_signal = 1.0 - l_i
        except Exception:
            clean_signal = np.ones(n) * 0.5

        try:
            if hasattr(model, "predict_proba_ensemble"):
                _, disagreement = model.predict_proba_ensemble(X_batch, n_ensemble=5)
            else:
                disagreement = np.zeros(n)
            agree_signal = 1.0 - disagreement
        except Exception:
            agree_signal = np.ones(n) * 0.5

        score = 0.35 * agree + 0.35 * clean_signal + 0.30 * agree_signal
        return 1.0 - score

    def select(self, dataset, model, budget):
        proba = model.predict_proba(dataset.X_train)
        pred = proba.argmax(axis=1)
        confidence = proba.max(axis=1)
        clean_idx = np.where(pred == dataset.y_train)[0]
        if len(clean_idx) == 0:
            return np.array([], dtype=int), {"augmented_X": dataset.X_train[:0], "augmented_y": dataset.y_train[:0]}

        counts = np.bincount(dataset.y_train, minlength=dataset.n_classes)
        inv_freq = 1.0 / (counts[dataset.y_train[clean_idx]] + 1e-8)
        inv_freq = inv_freq / (inv_freq.max() + 1e-8)
        try:
            losses = model.losses(dataset.X_train[clean_idx], dataset.y_train[clean_idx])
            uncertainty = 1.0 - confidence[clean_idx]
        except Exception:
            losses = np.zeros(len(clean_idx))
            uncertainty = 1.0 - confidence[clean_idx]
        loss_score = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
        utility = np.clip(0.45 * inv_freq + 0.30 * uncertainty + 0.25 * (1.0 - loss_score), 0.0, 1.0)
        features = np.column_stack([inv_freq, uncertainty, confidence[clean_idx], 1.0 - loss_score])
        idx = self._rank_with_policy(clean_idx, features, utility, budget)

        X_aug = dataset.X_train[idx] + self.rng.normal(0, self.noise_scale, size=dataset.X_train[idx].shape)
        mods = {"augmented_X": X_aug, "augmented_y": dataset.y_train[idx]}
        if dataset.y_train_clean is not None:  # evaluation bookkeeping only
            mods["augmented_y_clean"] = dataset.y_train_clean[idx]
        return idx, mods


class NeuralRepairSelector(NeuralSelectorBase):
    """Repair selector: identifies samples with anomalous features to repair."""

    def __init__(self):
        super().__init__()
        self._repair_coef = None
        self._repair_intercept = None
        self._repair_correction_loss = None

    def warmup_with_pairs(self, pairs, X_all, y_all, model, epochs=2,
                          batch_size=64, lr=0.001, verbose=False):
        result = super().warmup_with_pairs(
            pairs, X_all, y_all, model, epochs=epochs,
            batch_size=batch_size, lr=lr, verbose=verbose,
        )
        repair_pairs = [
            pair for pair in pairs
            if pair.noise_type in {"feature_corrupt", "outlier"}
            and getattr(pair, "x_clean", None) is not None
        ]
        if repair_pairs:
            corrupted = np.asarray([np.asarray(p.x).reshape(-1) for p in repair_pairs], dtype=float)
            clean = np.asarray([np.asarray(p.x_clean).reshape(-1) for p in repair_pairs], dtype=float)
            corrupted = np.nan_to_num(corrupted, nan=0.0, posinf=0.0, neginf=0.0)
            clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
            x_mean, y_mean = corrupted.mean(axis=0), clean.mean(axis=0)
            covariance = np.mean((corrupted - x_mean) * (clean - y_mean), axis=0)
            variance = np.mean((corrupted - x_mean) ** 2, axis=0) + 1e-6
            self._repair_coef = np.clip(covariance / variance, 0.0, 2.0)
            self._repair_intercept = y_mean - self._repair_coef * x_mean
            predicted = corrupted * self._repair_coef + self._repair_intercept
            self._repair_correction_loss = float(np.mean((predicted - clean) ** 2))
            result["correction_loss"] = self._repair_correction_loss
        return result

    def _score_batch(self, X_batch, y_synth_batch, model):
        """Repair-specific scoring: based on MAD outlier detection.

        Following §5.2: o_i = (1/d) Σ_j min(|x_ij - med_j| / MAD_j, 10)
        """
        try:
            Xd = np.asarray(X_batch)
            if Xd.ndim > 2:
                Xd = Xd.reshape(len(Xd), -1)
            Xd = np.nan_to_num(Xd.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

            med = np.median(Xd, axis=0)
            mad = np.median(np.abs(Xd - med), axis=0) + 1e-8
            z = np.abs((Xd - med) / mad)
            raw = np.mean(np.minimum(z, 10.0), axis=1)
            o_i = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
            return o_i
        except Exception:
            return np.zeros(len(X_batch))

    def select(self, dataset, model, budget):
        diag = compute_diagnostics(dataset.X_train, dataset.y_train, model)
        X = np.asarray(dataset.X_train).copy()
        med = np.median(X, axis=0)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        z_all = np.abs((X - mean) / std)
        row_inconsistency = np.mean(np.minimum(z_all, 10.0), axis=1) / 10.0
        missing = np.mean(~np.isfinite(X), axis=1).astype(float)
        features = np.column_stack([
            diag.outlier_scores, row_inconsistency, missing,
            np.full(len(X), diag.estimated_outlier_ratio),
        ])
        heuristic = np.clip(0.6 * diag.outlier_scores + 0.4 * row_inconsistency, 0.0, 1.0)
        idx = self._rank_with_policy(np.arange(len(X)), features, heuristic, budget)
        repaired = X[idx].copy()
        z = np.abs((repaired - mean) / std)
        rows, cols = np.where((z > 3.0) | ~np.isfinite(repaired))
        if len(rows) > 0:
            if self._repair_coef is not None and self._repair_coef.size == X.shape[1]:
                safe = repaired.copy()
                bad_rows, bad_cols = np.where(~np.isfinite(safe))
                if len(bad_rows):
                    safe[bad_rows, bad_cols] = med[bad_cols]
                predicted = safe * self._repair_coef + self._repair_intercept
                low = np.nanquantile(X, 0.01, axis=0)
                high = np.nanquantile(X, 0.99, axis=0)
                predicted = np.clip(predicted, low, high)
                repaired[rows, cols] = predicted[rows, cols]
            else:
                repaired[rows, cols] = med[cols]
        return idx, {"repaired_X": repaired}

    def stats(self):
        out = super().stats()
        out["repair_correction_loss"] = self._repair_correction_loss
        return out
