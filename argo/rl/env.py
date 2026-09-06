from dataclasses import dataclass
import copy
import numpy as np
from argo.governance.operations import apply_operation
from argo.governance.reward import compute_reward
from argo.data.diagnostics import compute_diagnostics


@dataclass
class StepResult:
    state: object
    next_state: object
    action_id: int
    action_name: str
    reward: float
    old_score: float
    new_score: float
    n_processed: int
    indices: object
    modifications: dict
    user_score: float = 0.0
    selector_updated: bool = False
    reverted: bool = False
    feedback: object = None
    used_proxy: bool = False
    diff_score: float = None
    diff_prefiltered: bool = False
    issues_resolved_ratio: float = 0.0
    operational_cost: float = 0.0
    exact_evaluation: bool = True
    decision_summary: object = None


class ARGOEnv:
    def __init__(
        self,
        dataset,
        model,
        state_encoder,
        selectors,
        budget_ratio=0.025,
        rollback_on_negative=True,
        proxy_reward_trainer=None,
        rollback_tolerance=0.003,
        use_differential_training=False,
        clean_indices=None,
        diff_clean_epochs=5,
        diff_dirty_epochs=1,
        diff_dirty_scope="active",
        diff_alpha=1e-4,
        diff_mode="shadow",
        diff_confirm_margin=0.02,
        diff_skip_user_threshold=0.0,
        image_ft_epochs=3,
        image_ft_lr_factor=0.05,
        exact_eval_interval=10,
        full_retrain_modified_ratio=0.20,
        verbose=False,
    ):
        self.dataset = dataset
        self.model = model
        self.state_encoder = state_encoder
        self.selectors = selectors
        self.budget_ratio = budget_ratio
        self.rollback_on_negative = rollback_on_negative
        self.proxy_reward_trainer = proxy_reward_trainer
        self.rollback_tolerance = rollback_tolerance
        self.verbose = bool(verbose)

        self._use_diff = bool(use_differential_training)
        self._diff_model = None
        self._diff_deferred = False
        self._dirty_indices = np.array([], dtype=int)
        self._diff_clean_epochs = int(diff_clean_epochs)
        self._diff_dirty_epochs = int(diff_dirty_epochs)
        self._diff_dirty_scope = str(diff_dirty_scope or "active").lower()
        self._diff_alpha = float(diff_alpha)
        self._diff_mode = str(diff_mode or "shadow").lower()
        self._diff_confirm_margin = float(diff_confirm_margin)
        self._diff_skip_user_threshold = float(diff_skip_user_threshold)
        self._img_ft_epochs = int(image_ft_epochs)
        self._img_ft_lr_factor = float(image_ft_lr_factor)
        self.exact_eval_interval = max(1, int(exact_eval_interval))
        self.full_retrain_modified_ratio = float(full_retrain_modified_ratio)
        self._steps_since_exact = 0
        self._modified_since_full = 0
        self._validation_score_override = None
        self._noop_streak = 0

        self.model.fit(self.dataset.X_train, self.dataset.y_train)
        initial_diag = compute_diagnostics(self.dataset.X_train, self.dataset.y_train, self.model)
        self._initial_issue_mass = {
            "remove": float(np.maximum(initial_diag.label_noise_scores, initial_diag.outlier_scores).sum()),
            "relabel": float(initial_diag.label_noise_scores.sum()),
            "repair": float(initial_diag.outlier_scores.sum()),
            "augment": float(max(initial_diag.imbalance_score, 1e-6) * len(self.dataset.y_train)),
        }
        if self._use_diff and clean_indices is not None:
            self.enable_differential(clean_indices)

    def get_state(self):
        return self.state_encoder.encode(
            self.dataset, self.model,
            validation_score_override=self._validation_score_override,
        )

    def _action_onehot(self, action_id):
        onehot = np.zeros(self.state_encoder.action_dim, dtype="float32")
        if action_id < len(onehot):
            onehot[action_id] = 1.0
        return onehot

    @staticmethod
    def _filter_array_like(value, keep):
        try:
            return np.asarray(value)[keep]
        except Exception:
            return value

    def _protected_clean_mask(self):
        mask = getattr(self.dataset, "protected_clean_mask", None)
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(self.dataset.y_train):
            return None
        return mask

    def _filter_protected_clean_indices(self, action_name, indices, modifications):
        if action_name in {"noop", "augment"} or len(indices) == 0:
            return indices, modifications
        mask = self._protected_clean_mask()
        if mask is None:
            return indices, modifications

        indices = np.asarray(indices, dtype=int)
        valid = (indices >= 0) & (indices < len(mask))
        safe_indices = indices.clip(0, max(0, len(mask) - 1))
        allowed = valid & (~mask[safe_indices])
        if allowed.all():
            return indices, modifications

        new_indices = indices[allowed]
        new_mods = dict(modifications)
        if action_name == "relabel" and "new_labels" in new_mods:
            new_mods["new_labels"] = self._filter_array_like(new_mods["new_labels"], allowed)
        elif action_name == "repair" and "repaired_X" in new_mods:
            new_mods["repaired_X"] = self._filter_array_like(new_mods["repaired_X"], allowed)
        elif action_name == "augment":
            for key in ["augmented_X", "augmented_y", "augmented_y_clean"]:
                if key in new_mods:
                    new_mods[key] = self._filter_array_like(new_mods[key], allowed)

        if self.verbose:
            dropped = int(len(indices) - len(new_indices))
            if dropped > 0:
                print(f"[ARGOEnv] protected clean filter dropped {dropped} {action_name} rows")
        return new_indices, new_mods

    @staticmethod
    def _dirty_indices_after_operation(action_name, old_n, new_n, indices):
        if action_name == "augment":
            return np.arange(old_n, new_n, dtype=int)
        if action_name == "remove":
            return np.array([], dtype=int)
        return np.asarray(indices, dtype=int)

    def _make_differential_model(self, dataset, clean_indices=None):
        from argo.models.downstream_model import DifferentialDownstreamModel
        diff_model = DifferentialDownstreamModel(
            base_model=self.model,
            clean_indices=clean_indices,
            use_differential=True,
            clean_epochs=self._diff_clean_epochs,
            dirty_epochs=self._diff_dirty_epochs,
            dirty_scope=self._diff_dirty_scope,
            alpha=self._diff_alpha,
            image_ft_epochs=self._img_ft_epochs,
            image_ft_lr_factor=self._img_ft_lr_factor,
            verbose=self.verbose,
        )
        diff_model.pretrain_clean(
            dataset.X_train,
            dataset.y_train,
            clean_mask=getattr(dataset, "protected_clean_mask", None),
        )
        return diff_model

    def _diff_fit(self, new_dataset, dirty_indices, action_name=None):
        """Fit the active downstream model.

        In legacy/replace mode, the differential model becomes self.model.
        In shadow mode this method should not be used for final scoring.
        """
        if not self._use_diff or self._diff_mode == "shadow":
            self.model.fit(new_dataset.X_train, new_dataset.y_train)
            return

        if self._diff_model is None:
            self._diff_model = self._make_differential_model(new_dataset)

        self._diff_model.fit(
            new_dataset.X_train,
            new_dataset.y_train,
            dirty_indices=dirty_indices,
            clean_mask=getattr(new_dataset, "protected_clean_mask", None),
            action_name=action_name,
        )
        self.model = self._diff_model

    def _diff_shadow_fit_score(self, new_dataset, dirty_indices, action_name=None):
        """Update the shadow differential model and return its validation score.

        The real downstream model is left untouched.  The shadow model is built
        lazily so a run does not pay the clean-pretrain cost unless an action
        can actually be filtered by the differential evaluator.  Caller is
        responsible for restoring self._diff_model when the candidate operation
        is rejected.
        """
        if not self._use_diff:
            return None
        if self._diff_model is None:
            self._diff_model = self._make_differential_model(new_dataset)
            self._diff_deferred = False
        self._diff_model.fit(
            new_dataset.X_train,
            new_dataset.y_train,
            dirty_indices=dirty_indices,
            clean_mask=getattr(new_dataset, "protected_clean_mask", None),
            action_name=action_name,
        )
        return float(self._diff_model.score(new_dataset.X_val, new_dataset.y_val))

    def _diff_shadow_should_confirm(self, action_name, old_score, diff_score, user_score):
        """Return True if the candidate should be checked by the real model.

        Shadow differential training is intentionally conservative: it may skip
        only actions that both the fast differential evaluator and the user/oracle
        signal consider weak.  Relabel/remove are always confirmed because, in
        Adult experiments, they are often beneficial under RandomForest even when
        an SGD dirty-replay approximation is pessimistic.
        """
        if diff_score is None:
            return True
        if action_name in {"relabel", "remove"}:
            return True
        if user_score > self._diff_skip_user_threshold:
            return True
        return diff_score >= old_score - self._diff_confirm_margin

    def enable_differential(self, clean_indices):
        if not self._use_diff:
            return

        clean_indices = np.asarray(clean_indices, dtype=int)
        n = len(self.dataset.y_train)
        clean_indices = clean_indices[(clean_indices >= 0) & (clean_indices < n)]
        clean_mask = np.zeros(n, dtype=bool)
        clean_mask[clean_indices] = True
        self.dataset.protected_clean_mask = clean_mask

        model_type = str(getattr(self.model, "model_type", "") or "").lower()
        modality = str(getattr(self.dataset, "modality", "") or "").lower()
        dataset_name = str(getattr(self.dataset, "name", "") or "").lower()
        if model_type in {"image", "cifar", "cifar10"} or modality in {"image", "cifar", "cifar10"} or dataset_name == "cifar10":
            # The current DifferentialDownstreamModel supports tabular/text via
            # SGD dirty replay.  Image models would otherwise fall back to a full
            # ResNet fit and become slower than the baseline, so disable the
            # differential evaluator while still keeping the Stage-1 clean mask
            # for selector protection.
            self._use_diff = False
            self._diff_model = None
            self._diff_deferred = False
            print(
                "[ARGOEnv] Differential training disabled for image downstream "
                "model: no incremental image dirty replay is implemented; "
                "protected clean rows are still used by selectors."
            )
            return

        if self._diff_mode == "shadow":
            # In shadow mode the diff model is only a prefilter.  Build it lazily
            # on the first action that is not force-confirmed; otherwise Adult
            # runs with relabel/remove/positive-feedback augment pay clean
            # pretrain overhead without any chance to skip a real model fit.
            self._diff_model = None
            self._diff_deferred = True
        else:
            # Legacy/replace mode: differential replay replaces the downstream model.
            self._diff_model = self._make_differential_model(self.dataset, clean_indices=clean_indices)
            self.model = self._diff_model
            self._diff_deferred = False

        if self.verbose:
            status = "deferred" if self._diff_deferred else "active"
            print(
                f"[ARGOEnv] Differential training enabled with {len(clean_indices)} "
                f"protected clean rows | mode={self._diff_mode} | status={status}"
            )

    @staticmethod
    def _val_score_from_state(state, fallback_fn):
        try:
            # StateEncoder global layout: train performance at 4, validation at 5.
            return float(state[5])
        except Exception:
            return float(fallback_fn())

    def _legacy_step(self, action_id, action_name, feedback_provider=None, user_feedback=None, precomputed_state=None):
        state = precomputed_state if precomputed_state is not None else self.get_state()
        action_onehot = self._action_onehot(action_id)
        proxy_trainer = self.proxy_reward_trainer

        # Fast proxy path: the current proxy semantics are an abstract reward
        # estimate for (state, action) and do not apply the cleaning operation.
        # Therefore we can decide before selector/oracle/model-copy work.
        # This makes proxy coverage translate into wall-clock savings.
        if (
            proxy_trainer is not None
            and proxy_trainer.is_active
            and action_name != "noop"
            and proxy_trainer.should_use_proxy(state, action_onehot)
        ):
            old_score = self._val_score_from_state(
                state, lambda: self.model.score(self.dataset.X_val, self.dataset.y_val)
            )
            reward = proxy_trainer.predict_proxy_reward(state, action_onehot)
            proxy_trainer.on_step_completed(
                state=state,
                action_id=action_id,
                action_onehot=action_onehot,
                real_reward=0.0,
                used_proxy=True,
            )
            self.state_encoder.add_action(action_id)
            return StepResult(
                state, state, action_id, action_name, reward,
                old_score, old_score, 0, np.array([], dtype=int), {},
                0.0, False, False, None, True, None, False,
            )

        old_score = self._val_score_from_state(
            state, lambda: self.model.score(self.dataset.X_val, self.dataset.y_val)
        )
        old_n = len(self.dataset.y_train)
        budget = max(1, int(len(self.dataset.y_train) * self.budget_ratio))

        if action_name == "noop":
            indices, modifications = np.array([], dtype=int), {}
        else:
            indices, modifications = self.selectors[action_name].select(self.dataset, self.model, budget)
            indices = np.asarray(indices, dtype=int)
            indices, modifications = self._filter_protected_clean_indices(action_name, indices, modifications)
            if len(indices) == 0:
                action_name = "noop"
                modifications = {}

        if user_feedback is None and feedback_provider is not None:
            user_feedback = feedback_provider(self, action_id, action_name, indices, modifications)

        user_score = user_feedback.normalized_action_score() if user_feedback is not None else 0.0
        selector_updated = False

        if action_name != "noop" and user_feedback is not None and user_feedback.has_instance_feedback():
            sel = self.selectors.get(action_name)
            if hasattr(sel, "update_with_feedback"):
                sel.update_with_feedback(self.dataset, self.model, indices, modifications, user_feedback)
                selector_updated = True

        if user_feedback is not None and user_feedback.rejected_indices:
            rejected = set(map(int, user_feedback.rejected_indices))
            keep = np.array([int(i) not in rejected for i in indices], dtype=bool)
            indices = indices[keep]
            modifications = dict(modifications)
            if action_name == "relabel" and "new_labels" in modifications:
                modifications["new_labels"] = np.asarray(modifications["new_labels"])[keep]
            if action_name == "augment":
                for k in ["augmented_X", "augmented_y", "augmented_y_clean"]:
                    if k in modifications:
                        modifications[k] = modifications[k][keep]
            if action_name == "repair" and "repaired_X" in modifications:
                modifications["repaired_X"] = np.asarray(modifications["repaired_X"])[keep]
            if len(indices) == 0:
                action_name = "noop"
                modifications = {}

        if user_feedback is not None and action_name == "relabel" and user_feedback.corrected_labels:
            modifications = dict(modifications)
            current = modifications.get("new_labels", self.dataset.y_train[indices])
            labels = np.asarray(current).copy()
            for pos, i in enumerate(indices):
                if int(i) in user_feedback.corrected_labels:
                    labels[pos] = int(user_feedback.corrected_labels[int(i)])
            modifications["new_labels"] = labels

        used_proxy = False

        if True:
            old_dataset = None
            old_model = None
            if action_name == "noop":
                new_dataset = self.dataset
            else:
                # These copies are only needed for real operations/rollback.
                # Avoiding them on proxy/noop steps makes proxy coverage translate
                # into actual wall-clock savings, especially for RandomForest.
                old_dataset = self.dataset.copy()
                old_model = copy.deepcopy(self.model)
                new_dataset = apply_operation(self.dataset, action_name, indices, modifications)

            diff_score = None
            diff_prefiltered = False
            shadow_backup = None

            if action_name != "noop":
                dirty_after = self._dirty_indices_after_operation(
                    action_name,
                    old_n,
                    len(new_dataset.y_train),
                    indices,
                )

                if self._use_diff and self._diff_mode == "shadow":
                    # Do not run differential replay for candidates that will be
                    # confirmed by the real downstream model anyway.  This avoids
                    # the previous "diff + full retrain" double cost on Adult
                    # relabel/remove and on positive-feedback augment steps.
                    force_confirm = (
                        action_name in {"relabel", "remove"}
                        or user_score > self._diff_skip_user_threshold
                    )
                    if force_confirm:
                        self.model.fit(new_dataset.X_train, new_dataset.y_train)
                        new_score = self.model.score(new_dataset.X_val, new_dataset.y_val)
                    else:
                        shadow_backup = copy.deepcopy(self._diff_model)
                        diff_score = self._diff_shadow_fit_score(
                            new_dataset, dirty_after, action_name=action_name
                        )
                        should_confirm = self._diff_shadow_should_confirm(
                            action_name, old_score, diff_score, user_score
                        )
                        if not should_confirm:
                            # Cheap rejection: differential replay says the action is
                            # clearly harmful and user/oracle feedback is not positive.
                            # Keep the real model and dataset unchanged.
                            self._diff_model = shadow_backup
                            self._diff_deferred = self._diff_model is None
                            new_score = old_score
                            diff_prefiltered = True
                        else:
                            self.model.fit(new_dataset.X_train, new_dataset.y_train)
                            new_score = self.model.score(new_dataset.X_val, new_dataset.y_val)
                else:
                    self._diff_fit(new_dataset, dirty_after, action_name=action_name)
                    new_score = self.model.score(new_dataset.X_val, new_dataset.y_val)
            else:
                new_score = old_score

            real_reward = compute_reward(
                old_score, new_score, action_name,
                processed_ratio=len(indices) / max(1, len(self.dataset.y_train)),
                user_satisfaction=user_score,
            )

            reverted = False
            if diff_prefiltered:
                reverted = True
                reward = min(real_reward, -0.05)
            elif self.rollback_on_negative and new_score < old_score - self.rollback_tolerance:
                self.dataset = old_dataset
                self.model = old_model
                if self._use_diff and self._diff_mode == "shadow":
                    self._diff_model = shadow_backup
                    self._diff_deferred = self._diff_model is None
                reverted = True
                reward = min(real_reward, 20.0 * (new_score - old_score) - 0.05)
            else:
                self.dataset = new_dataset
                reward = real_reward

            if action_name != "noop" and action_name in self.selectors:
                sel = self.selectors[action_name]
                if hasattr(sel, "update_with_reward"):
                    sel.update_with_reward(reward)

            if proxy_trainer is not None:
                proxy_trainer.on_step_completed(
                    state=state,
                    action_id=action_id,
                    action_onehot=action_onehot,
                    real_reward=real_reward,
                    used_proxy=False,
                )

            self.state_encoder.add_action(action_id)
            next_state = state
            return StepResult(
                state, next_state, action_id, action_name, reward,
                old_score, new_score, len(indices), indices, modifications,
                user_score, selector_updated, reverted, user_feedback, used_proxy,
                diff_score, diff_prefiltered,
            )

        self.state_encoder.add_action(action_id)
        next_state = state
        return StepResult(
            state, next_state, action_id, action_name, reward,
            old_score, new_score, len(indices), indices, modifications,
            user_score, selector_updated, False, user_feedback, used_proxy,
        )

    @staticmethod
    def _decision_summary(old_dataset, new_dataset, action_name, indices, modifications, user_score):
        """rho(Delta D_t) used by the proxy in Eq. (10)."""
        old_n = max(1, len(old_dataset.y_train))
        new_n = len(new_dataset.y_train)
        old_dist = np.bincount(old_dataset.y_train, minlength=old_dataset.n_classes) / old_n
        new_dist = np.bincount(new_dataset.y_train, minlength=old_dataset.n_classes) / max(1, new_n)
        labels_changed = len(indices) / old_n if action_name == "relabel" else 0.0
        removed = max(0, old_n - new_n) / old_n
        added = max(0, new_n - old_n) / old_n
        feature_shift = 0.0
        if action_name == "repair" and len(indices):
            try:
                before = np.asarray(old_dataset.X_train)[np.asarray(indices, dtype=int)].astype(float)
                after = np.asarray(modifications.get("repaired_X", before)).astype(float)
                feature_shift = float(np.tanh(np.mean(np.abs(after - before))))
            except Exception:
                feature_shift = 0.0
        return np.asarray([
            len(indices) / old_n,
            (new_n - old_n) / old_n,
            float(np.abs(new_dist - old_dist).sum()),
            feature_shift,
            labels_changed,
            removed,
            added,
            float(user_score),
        ], dtype="float32")

    def _quality_and_cost(self, action_name, indices, modifications, user_feedback):
        indices = np.asarray(indices, dtype=int)
        n = max(1, len(self.dataset.y_train))
        if action_name == "noop" or len(indices) == 0:
            self._noop_streak += 1
            return 0.0, 0.1 * self._noop_streak
        self._noop_streak = 0
        diag = compute_diagnostics(self.dataset.X_train, self.dataset.y_train, self.model)
        valid = indices[(indices >= 0) & (indices < n)]
        if action_name == "relabel":
            mass = float(diag.label_noise_scores[valid].sum()) if len(valid) else 0.0
        elif action_name == "repair":
            mass = float(diag.outlier_scores[valid].sum()) if len(valid) else 0.0
        elif action_name == "remove":
            mass = float(np.maximum(diag.label_noise_scores[valid], diag.outlier_scores[valid]).sum()) if len(valid) else 0.0
        else:
            labels = np.asarray(modifications.get("augmented_y", []), dtype=int)
            counts = np.bincount(self.dataset.y_train, minlength=self.dataset.n_classes)
            mass = float(np.sum(1.0 - counts[labels] / max(1, counts.max()))) if len(labels) else 0.0
        eta = float(np.clip(mass / max(1e-8, self._initial_issue_mass.get(action_name, 1.0)), 0.0, 1.0))
        weights = {"remove": 1.5, "augment": 1.0, "repair": 1.2, "relabel": 1.0}
        cost = weights.get(action_name, 1.0) * len(indices) / n
        if user_feedback is not None:
            reviewed = len(getattr(user_feedback, "accepted_indices", []) or []) + len(getattr(user_feedback, "rejected_indices", []) or [])
            cost += 0.25 * reviewed / n
        return eta, float(cost)

    @staticmethod
    def _filter_rejected(action_name, indices, modifications, user_feedback):
        if user_feedback is None or not getattr(user_feedback, "rejected_indices", None):
            return indices, modifications
        rejected = set(map(int, user_feedback.rejected_indices))
        keep = np.array([int(i) not in rejected for i in indices], dtype=bool)
        indices = np.asarray(indices, dtype=int)[keep]
        modifications = dict(modifications)
        keys = {
            "relabel": ["new_labels"],
            "repair": ["repaired_X"],
            "augment": ["augmented_X", "augmented_y", "augmented_y_clean", "augmented_X_clean"],
        }.get(action_name, [])
        for key in keys:
            if key in modifications:
                modifications[key] = np.asarray(modifications[key])[keep]
        return indices, modifications

    def step(self, action_id, action_name, feedback_provider=None, user_feedback=None, precomputed_state=None):
        """Execute first, then choose proxy or exact downstream evaluation.

        This ordering is essential: Eq. (10) conditions on rho(Delta D_t), so a
        proxy decision cannot be made from only the pre-operation state/action.
        """
        state = precomputed_state if precomputed_state is not None else self.get_state()
        old_score = self._val_score_from_state(
            state, lambda: self.model.score(self.dataset.X_val, self.dataset.y_val)
        )
        requested_action = action_name
        old_n = len(self.dataset.y_train)
        budget = max(1, int(old_n * self.budget_ratio))

        if action_name == "noop":
            indices, modifications = np.array([], dtype=int), {}
        else:
            indices, modifications = self.selectors[action_name].select(self.dataset, self.model, budget)
            indices = np.asarray(indices, dtype=int)
            indices, modifications = self._filter_protected_clean_indices(action_name, indices, modifications)
            if len(indices) == 0:
                action_name, modifications = "noop", {}

        if user_feedback is None and feedback_provider is not None:
            user_feedback = feedback_provider(self, action_id, action_name, indices, modifications)
        user_score = user_feedback.normalized_action_score() if user_feedback is not None else 0.0
        selector_updated = False
        if action_name != "noop" and user_feedback is not None and user_feedback.has_instance_feedback():
            selector = self.selectors.get(action_name)
            if hasattr(selector, "update_with_feedback"):
                selector.update_with_feedback(self.dataset, self.model, indices, modifications, user_feedback)
                selector_updated = True
        indices, modifications = self._filter_rejected(action_name, indices, modifications, user_feedback)
        if len(indices) == 0 and action_name != "noop":
            action_name, modifications = "noop", {}
        if user_feedback is not None and action_name == "relabel" and user_feedback.corrected_labels:
            labels = np.asarray(modifications.get("new_labels", self.dataset.y_train[indices])).copy()
            for pos, index in enumerate(indices):
                if int(index) in user_feedback.corrected_labels:
                    labels[pos] = int(user_feedback.corrected_labels[int(index)])
            modifications = dict(modifications)
            modifications["new_labels"] = labels

        eta, operational_cost = self._quality_and_cost(action_name, indices, modifications, user_feedback)
        old_dataset = self.dataset.copy()
        old_model = copy.deepcopy(self.model)
        new_dataset = apply_operation(self.dataset, action_name, indices, modifications)
        decision_summary = self._decision_summary(
            old_dataset, new_dataset, action_name, indices, modifications, user_score
        )
        dirty_after = self._dirty_indices_after_operation(action_name, old_n, len(new_dataset.y_train), indices)
        action_onehot = self._action_onehot(action_id)
        proxy = self.proxy_reward_trainer
        modified_count = max(len(indices), abs(len(new_dataset.y_train) - old_n))
        accumulated_ratio = (self._modified_since_full + modified_count) / max(1, old_n)
        periodic_exact = self._steps_since_exact >= self.exact_eval_interval - 1
        force_exact = periodic_exact or accumulated_ratio >= self.full_retrain_modified_ratio
        use_proxy = (
            proxy is not None and proxy.is_active and action_name != "noop" and not force_exact
            and proxy.should_use_proxy(state, action_onehot, decision_summary)
        )

        diff_score = None
        if action_name == "noop":
            new_score = old_score
            exact_evaluation = False
        elif use_proxy:
            # Evolve the model using differential training. Without it, retain
            # semantic correctness by fitting the downstream model but skip exact
            # validation; users obtain the intended speedup when both mechanisms
            # are enabled, as in the paper.
            self._diff_fit(new_dataset, dirty_after, action_name=action_name)
            new_score = proxy.predict_post_performance(state, action_onehot, decision_summary)
            exact_evaluation = False
        else:
            # Exact evaluation doubles as the shared full-retraining checkpoint
            # for differential training and proxy supervision.
            full_model = getattr(self.model, "_base", self.model)
            full_model.fit(new_dataset.X_train, new_dataset.y_train)
            self.model = full_model
            if self._use_diff:
                self._diff_model = None
                self._diff_deferred = True
            new_score = float(self.model.score(new_dataset.X_val, new_dataset.y_val))
            exact_evaluation = True

        reward = compute_reward(
            old_score, new_score, action_name,
            processed_ratio=len(indices) / max(1, old_n),
            user_satisfaction=user_score,
            issues_resolved_ratio=eta,
            operational_cost=operational_cost,
        )
        reverted = False
        if action_name != "noop" and self.rollback_on_negative and new_score < old_score - self.rollback_tolerance:
            self.dataset, self.model = old_dataset, old_model
            self._validation_score_override = None
            reverted = True
        else:
            self.dataset = new_dataset
            self._validation_score_override = None if exact_evaluation else float(new_score)

        if action_name != "noop" and action_name in self.selectors:
            selector = self.selectors[action_name]
            if hasattr(selector, "update_with_reward"):
                selector.update_with_reward(reward)

        if exact_evaluation:
            self._steps_since_exact = 0
            self._modified_since_full = 0
            if proxy is not None:
                proxy.on_step_completed(
                    state=state, action_id=action_id, action_onehot=action_onehot,
                    decision_summary=decision_summary, exact_score=new_score, used_proxy=False,
                )
        else:
            self._steps_since_exact += 1
            if not reverted:
                self._modified_since_full += modified_count
            if proxy is not None:
                proxy.on_step_completed(
                    state=state, action_id=action_id, action_onehot=action_onehot,
                    decision_summary=decision_summary, exact_score=None, used_proxy=use_proxy,
                )

        self.state_encoder.add_action(action_id, reward=reward)
        next_state = self.get_state()
        return StepResult(
            state=state, next_state=next_state, action_id=action_id,
            action_name=action_name, reward=float(reward), old_score=float(old_score),
            new_score=float(new_score), n_processed=len(indices), indices=indices,
            modifications=modifications, user_score=float(user_score),
            selector_updated=selector_updated, reverted=reverted, feedback=user_feedback,
            used_proxy=bool(use_proxy), diff_score=diff_score, diff_prefiltered=False,
            issues_resolved_ratio=eta, operational_cost=operational_cost,
            exact_evaluation=exact_evaluation, decision_summary=decision_summary,
        )
