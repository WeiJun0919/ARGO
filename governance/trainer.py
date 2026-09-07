import copy
from pathlib import Path
import pandas as pd
from governance.feedback import FeedbackBuffer
from data.io_utils import save_dataset_artifact


class ARGOTrainer:
    """
    Trainer for ARGO framework.

    - Supports warmup selectors
    - Fixed-policy refinement
    - End-to-end joint training
    - Tracks best dataset and model
    - Automatically saves best cleaned training set to CSV
    - Differential training: after warm-up, activates selective fine-tuning
      instead of full downstream retrain on every step.
    """
    def __init__(self, env, agent, save_best_csv: str = "best_cleaned_train.csv", save_best_dataset: str = "outputs/best_cleaned_dataset.pkl", proxy_reward_trainer=None):
        self.env = env
        self.agent = agent
        self.proxy_reward_trainer = proxy_reward_trainer
        self.logs = []
        self.feedback_buffer = FeedbackBuffer()
        self.best_score = env.model.score(env.dataset.X_val, env.dataset.y_val)
        self.best_dataset = env.dataset.copy()
        self.best_model = copy.deepcopy(env.model)
        self.save_best_csv = Path(save_best_csv) if save_best_csv else None
        self.save_best_dataset = Path(save_best_dataset) if save_best_dataset else None

        if self.proxy_reward_trainer is not None and hasattr(env, "proxy_reward_trainer"):
            env.proxy_reward_trainer = self.proxy_reward_trainer

        self._save_best_artifacts(self.best_dataset, reason="initial")

    def warmup_selectors(self, clean_y_train=None, epochs=6):
        """
        Stage 1 warmup with shared ILDE/ED2 detection.

        The expensive detection pass is selector-independent, so we run it once
        and then let each selector train on the cached pseudo-pair batches.  This
        avoids repeating ED2, ILDE clean-sample detection, pseudo-pair building,
        and downstream model refits for remove/augment/relabel/repair.
        """
        warmed = []
        ilde_results = {}
        clean_indices = None

        warmup_items = [
            (name, sel)
            for name, sel in self.env.selectors.items()
            if hasattr(sel, "warmup") or hasattr(sel, "warmup_from_stage1_cache")
        ]

        stage1_cache = None
        used_shared_cache = False

        # Preferred fast path: run Stage-1 detection once, then reuse it for all
        # neural selectors that expose warmup_from_stage1_cache().
        if warmup_items and all(hasattr(sel, "warmup_from_stage1_cache") for _, sel in warmup_items):
            from argo_selectors.ide_module import run_stage1_ilde_detection

            stage1_cache = run_stage1_ilde_detection(
                dataset=self.env.dataset,
                model=self.env.model,
                clean_y_train=clean_y_train,
                epochs=epochs,
                verbose=False,
                modality=getattr(self.env.dataset, "modality", None),
            )
            used_shared_cache = True
            clean_indices = stage1_cache.get("_final_clean_indices")

            for name, sel in warmup_items:
                result = sel.warmup_from_stage1_cache(
                    stage1_cache,
                    self.env.dataset,
                    self.env.model,
                    epochs=epochs,
                    verbose=False,
                )
                warmed.append(name)
                if result is not None:
                    ilde_results[name] = self._summarize_ilde_result_for_log(result)

        else:
            # Backward-compatible fallback for custom selectors.
            for name, sel in warmup_items:
                if hasattr(sel, "warmup"):
                    result = sel.warmup(
                        self.env.dataset,
                        self.env.model,
                        clean_y_train=clean_y_train,
                        epochs=epochs,
                    )
                    warmed.append(name)
                    if result is None and hasattr(sel, "ilde_result"):
                        result = sel.ilde_result
                    if result is not None:
                        ilde_results[name] = self._summarize_ilde_result_for_log(result)
                    if clean_indices is None and hasattr(sel, "get_clean_indices"):
                        try:
                            clean_indices = sel.get_clean_indices()
                        except Exception:
                            pass

        # ── Activate differential training once warm-up provides clean indices ─
        if clean_indices is not None and self.env._use_diff:
            self.env.enable_differential(clean_indices)

        diff_summary = {
            "enabled": bool(getattr(self.env, "_use_diff", False)),
            "activated": bool(getattr(self.env, "_diff_model", None) is not None),
            "deferred": bool(getattr(self.env, "_diff_deferred", False)),
            "mode": getattr(self.env, "_diff_mode", None),
            "dirty_scope": getattr(self.env, "_diff_dirty_scope", None),
            "protected_clean_count": (
                int(len(clean_indices)) if clean_indices is not None else 0
            ),
        }

        return {
            "stage": "stage_1_ilde_warmup",
            "epochs": epochs,
            "shared_detection_cache": used_shared_cache,
            "stage1_detection_runs": 1 if used_shared_cache else len(warmed),
            "warmed_selectors": warmed,
            "ilde_results": ilde_results,
            "clean_indices_from_warmup": (
                int(len(clean_indices)) if clean_indices is not None else 0
            ),
            "differential_training": diff_summary,
        }

    def _summarize_ilde_result_for_log(self, result):
        """Return a compact Stage-1 summary safe to print in demos."""
        if not isinstance(result, dict):
            return result

        out = {}
        for key, value in result.items():
            if key.startswith("_"):
                continue
            if key == "ed2_results" and isinstance(value, dict):
                # Keep Stage-1 logs compact.  ED2 row/column arrays are useful
                # internally but too large to print in terminal summaries.
                out[key] = {
                    k: v
                    for k, v in value.items()
                    if k not in {
                        "row_noise_scores",
                        "column_error_proba",
                        "cell_error_matrix",
                        "error_matrix",
                    }
                }
            elif key == "selector_pair_batches" and isinstance(value, list):
                out[key] = value
            else:
                out[key] = value
        return out


    def _maybe_update_best_from_result(self, result):
        """Use the score already computed inside env.step.

        Avoid an extra downstream score pass on every step, and skip cases where
        the dataset/model did not actually change (proxy, noop, rejected/reverted).
        """
        if getattr(result, "used_proxy", False):
            return
        if getattr(result, "reverted", False):
            return
        if getattr(result, "action_name", "noop") == "noop":
            return
        try:
            new_score = float(result.new_score)
        except Exception:
            return
        if new_score > self.best_score:
            self.best_score = new_score
            self.best_dataset = self.env.dataset.copy()
            self.best_model = copy.deepcopy(self.env.model)
            self._save_best_artifacts(self.best_dataset, reason="score_improved")

    def _maybe_update_best(self):
        score = self.env.model.score(self.env.dataset.X_val, self.env.dataset.y_val)
        if score > self.best_score:
            self.best_score = score
            self.best_dataset = self.env.dataset.copy()
            self.best_model = copy.deepcopy(self.env.model)
            self._save_best_artifacts(self.best_dataset, reason="score_improved")


    def _save_best_artifacts(self, dataset, reason="best"):
        if self.save_best_dataset is not None:
            save_dataset_artifact(
                dataset,
                self.save_best_dataset,
                metadata={
                    "artifact_type": "best_cleaned_dataset",
                    "reason": reason,
                    "best_score": float(self.best_score),
                    "dataset_name": getattr(dataset, "name", "unknown"),
                    "modality": getattr(dataset, "modality", "unknown"),
                    "train_size": int(len(dataset.y_train)) if getattr(dataset, "y_train", None) is not None else 0,
                    "val_size": int(len(dataset.y_val)) if getattr(dataset, "y_val", None) is not None else 0,
                },
            )
            print(f"[BEST] saved dataset artifact: {self.save_best_dataset} | score={self.best_score:.4f}")
        if self.save_best_csv is not None:
            self._save_best_csv(dataset)

    def _save_best_csv(self, dataset):
        if self.save_best_csv is None or dataset.X_train is None or dataset.y_train is None:
            return

        self.save_best_csv.parent.mkdir(parents=True, exist_ok=True)

        # 1. 模型用版本：标准化 + one-hot 后的矩阵
        feature_names = dataset.feature_names
        if feature_names is None:
            n_features = dataset.X_train.shape[1] if len(dataset.X_train.shape) > 1 else 1
            feature_names = [str(i) for i in range(n_features)]

        X_for_csv = dataset.X_train
        if hasattr(X_for_csv, "shape") and len(X_for_csv.shape) > 2:
            X_for_csv = X_for_csv.reshape(len(X_for_csv), -1)
            feature_names = [str(i) for i in range(X_for_csv.shape[1])]

        df_model = pd.DataFrame(X_for_csv, columns=feature_names)
        df_model["label"] = dataset.y_train
        model_path = self.save_best_csv.with_name(self.save_best_csv.stem + "_model.csv")
        df_model.to_csv(model_path, index=False)

        # 2. 用户可读版本：原始 Adult 字段
        if getattr(dataset, "raw_train_df", None) is not None:
            df_readable = dataset.raw_train_df.copy()
            label_col = getattr(dataset, "label_col", "label")
            if label_col not in df_readable.columns:
                label_col = "label"
            df_readable[label_col] = dataset.y_train

            readable_path = self.save_best_csv.with_name(self.save_best_csv.stem + "_readable.csv")
            df_readable.to_csv(readable_path, index=False)
            print(f"[BEST] saved readable CSV: {readable_path}")
            print(f"[BEST] saved model CSV: {model_path}")
        else:
            df_model.to_csv(self.save_best_csv, index=False)
            print(f"[BEST] saved model CSV: {self.save_best_csv}")

    def restore_best(self):
        self.env.dataset = self.best_dataset.copy()
        self.env.model = copy.deepcopy(self.best_model)

    def refine_selectors_fixed_policy(self, steps_per_action=3, feedback_provider=None):
        """Stage II: strict round-robin selector refinement with PPO frozen."""
        logs = []
        self.agent.training_enabled = False
        selector_actions = [
            (i, name) for i, name in enumerate(self.agent.action_names)
            if name != "noop" and name in self.env.selectors
        ]
        for _ in range(int(steps_per_action)):
            for action_id, action_name in selector_actions:
                result = self.env.step(action_id, action_name, feedback_provider=feedback_provider)
                self.feedback_buffer.add(result.feedback)
                self.logs.append(result)
                logs.append(result)
                self._maybe_update_best_from_result(result)
        self.agent.training_enabled = True
        return logs

    def joint_train(self, total_steps=30, feedback_provider=None):
        """Stage III: jointly update the PPO/gate and invoked selectors."""
        self.agent.training_enabled = True
        if hasattr(self.env.state_encoder, "max_steps"):
            self.env.state_encoder.max_steps = max(1, int(total_steps))
        if self.proxy_reward_trainer is not None:
            self.proxy_reward_trainer.activate()
            self.proxy_reward_trainer.maybe_recalibrate(force=True)

        logs = []
        n_proxy, n_real = 0, 0
        for step in range(total_steps):
            state = self.env.get_state()
            action_id = self.agent.select_action(state)
            action_name = self.agent.action_name(action_id)
            result = self.env.step(
                action_id,
                action_name,
                feedback_provider=feedback_provider,
                precomputed_state=state,
            )
            self.agent.record_transition(
                result.state, action_id, result.reward, result.next_state,
                done=(step == total_steps - 1),
            )
            self.feedback_buffer.add(result.feedback)
            self.logs.append(result)
            logs.append(result)
            self._maybe_update_best_from_result(result)

            if self.proxy_reward_trainer is not None and not result.used_proxy:
                self.proxy_reward_trainer.maybe_recalibrate(force=False)

            if result.used_proxy:
                n_proxy += 1
            elif getattr(result, "exact_evaluation", True):
                n_real += 1

        self.agent.update_policy(force=True)

        total = n_proxy + n_real
        print(
            f"[STAGE 3] done | proxy={n_proxy} ({100 * n_proxy / max(1, total):.0f}%) "
            f"| real={n_real} ({100 * n_real / max(1, total):.0f}%)"
        )
        if self.proxy_reward_trainer is not None and hasattr(self.proxy_reward_trainer, "print_timing_summary"):
            self.proxy_reward_trainer.print_timing_summary()
        return logs

    def summarize_logs(self, logs):
        counts, reverted = {}, 0
        for log in logs:
            counts[log.action_name] = counts.get(log.action_name, 0) + 1
            reverted += int(log.reverted)
        return {
            "steps": len(logs),
            "action_counts": counts,
            "reverted_steps": reverted,
            "proxy_used_steps": sum(int(getattr(log, "used_proxy", False)) for log in logs),
            "real_eval_steps": sum(int(getattr(log, "exact_evaluation", True)) for log in logs),
            "diff_prefiltered_steps": sum(int(getattr(log, "diff_prefiltered", False)) for log in logs),
            "current_score": self.env.model.score(self.env.dataset.X_val, self.env.dataset.y_val),
            "best_score": self.best_score,
        }

    def summary(self):
        out = self.summarize_logs(self.logs)
        out["feedback"] = self.feedback_buffer.summary()
        out["selector_stats"] = {
            k: v.stats()
            for k, v in self.env.selectors.items()
            if hasattr(v, "stats")
        }
        if self.proxy_reward_trainer is not None:
            out["proxy_reward"] = self.proxy_reward_trainer.summary()
        if self.agent.model is not None:
            state = self.env.get_state()
            out["ppo"] = {
                "updates": int(self.agent.update_count),
                "last_losses": dict(self.agent.last_losses),
                "action_probabilities": self.agent.action_probabilities(state),
                "gating_weights": self.agent.gating_weights(state),
            }
        return out
