from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import numpy as np


@dataclass
class HumanFeedback:
    accepted_indices: Sequence[int] = field(default_factory=list)
    rejected_indices: Sequence[int] = field(default_factory=list)
    corrected_labels: Optional[Dict[int, int]] = None
    action_acceptance: Optional[float] = None
    note: str = ""

    def normalized_action_score(self):
        if self.action_acceptance is not None:
            return float(np.clip(self.action_acceptance, -1.0, 1.0))
        total = len(self.accepted_indices) + len(self.rejected_indices)
        if total == 0:
            return 0.0
        return float((len(self.accepted_indices) - len(self.rejected_indices)) / total)

    def acceptance_rate(self):
        total = len(self.accepted_indices) + len(self.rejected_indices)
        return 0.0 if total == 0 else float(len(self.accepted_indices) / total)

    def has_instance_feedback(self):
        return bool(self.accepted_indices or self.rejected_indices or self.corrected_labels)


class FeedbackBuffer:
    def __init__(self):
        self.records = []

    def add(self, feedback):
        if feedback is not None:
            self.records.append(feedback)

    def summary(self):
        if not self.records:
            return {"n_records": 0, "mean_action_score": 0.0, "mean_acceptance_rate": 0.0}
        return {
            "n_records": len(self.records),
            "mean_action_score": float(np.mean([r.normalized_action_score() for r in self.records])),
            "mean_acceptance_rate": float(np.mean([r.acceptance_rate() for r in self.records])),
        }


def oracle_feedback_from_clean_labels(dataset, action_name, indices, modifications):
    """
    Backend simulated human feedback.

    For synthetic/noise-injection experiments, y_train_clean is available.
    It acts as a human oracle. In real usage, replace this with real user
    accept/reject/correction.
    """
    indices = list(map(int, indices))
    if action_name == "noop":
        return HumanFeedback(action_acceptance=-0.2)

    if dataset.y_train_clean is None or not indices:
        return HumanFeedback(action_acceptance=0.0)

    accepted, rejected, corrected = [], [], {}

    if action_name == "relabel":
        proposed = np.asarray(modifications.get("new_labels", []))
        for pos, i in enumerate(indices):
            clean = int(dataset.y_train_clean[i])
            pred = int(proposed[pos]) if pos < len(proposed) else int(dataset.y_train[i])
            if pred == clean:
                accepted.append(i)
            else:
                # Do not discard useful relabel opportunities. A real human
                # would correct the label here.
                accepted.append(i)
                corrected[i] = clean

        return HumanFeedback(
            accepted_indices=accepted,
            rejected_indices=rejected,
            corrected_labels=corrected or None,
            action_acceptance=0.8,
            note="oracle relabel correction"
        )

    if action_name == "remove":
        for i in indices:
            if int(dataset.y_train[i]) != int(dataset.y_train_clean[i]):
                accepted.append(i)
            else:
                rejected.append(i)
        return HumanFeedback(accepted, rejected, action_acceptance=0.1)

    if action_name == "augment":
        # Only augment clean samples. Reject noisy labels to prevent noise amplification.
        for i in indices:
            if int(dataset.y_train[i]) == int(dataset.y_train_clean[i]):
                accepted.append(i)
            else:
                rejected.append(i)
        return HumanFeedback(accepted, rejected, action_acceptance=0.05)

    if action_name == "repair":
        return HumanFeedback(action_acceptance=-0.1)

    return HumanFeedback()
