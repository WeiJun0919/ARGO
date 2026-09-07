import numpy as np
from argo_selectors.base_selector import BaseSelector


class AugmentSelector(BaseSelector):
    def __init__(self, noise_scale=0.03, random_state=42):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(random_state)

    def select(self, dataset, model, budget: int):
        counts = np.bincount(dataset.y_train, minlength=dataset.n_classes)
        minority_classes = np.argsort(counts)
        selected = []

        for c in minority_classes:
            class_idx = np.where(dataset.y_train == c)[0]
            if len(class_idx) == 0:
                continue
            need = min(budget - len(selected), max(1, budget // dataset.n_classes))
            selected.extend(self.rng.choice(class_idx, size=need, replace=True).tolist())
            if len(selected) >= budget:
                break

        idx = np.array(selected[:budget], dtype=int)
        noise = self.rng.normal(0, self.noise_scale, size=dataset.X_train[idx].shape)
        return idx, {"augmented_X": dataset.X_train[idx] + noise, "augmented_y": dataset.y_train[idx]}
