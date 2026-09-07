import numpy as np
from data.diagnostics import compute_diagnostics
from argo_selectors.base_selector import BaseSelector


class RemoveSelector(BaseSelector):
    def select(self, dataset, model, budget: int):
        diag = compute_diagnostics(dataset.X_train, dataset.y_train, model)
        losses = model.losses(dataset.X_train, dataset.y_train)
        loss_score = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
        score = 0.6 * diag.label_noise_scores + 0.4 * loss_score
        idx = np.argsort(score)[-budget:]
        return idx, {}
