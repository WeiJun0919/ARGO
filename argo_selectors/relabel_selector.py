import numpy as np
from data.diagnostics import compute_diagnostics
from argo_selectors.base_selector import BaseSelector


class RelabelSelector(BaseSelector):
    def select(self, dataset, model, budget: int):
        proba = model.predict_proba(dataset.X_train)
        pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)

        diag = compute_diagnostics(dataset.X_train, dataset.y_train, model)
        losses = model.losses(dataset.X_train, dataset.y_train)
        loss_score = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)

        # Ensemble disagreement: independent of current model's biases
        if hasattr(model, "predict_proba_ensemble"):
            _, disagreement = model.predict_proba_ensemble(dataset.X_train, n_ensemble=5)
        else:
            disagreement = np.zeros(len(dataset.y_train))

        # Score components (higher = more likely noisy):
        #   conf_disagreement: model confident but wrong -> likely label error
        #   loss: high-loss samples are often mislabeled
        #   ensemble_disagree: multiple models disagree with the label
        score = (
            0.35 * np.where(pred != dataset.y_train, conf, 0.0)
            + 0.30 * loss_score
            + 0.35 * disagreement
        )

        idx = np.argsort(score)[-budget:]
        new_labels = pred[idx]
        return idx, {"new_labels": new_labels}
