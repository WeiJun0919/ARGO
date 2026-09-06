import numpy as np
from argo.data.diagnostics import compute_diagnostics
from argo.selectors.base_selector import BaseSelector


class RepairSelector(BaseSelector):
    def select(self, dataset, model, budget: int):
        diag = compute_diagnostics(dataset.X_train, dataset.y_train, model)
        idx = np.argsort(diag.outlier_scores)[-budget:]

        X = dataset.X_train.copy()
        med = np.median(X, axis=0)
        repaired = X[idx].copy()

        # 简化版特征修复：把极端 z-score 特征替换为列中位数
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        z = np.abs((repaired - mean) / std)
        repaired[z > 3.0] = np.take(med, np.where(z > 3.0)[1])

        return idx, {"repaired_X": repaired}
