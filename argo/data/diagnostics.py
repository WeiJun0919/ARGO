from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score


@dataclass
class DataDiagnostics:
    label_noise_scores: np.ndarray
    outlier_scores: np.ndarray
    class_distribution: np.ndarray
    imbalance_score: float
    estimated_label_noise_ratio: float
    estimated_outlier_ratio: float


def _to_dense_safe(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim > 2:
        X = X.reshape(len(X), -1)
    return np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


def compute_diagnostics(X, y, model=None, random_state=42):
    n_classes = int(np.max(y) + 1)
    class_distribution = np.bincount(y, minlength=n_classes) / max(len(y), 1)
    imbalance_score = float(class_distribution.max() - class_distribution.min())

    Xd = _to_dense_safe(X)
    med = np.median(Xd, axis=0)
    mad = np.median(np.abs(Xd - med), axis=0) + 1e-8
    z = np.abs((Xd - med) / mad)
    raw = np.mean(np.minimum(z, 10.0), axis=1)
    outlier_scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

    if model is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = proba.max(axis=1)
        pred = proba.argmax(axis=1)
        label_noise_scores = np.where(pred != y, confidence, 1 - confidence)
    else:
        label_noise_scores = np.zeros(len(y), dtype=float)

    return DataDiagnostics(
        label_noise_scores=label_noise_scores,
        outlier_scores=outlier_scores,
        class_distribution=class_distribution,
        imbalance_score=imbalance_score,
        estimated_label_noise_ratio=float((label_noise_scores > 0.7).mean()),
        estimated_outlier_ratio=float((outlier_scores > 0.75).mean()),
    )


def validation_accuracy(model, X_val, y_val):
    return float(accuracy_score(y_val, model.predict(X_val)))
