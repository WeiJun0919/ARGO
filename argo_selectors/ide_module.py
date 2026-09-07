"""
Stage 1: Noise-Aware Bootstrapping — ILDE + ED2.

This module implements Stage 1 of the ARGO training curriculum.
It combines ILDE (label noise) with ED2 (tabular attribute noise):

Modality-aware strategy:
  - Image / Text data  →  ILDE only (label noise detection)
  - Tabular data       →  ILDE + ED2 (label noise + attribute noise detection)

ILDE (Liao et al., IJCV 2024):
  Estimates instance-dependent noise transition matrices via inter-class
  correlation (Gram matrix), without requiring anchor points.
    G_{ij} = (1/N_b) Σ_k y'_ki · y'_kj
    T ≈ diag(G)^{-1} · G
  Temporal ensembling: LD_ensemble^{(t)} = β · LD_ensemble^{(t-1)} + (1-β)·LD_sharpened

ED2 (Eister et al., SIGMOD 2019):
  Example-driven error detection for tabular data at the cell level.
  Multi-column features: metadata (ρ_meta), text n-grams (ρ_text),
  cell value correlation (ρ_corr), error correlation (ρ_error).
  One RandomForest classifier per column, Min Certainty active learning.
  In ARGO Stage 1, used unsupervised via statistical pseudo-labels.

Synthetic Noise Injection (§4.1.4):
  High-confidence clean samples → inject label/feature/outlier noise
  → pseudo-label pairs {(x, y_synth), (x, y_clean)}
  → selector binary classification warmup

References:
  - Liao et al., "ILDE", IJCV 2024 / arXiv:2212.08380
  - Eister et al., "ED2", SIGMOD 2019
  - ARGO thesis description §4.1
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PseudoLabelPair:
    """One pseudo training pair: (x, y_clean, y_synthetic, noise_type, severity)."""
    x: np.ndarray
    y_clean: int
    y_synthetic: int
    noise_type: str          # 'label_flip', 'feature_corrupt', 'outlier'
    severity: float          # noise strength (0-1 scale)
    x_clean: Optional[np.ndarray] = None  # correction target for feature repair


from argo_selectors.ed2_module import (
    run_ed2_error_detection,
    ED2State,
)

@dataclass
class ILDEState:
    ld_ensemble: Optional[np.ndarray] = None   # shape: (n_train, n_classes)
    epoch: int = 0
    gram_accum: Optional[np.ndarray] = None  # shape: (n_classes, n_classes)
    gram_count: int = 0
    pair_losses: List[float] = field(default_factory=list)
    ed2_state: Optional[ED2State] = None


# ---------------------------------------------------------------------------
# ILDE core methods
# ---------------------------------------------------------------------------

def estimate_noise_transition_matrix(
    soft_preds: np.ndarray,
    n_classes: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Estimate noise transition matrix via inter-class correlation (Gram matrix).

    Key insight (ILDE): mislabeling probability is correlated with inter-class similarity.
    The Gram matrix G encodes co-occurrence of class predictions across the mini-batch.
    The transition matrix T is approximated as: T = diag(G)^{-1} · G

    This naturally gives instance-dependent (batch-level) estimates because different
    mini-batches contain different class subsets, changing the confusion structure.

    Args:
        soft_preds: shape (batch_size, n_classes), probability predictions
        n_classes: number of classes
        eps: numerical stability constant

    Returns:
        T: shape (n_classes, n_classes), estimated noise transition matrix.
           T[i, j] ≈ Pr(label corrupted from i to j)
    """
    soft_preds = np.asarray(soft_preds)
    if soft_preds.ndim == 1:
        soft_preds = soft_preds.reshape(1, -1)

    # Gram matrix: inter-class co-occurrence
    G = (soft_preds.T @ soft_preds) / max(soft_preds.shape[0], 1)

    # Row-normalize to get transition probabilities: T[i,j] = G[i,j] / G[i,:]
    row_sums = G.sum(axis=1, keepdims=True) + eps
    T = G / row_sums

    return T


def compute_instance_dependent_ld(
    noisy_posterior: np.ndarray,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute instance-dependent label distribution (LD) from noisy posterior.

    Args:
        noisy_posterior: shape (batch_size, n_classes), model prediction
        transition_matrix: shape (n_classes, n_classes), noise transition matrix

    Returns:
        LD: shape (batch_size, n_classes), instance-dependent label distribution
    """
    # LD = T^T @ noisy_posterior^T (each row is a label distribution)
    # Using: LD_j = Σ_i T_{i,j} · noisy_posterior_i
    LD = noisy_posterior @ transition_matrix.T
    # Renormalize to probability simplex
    LD = np.maximum(LD, 0)
    row_sums = LD.sum(axis=1, keepdims=True)
    LD = LD / (row_sums + 1e-8)
    return LD


def sharpen_ld(ld: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """
    Sharpen label distribution to reduce noise impact.

    Lower temperature → sharper distribution → less noise influence.
    This is the sharpening step from ILDE Eq.(8).
    """
    if temperature <= 0:
        return ld
    log_ld = np.log(ld + 1e-12)
    sharpened = np.exp(log_ld / temperature)
    sharpened = np.maximum(sharpened, 0)
    row_sums = sharpened.sum(axis=1, keepdims=True)
    sharpened = sharpened / (row_sums + 1e-8)
    return sharpened


def temporal_ensemble_ld(
    new_ld: np.ndarray,
    ensemble_ld: Optional[np.ndarray],
    beta: float = 0.6,
) -> np.ndarray:
    """
    Update temporal ensemble of label distributions.

    From ILDE Eq.(9):
      LD_ensemble^{(t)} = β · LD_ensemble^{(t-1)} + (1-β) · LD_sharpened^{(t)}

    This smooths the LD estimate across epochs, stabilizing supervision.
    """
    if ensemble_ld is None:
        return new_ld.copy()
    return beta * ensemble_ld + (1 - beta) * new_ld


def ilde_loss(
    logits: np.ndarray,
    noisy_labels: np.ndarray,
    ensemble_ld: np.ndarray,
    alpha: float = 2.0,
    beta_ilde: float = 0.6,
) -> Tuple[float, np.ndarray]:
    """
    Compute ILDE combined loss.

    Args:
        logits: model logits (batch_size, n_classes)
        noisy_labels: one-hot noisy labels (batch_size, n_classes)
        ensemble_ld: temporally-ensembled label distribution (batch_size, n_classes)
        alpha: weight of LD supervision vs noisy label supervision
        beta_ilde: temporal ensembling coefficient (shared with temporal_ensemble_ld)

    Returns:
        total_loss: scalar combined loss
        ld: the computed label distribution (before sharpening, for temporal update)
    """
    eps = 1e-8

    # Softmax probabilities
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + eps)

    # Noisy label cross-entropy: L_NL = -Σ y*_i · log(p_i)
    log_proba = np.log(proba + eps)
    L_nl = -np.sum(noisy_labels * log_proba, axis=1).mean()

    # LD supervision: L_LD = -Σ p_i · log(LD_i)
    log_ld = np.log(ensemble_ld + eps)
    L_ld = -np.sum(proba * log_ld, axis=1).mean()

    total_loss = L_nl + alpha * L_ld
    return float(total_loss), proba


# ---------------------------------------------------------------------------
# Synthetic noise injection & pseudo-label pair construction
# ---------------------------------------------------------------------------

def identify_high_confidence_clean_samples(
    X: np.ndarray,
    y: np.ndarray,
    model,
    confidence_threshold: float = 0.85,
    top_k_ratio: float = 0.30,
    use_clean_reference: bool = False,
    dataset_clean_labels: Optional[np.ndarray] = None,
    ed2_results: Optional[Dict] = None,
) -> np.ndarray:
    """
    Identify a high-confidence clean subset using multiple noise signals.

    Uses the three-signal approach from ARGO §4.1.2:
      1. Confidence score (model prediction agreement with label)
      2. Per-sample loss (cross-entropy on true label)
      3. Ensemble disagreement (consistency across multiple forward passes)

    When a clean reference is available (simulation), we use it directly for ground truth.
    Otherwise we fall back to the three-signal heuristic.

    Args:
        X: training features
        y: training labels (noisy)
        model: downstream model with predict_proba, losses, predict_proba_ensemble
        confidence_threshold: minimum confidence for a sample to be considered clean
        top_k_ratio: fraction of highest-confidence samples to return
        use_clean_reference: if True and dataset_clean_labels provided, use it directly
        dataset_clean_labels: ground-truth clean labels (simulation only)
        ed2_results: ED2 attribute noise results (tabular only). Samples with high
                      attribute noise scores are DOWN-weighted from clean set.

    Returns:
        clean_indices: indices into X of high-confidence clean samples
    """
    n = len(y)

    # Signal 1: confidence score
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    # For samples where model agrees with label → high confidence
    # For samples where model disagrees → low confidence (potentially noisy)
    conf_signal = np.where(pred == y, confidence, 1 - confidence)

    # Signal 2: per-sample loss (lower loss = more likely clean)
    try:
        losses = model.losses(X, y)
        loss_normalized = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
        loss_signal = 1.0 - loss_normalized
    except Exception:
        loss_signal = np.ones(n) * 0.5

    # Signal 3: ensemble disagreement (lower disagreement = more likely clean)
    try:
        if hasattr(model, "predict_proba_ensemble"):
            _, disagreement = model.predict_proba_ensemble(X, n_ensemble=5)
        else:
            disagreement = np.zeros(n)
    except Exception:
        disagreement = np.zeros(n)
    disagreement_signal = 1.0 - disagreement

    # Signal 4: ED2 attribute noise score (tabular only; 0 for image/text)
    # High ED2 score → likely attribute noise → NOT a clean sample
    if ed2_results is not None:
        ed2_scores = ed2_results.get("row_noise_scores", np.zeros(n))
        # Invert: 1 - score → 1.0 means no attribute noise (more likely clean)
        ed2_signal = np.clip(1.0 - ed2_scores, 0.0, 1.0)
    else:
        ed2_signal = np.ones(n)

    # Combined score: weighted average of available signals
    if ed2_results is not None:
        # Tabular: ILDE label signals (65%) + ED2 attribute signal (35%)
        combined_score = (
            0.65 * (0.35 * conf_signal + 0.35 * loss_signal + 0.30 * disagreement_signal)
            + 0.35 * ed2_signal
        )
    else:
        # Image/text: ILDE label signals only
        combined_score = (
            0.35 * conf_signal +
            0.35 * loss_signal +
            0.30 * disagreement_signal
        )

    if use_clean_reference and dataset_clean_labels is not None:
        # Ground-truth verification: count correct predictions
        gt_correct = (y == dataset_clean_labels).astype(float)
        combined_score = combined_score * 0.5 + gt_correct * 0.5

    # Select top_k_ratio samples with highest combined score
    k = max(1, int(n * top_k_ratio))
    top_indices = np.argsort(combined_score)[-k:]

    # Additional filter: confidence must exceed threshold
    filtered = top_indices[conf_signal[top_indices] >= confidence_threshold]
    if len(filtered) < max(1, int(n * 0.05)):
        # Fall back to top-k if filter is too aggressive
        filtered = top_indices

    return filtered


def inject_synthetic_label_noise(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    noise_ratio: float = 0.25,
    n_classes: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Inject synthetic label noise into selected clean samples.

    Implements §4.1.4: "标签噪声注入" — flip labels to different classes.
    """
    rng = np.random.default_rng(random_state)
    y_out = y.copy()
    n_classes = n_classes or int(np.max(y) + 1)

    n_inject = max(1, int(len(indices) * noise_ratio))
    inject_idx = indices[rng.choice(len(indices), size=min(n_inject, len(indices)), replace=False)]

    flipped_from, flipped_to = [], []
    for i in inject_idx:
        old_label = int(y_out[i])
        candidates = [c for c in range(n_classes) if c != old_label]
        if candidates:
            new_label = rng.choice(candidates)
            y_out[i] = new_label
            flipped_from.append(old_label)
            flipped_to.append(new_label)

    return y_out, np.array(inject_idx), flipped_from


def inject_synthetic_feature_noise(
    X: np.ndarray,
    indices: np.ndarray,
    noise_ratio: float = 0.03,
    outlier_ratio: float = 0.03,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[int]]:
    """
    Inject synthetic feature noise into selected clean samples.

    Implements §4.1.4: "特征噪声注入" — for tabular: add Gaussian noise;
    for images: patch corruption.

    Args:
        X: feature matrix, shape (n, ...)
        indices: indices of samples to inject noise into
        noise_ratio: fraction of features to corrupt
        outlier_ratio: fraction of samples to make into outliers
        random_state: RNG seed

    Returns:
        X_out: corrupted feature matrix (copy)
        outlier_indices: which samples became outliers
    """
    rng = np.random.default_rng(random_state)
    X_out = X.copy() if isinstance(X, np.ndarray) else np.array(X, copy=True)
    X_out = np.asarray(X_out, dtype=float)

    if X_out.ndim > 2:
        # Image: patch corruption (per argo/data/noise.py pattern)
        outlier_rows = _inject_image_noise(X_out, indices, rng, noise_ratio, outlier_ratio)
    else:
        # Tabular: Gaussian feature noise + outlier injection
        outlier_rows = _inject_tabular_noise(X_out, indices, rng, noise_ratio, outlier_ratio)

    return X_out, outlier_rows


def _inject_tabular_noise(
    X: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    noise_ratio: float,
    outlier_ratio: float,
) -> List[int]:
    n, d = X.shape
    std = np.std(X, axis=0) + 1e-8
    outlier_rows = []

    # Feature-level Gaussian noise
    if noise_ratio > 0 and len(indices) > 0:
        mask = rng.random((len(indices), d)) < noise_ratio
        noise = rng.normal(0, 1.0, size=mask.shape)
        X[indices] = X[indices] + noise * std[np.newaxis, :] * mask

    # Outlier injection: replace some rows with extreme values
    n_out = max(1, int(len(indices) * outlier_ratio))
    out_idx = indices[rng.choice(len(indices), size=min(n_out, len(indices)), replace=False)]
    for r in out_idx:
        outlier_rows.append(int(r))
        n_cols = max(1, min(d, d // 12))
        cols = rng.choice(d, size=n_cols, replace=False)
        X[r, cols] += rng.normal(0, 6.0, size=len(cols)) * std[cols]

    return outlier_rows


def _inject_image_noise(
    X: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    noise_ratio: float,
    outlier_ratio: float,
) -> List[int]:
    """
    Image patch corruption: preserves [0,1] pixel range.
    """
    X_out = X.copy() if isinstance(X, np.ndarray) else np.array(X)
    original_shape = X_out.shape

    # Ensure NCHW format for 32x32 CIFAR
    if X_out.ndim == 2 and X_out.shape[1] == 3072:
        X_out = X_out.reshape(-1, 3, 32, 32)
    elif X_out.ndim == 4 and X_out.shape[-1] == 3:
        X_out = X_out.transpose(0, 3, 1, 2)

    n = len(X_out)
    outlier_rows = []

    # Pixel-level Gaussian noise
    if noise_ratio > 0:
        mask = rng.random(X_out.shape) < noise_ratio
        X_out[mask] += rng.normal(0.0, 0.08, size=int(mask.sum())).astype(np.float32)

    # Patch corruption (outliers)
    n_out = max(1, int(len(indices) * outlier_ratio))
    out_idx = indices[rng.choice(len(indices), size=min(n_out, len(indices)), replace=False)]
    for r in out_idx:
        outlier_rows.append(int(r))
        h = int(rng.integers(4, 9))
        w = int(rng.integers(4, 9))
        top = int(rng.integers(0, 32 - h + 1))
        left = int(rng.integers(0, 32 - w + 1))
        mode = int(rng.integers(0, 3))
        if mode == 0:
            patch = rng.random((3, h, w), dtype=np.float32)
        elif mode == 1:
            patch = np.zeros((3, h, w), dtype=np.float32)
        else:
            patch = np.ones((3, h, w), dtype=np.float32)
        X_out[r, :, top:top + h, left:left + w] = patch

    X_out = np.clip(X_out, 0.0, 1.0)

    # Restore original layout
    if original_shape != X_out.shape:
        X_out = X_out.reshape(original_shape)

    return outlier_rows


def build_pseudo_label_pairs(
    X: np.ndarray,
    y: np.ndarray,
    clean_indices: np.ndarray,
    label_flip_ratio: float = 0.40,
    feature_corrupt_ratio: float = 0.30,
    outlier_ratio: float = 0.30,
    n_classes: Optional[int] = None,
    random_state: int = 42,
) -> List[PseudoLabelPair]:
    """
    Build pseudo-label pairs from high-confidence clean samples.

    Implements §4.1.4: "伪标签对构建"
      {(x_i, y_i^{noisy}), (x_i, y_i^{clean})}

    Three noise types are injected with controlled proportions:
      - label_flip: randomly flip label to a different class
      - feature_corrupt: add Gaussian/patch noise to features
      - outlier: replace some features with extreme values

    Each pair gives the selector a binary classification target:
      which version is the correct (clean) label?

    Args:
        X: training features
        y: training labels
        clean_indices: indices of high-confidence clean samples
        label_flip_ratio: fraction of pairs using label flip noise
        feature_corrupt_ratio: fraction using feature corruption
        outlier_ratio: fraction using outlier injection
        n_classes: number of classes
        random_state: RNG seed

    Returns:
        pairs: list of PseudoLabelPair objects
    """
    rng = np.random.default_rng(random_state)
    n_classes = n_classes or int(np.max(y) + 1)
    pairs: List[PseudoLabelPair] = []

    n_clean = len(clean_indices)
    if n_clean == 0:
        return pairs

    # Work on a local copy.  Stage-1 detection results may be cached and
    # shared by multiple selectors, so this function must not mutate the
    # caller's clean_indices array in-place.
    clean_indices = np.asarray(clean_indices, dtype=int).copy()

    # Shuffle and split into three noise-type groups
    rng.shuffle(clean_indices)
    n_total = n_clean
    n_label = max(1, int(n_total * label_flip_ratio))
    n_feature = max(1, int(n_total * feature_corrupt_ratio))
    n_outlier = max(1, n_total - n_label - n_feature)

    indices_label = clean_indices[:n_label]
    indices_feature = clean_indices[n_label:n_label + n_feature]
    indices_outlier = clean_indices[n_label + n_feature:]

    # --- Type 1: Label flip ---
    y_flipped, flipped_idx, _ = inject_synthetic_label_noise(
        X, y, indices_label,
        noise_ratio=0.9,   # High injection rate since these are clean samples
        n_classes=n_classes,
        random_state=random_state,
    )
    for idx in flipped_idx:
        pairs.append(PseudoLabelPair(
            x=X[idx].copy(),
            y_clean=int(y[idx]),
            y_synthetic=int(y_flipped[idx]),
            noise_type="label_flip",
            severity=1.0,
            x_clean=X[idx].copy(),
        ))

    # --- Type 2: Feature corruption ---
    X_corrupted, _ = inject_synthetic_feature_noise(
        X, indices_feature,
        noise_ratio=0.03,
        outlier_ratio=0.0,  # Feature corruption only, no outliers here
        random_state=random_state + 1,
    )
    for idx in indices_feature:
        pairs.append(PseudoLabelPair(
            x=X_corrupted[idx].copy(),
            y_clean=int(y[idx]),
            y_synthetic=int(y[idx]),  # Label unchanged; feature is corrupted
            noise_type="feature_corrupt",
            severity=0.5,
            x_clean=X[idx].copy(),
        ))

    # --- Type 3: Outlier injection ---
    X_outlier, outlier_rows = inject_synthetic_feature_noise(
        X, indices_outlier,
        noise_ratio=0.0,  # No mild noise; only extreme outlier
        outlier_ratio=0.03,
        random_state=random_state + 2,
    )
    for idx in indices_outlier:
        pairs.append(PseudoLabelPair(
            x=X_outlier[idx].copy(),
            y_clean=int(y[idx]),
            y_synthetic=int(y[idx]),  # Label unchanged; outlier is in features
            noise_type="outlier",
            severity=1.0,
            x_clean=X[idx].copy(),
        ))

    return pairs


# ---------------------------------------------------------------------------
# Selector warmup via pseudo-label pairs (supervised binary classification)
# ---------------------------------------------------------------------------

def train_selector_on_pairs(
    selector,
    pairs: List[PseudoLabelPair],
    X_all: np.ndarray,
    y_all: np.ndarray,
    model,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 0.001,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Supervised warmup of a selector using pseudo-label pairs.

    Implements §4.1.4: "选择器预热" — train the selector to distinguish
    clean from corrupted instances using the constructed pairs.

    Training signal per pair:
      - The selector receives (x, y_synthetic) and must predict the clean label y_clean.
      - A binary "correction needed" signal is derived:
          correction_needed = 1 if y_synthetic != y_clean else 0
      - Binary CE loss between correction_needed and selector's internal prediction.

    The selector's internal neural network (if any) is updated via gradient steps
    on this binary classification loss, giving it a discriminative prior about
    what different noise types look like.

    Args:
        selector: the selector to warm up (must have `update_with_supervised` method
                  or accept loss updates)
        pairs: pseudo-label pairs
        X_all: full training feature matrix
        y_all: full training labels
        model: downstream model (for computing noise signals during training)
        epochs: number of warmup epochs
        batch_size: training batch size
        lr: learning rate for any internal optimizer
        verbose: print progress

    Returns:
        stats: warmup statistics (loss curve, accuracy on pairs)
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {"n_pairs": 0, "mean_loss": 0.0, "final_pair_acc": 0.0}

    # Build pair features and labels
    X_pairs = np.array([p.x.flatten() if p.x.ndim > 1 else p.x for p in pairs])
    y_clean_arr = np.array([p.y_clean for p in pairs])
    y_synth_arr = np.array([p.y_synthetic for p in pairs])
    noise_types = np.array([p.noise_type for p in pairs])
    severities = np.array([p.severity for p in pairs])

    # Binary correction target: 1 = labels differ (selector should correct), 0 = same
    correction_target = (y_synth_arr != y_clean_arr).astype(float)

    # Score: how much the selector should "pay attention" to this pair
    # Higher severity = more weight
    pair_weights = severities / (severities.sum() + 1e-8)

    losses, accuracies = [], []

    if hasattr(selector, "warmup_with_pairs"):
        # Delegate to selector's own implementation (preferred path)
        result = selector.warmup_with_pairs(
            pairs=pairs,
            X_all=X_all,
            y_all=y_all,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )
        return result

    # Fallback: simple scoring-based warmup (no internal gradient updates)
    # The selector scores samples; we compare its outputs to ground truth
    budget = max(1, int(len(y_all) * 0.05))

    for epoch in range(epochs):
        epoch_losses = []
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            batch_idx = np.arange(start, end)

            # Compute noise signal for this batch (simplified scoring)
            batch_X = X_pairs[batch_idx]
            batch_y = y_synth_arr[batch_idx]
            batch_target = correction_target[batch_idx]

            try:
                proba = model.predict_proba(batch_X)
                pred = proba.argmax(axis=1)
                conf = proba.max(axis=1)

                # Score: how suspicious is this sample?
                score = np.where(pred != batch_y, conf, 1 - conf)
                # Map score to binary prediction (threshold = 0.5)
                pred_correction = (score < 0.5).astype(float)

                # Weighted binary cross-entropy
                eps = 1e-8
                bce = -(
                    batch_target * np.log(pred_correction + eps) +
                    (1 - batch_target) * np.log(1 - pred_correction + eps)
                )
                loss = float(np.average(bce, weights=pair_weights[batch_idx] + 1e-8))
                acc = float((pred_correction == batch_target).mean())
            except Exception:
                loss, acc = 1.0, 0.5

            epoch_losses.append(loss)
            accuracies.append(acc)

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        if verbose and (epoch + 1) % 2 == 0:
            print(f"  [warmup epoch {epoch+1}/{epochs}] loss={mean_loss:.4f} acc={np.mean(accuracies[-len(epoch_losses):]):.3f}")

    return {
        "n_pairs": n_pairs,
        "mean_loss": float(np.mean(losses[-3:])) if len(losses) >= 3 else float(np.mean(losses)),
        "final_pair_acc": float(np.mean(accuracies[-max(1, len(accuracies)//4):])) if accuracies else 0.0,
        "pair_noise_type_counts": {
            "label_flip": int((noise_types == "label_flip").sum()),
            "feature_corrupt": int((noise_types == "feature_corrupt").sum()),
            "outlier": int((noise_types == "outlier").sum()),
        },
    }


# ---------------------------------------------------------------------------
# Full Stage 1 pipeline
# ---------------------------------------------------------------------------

def _stage1_public_view(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the lightweight, externally visible part of a Stage-1 result.

    The detection cache intentionally stores pseudo-pair objects and raw ED2
    column probabilities under private keys.  Those objects are useful for
    selector warmup but too large/noisy for logging and should not leak into
    ARGOTrainer's printed summary.
    """
    hidden = {
        "_pseudo_pair_batches",
        "_ed2_raw",
        "_ilde_state",
        "_selector_pair_results",
    }
    return {k: v for k, v in result.items() if k not in hidden}


def _build_ed2_state(ed2_results: Optional[Dict[str, Any]]) -> Optional[ED2State]:
    if ed2_results is None:
        return None
    return ED2State(
        row_noise_scores=ed2_results["row_noise_scores"],
        column_results=ed2_results.get("column_error_proba", {}),
        n_rows=ed2_results["n_rows"],
        n_cols=ed2_results["n_cols"],
    )


def _summarize_ed2_results(ed2_results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a print-safe ED2 summary without row/column matrices.

    The full ED2 output contains row-level scores and per-column probability
    arrays.  Those are still kept privately in ``_ed2_raw`` and ``ILDEState``
    for downstream logic, but they should not be placed in the public Stage-1
    result because ``examples/run_tabular_demo.py`` prints that result directly.
    """
    if ed2_results is None:
        return None
    scores = np.asarray(ed2_results["row_noise_scores"], dtype=float)
    return {
        "n_rows": int(ed2_results["n_rows"]),
        "n_cols": int(ed2_results["n_cols"]),
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "p90": float(np.percentile(scores, 90)),
    }


def run_stage1_ilde_detection(
    dataset,
    model,
    clean_y_train: Optional[np.ndarray] = None,
    epochs: int = 6,
    confidence_threshold: float = 0.85,
    top_k_ratio: float = 0.30,
    pair_label_flip_ratio: float = 0.40,
    pair_feature_corrupt_ratio: float = 0.30,
    pair_outlier_ratio: float = 0.30,
    ilde_alpha: float = 2.0,
    ilde_beta: float = 0.6,
    ilde_temperature: float = 0.1,
    batch_size: int = 64,
    verbose: bool = False,
    random_state: int = 42,
    modality: Optional[str] = None,
    ed2_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Run Stage-1 ILDE/ED2 detection exactly once and cache reusable artifacts.

    This function performs the expensive, selector-independent work:
      - ED2 attribute-noise detection for tabular data
      - downstream model refits used by ILDE
      - high-confidence clean sample identification
      - pseudo-pair construction for each ILDE pair-building epoch

    It deliberately does NOT warm up a selector.  The returned cache can be
    consumed by any number of selectors via warmup_selector_from_stage1_cache().
    """
    # --- Modality detection ---
    modality = modality if modality else str(getattr(dataset, "modality", "tabular")).lower()
    if modality not in ("tabular", "image", "text", "adult", "synthetic", "imdb", "cifar10"):
        modality = "tabular"

    is_tabular = modality in ("tabular", "adult", "synthetic", "csv")

    X_train = np.asarray(dataset.X_train)
    y_train = np.asarray(dataset.y_train)
    n_classes = int(dataset.n_classes)

    # The paper's warm-up is self-supervised: detect a high-confidence clean
    # subset, then inject controlled corruptions.  Synthetic ground truth may
    # exist on Dataset for evaluation, but must not silently leak into Stage I.
    use_clean_ref = clean_y_train is not None and len(clean_y_train) > 0
    ref_labels = clean_y_train if use_clean_ref else None

    if verbose:
        print(
            f"[Stage1] detection once | dataset={dataset.name} | modality={modality} "
            f"| n_train={len(y_train)} | n_classes={n_classes} | epochs={epochs}"
        )
        print(f"[Stage1] clean_reference={use_clean_ref} | ed2_active={is_tabular}")

    # --- Step 1 (tabular): Run ED2 once for attribute noise detection ---
    ed2_results = None
    if is_tabular:
        try:
            ed2_results = run_ed2_error_detection(
                X=X_train,
                n_iterations=ed2_iterations,
                random_state=random_state,
                verbose=verbose,
            )
            if verbose:
                print(
                    f"[Stage1] ED2 attribute noise: "
                    f"mean={float(np.mean(ed2_results['row_noise_scores'])):.4f} "
                    f"| max={float(np.max(ed2_results['row_noise_scores'])):.4f}"
                )
        except Exception as e:
            if verbose:
                print(f"[Stage1] ED2 failed (fallback to no attribute noise detection): {e}")
            ed2_results = None

    # --- Step 2: Fit downstream model on noisy data once for the shared pass ---
    if hasattr(model, "fit"):
        model.fit(X_train, y_train)

    ilde_state = ILDEState(ed2_state=_build_ed2_state(ed2_results))
    pair_counts: List[int] = []
    clean_sample_counts: List[int] = []
    pseudo_pair_batches: List[Dict[str, Any]] = []
    final_clean_idx = np.array([], dtype=int)

    for epoch in range(epochs):
        ilde_state.epoch = epoch

        # --- ILDE: estimate noise transition matrix and temporal LD ensemble ---
        try:
            proba = model.predict_proba(X_train)
            T = estimate_noise_transition_matrix(proba, n_classes)
            ld = compute_instance_dependent_ld(proba, T)
            ld_sharpened = sharpen_ld(ld, temperature=ilde_temperature)
            ld_ensemble = temporal_ensemble_ld(
                ld_sharpened,
                ilde_state.ld_ensemble,
                beta=ilde_beta,
            )
            ilde_state.ld_ensemble = ld_ensemble

            if verbose and epoch % 2 == 0:
                diag_ld = ld_ensemble.max(axis=1)
                print(
                    f"  [epoch {epoch + 1}] mean_LD_max={diag_ld.mean():.4f} "
                    f"| LD_std={diag_ld.std():.4f}"
                )
        except Exception as e:
            if verbose:
                print(f"  [epoch {epoch + 1}] ILDE estimation failed: {e}. Using fallback.")

        # --- Identify high-confidence clean samples ---
        ed2_for_clean_id = ed2_results if is_tabular else None
        clean_idx = identify_high_confidence_clean_samples(
            X_train,
            y_train,
            model,
            confidence_threshold=confidence_threshold,
            top_k_ratio=top_k_ratio,
            use_clean_reference=use_clean_ref,
            dataset_clean_labels=ref_labels,
            ed2_results=ed2_for_clean_id,
        )
        clean_idx = np.asarray(clean_idx, dtype=int)
        clean_sample_counts.append(int(len(clean_idx)))
        final_clean_idx = clean_idx.copy()

        if verbose and epoch % 2 == 0:
            extra = ""
            if is_tabular and ed2_results is not None and len(clean_idx) > 0:
                extra = f"| ed2={float(np.mean(ed2_results['row_noise_scores'][clean_idx])):.4f}"
            print(f"  [epoch {epoch + 1}] clean samples: {len(clean_idx)}/{len(y_train)} {extra}")

        # --- Build pseudo-label pairs every 2 epochs, once, then cache them ---
        if epoch % 2 == 0 or epoch == epochs - 1:
            pairs = build_pseudo_label_pairs(
                X=X_train,
                y=y_train,
                clean_indices=clean_idx.copy(),
                label_flip_ratio=pair_label_flip_ratio,
                feature_corrupt_ratio=pair_feature_corrupt_ratio,
                outlier_ratio=pair_outlier_ratio,
                n_classes=n_classes,
                random_state=random_state + epoch,
            )
            pair_counts.append(int(len(pairs)))
            pseudo_pair_batches.append({
                "epoch": int(epoch),
                "pairs": pairs,
                "n_pairs": int(len(pairs)),
            })

            if verbose:
                n_label = sum(1 for p in pairs if p.noise_type == "label_flip")
                n_feat = sum(1 for p in pairs if p.noise_type == "feature_corrupt")
                n_out = sum(1 for p in pairs if p.noise_type == "outlier")
                print(
                    f"  [epoch {epoch + 1}] pseudo pairs cached: total={len(pairs)} "
                    f"| label_flip={n_label} | feature={n_feat} | outlier={n_out}"
                )

        # --- Refit model every 2 epochs to update ILDE estimates, once only ---
        if epoch < epochs - 1 and epoch % 2 == 0:
            try:
                model.fit(X_train, y_train)
            except Exception:
                pass

    mean_pairs = float(np.mean(pair_counts)) if pair_counts else 0.0
    mean_clean = float(np.mean(clean_sample_counts)) if clean_sample_counts else 0.0

    result = {
        "stage": "stage_1_ilde_detection_cache",
        "epochs": epochs,
        "modality": modality,
        "is_tabular": is_tabular,
        "ed2_active": is_tabular and ed2_results is not None,
        "n_clean_samples_mean": mean_clean,
        "n_pairs_mean": mean_pairs,
        "pair_loss_final": None,  # selector-specific; filled after selector warmup
        "clean_sample_counts": clean_sample_counts,
        "pair_counts": pair_counts,
        "ilde_beta": ilde_beta,
        "ilde_alpha": ilde_alpha,
        "ilde_temperature": ilde_temperature,
        "confidence_threshold": confidence_threshold,
        "top_k_ratio": top_k_ratio,
        "pair_ratios": {
            "label_flip": pair_label_flip_ratio,
            "feature_corrupt": pair_feature_corrupt_ratio,
            "outlier": pair_outlier_ratio,
        },
        "ed2_results": _summarize_ed2_results(ed2_results),
        "stage1_detection_runs": 1,
        "_final_clean_indices": final_clean_idx,
        "_pseudo_pair_batches": pseudo_pair_batches,
        "_ed2_raw": ed2_results,
        "_ilde_state": ilde_state,
    }

    if verbose:
        ed2_info = f" | ed2_attr={float(np.mean(ed2_results['row_noise_scores'])):.4f}" if ed2_results else ""
        print(
            f"[Stage1] Detection done. modality={modality} | clean_mean={mean_clean:.0f} "
            f"| pairs_mean={mean_pairs:.0f}{ed2_info}"
        )

    return result


def warmup_selector_from_stage1_cache(
    selector,
    stage1_cache: Dict[str, Any],
    dataset,
    model,
    inner_epochs: int = 2,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Warm up one selector using a shared Stage-1 detection cache.

    This keeps selector-specific training separate while avoiding repeated ED2,
    ILDE clean-sample detection, pseudo-pair construction, and model refits.
    """
    X_train = np.asarray(dataset.X_train)
    y_train = np.asarray(dataset.y_train)

    pair_results: List[Dict[str, Any]] = []
    pair_losses: List[float] = []
    pair_batches = stage1_cache.get("_pseudo_pair_batches", [])

    for batch in pair_batches:
        pairs = batch.get("pairs", [])
        if not pairs:
            continue
        try:
            pair_result = train_selector_on_pairs(
                selector=selector,
                pairs=pairs,
                X_all=X_train,
                y_all=y_train,
                model=model,
                epochs=inner_epochs,
                verbose=verbose,
            )
            pair_results.append({
                "epoch": int(batch.get("epoch", -1)),
                "n_pairs": int(len(pairs)),
                "mean_loss": float(pair_result.get("mean_loss", 1.0)),
                "final_pair_acc": float(pair_result.get("final_pair_acc", 0.0)),
            })
            pair_losses.append(float(pair_result.get("mean_loss", 1.0)))
        except Exception as e:
            if verbose:
                print(f"  [selector warmup] failed on cached epoch {batch.get('epoch')}: {e}")

    public = _stage1_public_view(stage1_cache).copy()
    mean_pair_loss = float(np.mean(pair_losses)) if pair_losses else 1.0
    public.update({
        "stage": "stage_1_ilde_warmup",
        "pair_loss_final": mean_pair_loss,
        "selector_cached_warmup": True,
        "selector_pair_batches": [
            {k: v for k, v in r.items() if k in ("epoch", "n_pairs", "mean_loss", "final_pair_acc")}
            for r in pair_results
        ],
        "_final_clean_indices": stage1_cache.get("_final_clean_indices"),
        "_selector_pair_results": pair_results,
    })
    return public


def run_stage1_ilde_warmup(
    dataset,
    model,
    clean_y_train: Optional[np.ndarray] = None,
    epochs: int = 6,
    confidence_threshold: float = 0.85,
    top_k_ratio: float = 0.30,
    pair_label_flip_ratio: float = 0.40,
    pair_feature_corrupt_ratio: float = 0.30,
    pair_outlier_ratio: float = 0.30,
    ilde_alpha: float = 2.0,
    ilde_beta: float = 0.6,
    ilde_temperature: float = 0.1,
    batch_size: int = 64,
    verbose: bool = False,
    selector=None,
    random_state: int = 42,
    modality: Optional[str] = None,
    ed2_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Backward-compatible Stage-1 entrypoint.

    For a single selector call, this still performs a full detection pass and
    then warms that selector.  ARGOTrainer.warmup_selectors() now uses the two
    lower-level functions directly so the expensive detection pass is shared.
    """
    cache = run_stage1_ilde_detection(
        dataset=dataset,
        model=model,
        clean_y_train=clean_y_train,
        epochs=epochs,
        confidence_threshold=confidence_threshold,
        top_k_ratio=top_k_ratio,
        pair_label_flip_ratio=pair_label_flip_ratio,
        pair_feature_corrupt_ratio=pair_feature_corrupt_ratio,
        pair_outlier_ratio=pair_outlier_ratio,
        ilde_alpha=ilde_alpha,
        ilde_beta=ilde_beta,
        ilde_temperature=ilde_temperature,
        batch_size=batch_size,
        verbose=verbose,
        random_state=random_state,
        modality=modality,
        ed2_iterations=ed2_iterations,
    )

    if selector is None:
        public = _stage1_public_view(cache).copy()
        public["stage"] = "stage_1_ilde_warmup"
        public["pair_loss_final"] = 1.0
        return public

    return warmup_selector_from_stage1_cache(
        selector=selector,
        stage1_cache=cache,
        dataset=dataset,
        model=model,
        inner_epochs=2,
        verbose=verbose,
    )
