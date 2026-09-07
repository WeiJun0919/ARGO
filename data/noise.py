import numpy as np


def inject_label_noise(y, noise_ratio=0.25, n_classes=None, random_state=42):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    y_noisy = y.copy()
    if n_classes is None:
        n_classes = int(np.max(y) + 1)
    n_flip = int(len(y) * noise_ratio)
    idx = rng.choice(len(y), size=n_flip, replace=False) if n_flip > 0 else np.array([], dtype=int)
    for i in idx:
        old = int(y_noisy[i])
        choices = [c for c in range(n_classes) if c != old]
        if choices:
            y_noisy[i] = rng.choice(choices)
    return y_noisy, idx


def inject_numeric_feature_noise(X, feature_noise_ratio=0.03, outlier_ratio=0.03, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float).copy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.ndim > 2:
        X = X.reshape(len(X), -1)

    n, d = X.shape
    std = X.std(axis=0) + 1e-8

    if feature_noise_ratio > 0:
        mask = rng.random(X.shape) < feature_noise_ratio
        if mask.any():
            noise = rng.normal(0, 1.0, size=X.shape)
            X[mask] += noise[mask] * std[np.where(mask)[1]]

    n_out = int(n * outlier_ratio)
    outlier_rows = rng.choice(n, size=n_out, replace=False) if n_out > 0 else np.array([], dtype=int)
    for r in outlier_rows:
        cols = rng.choice(d, size=max(1, min(d, max(1, d // 12))), replace=False)
        X[r, cols] += rng.normal(0, 6.0, size=len(cols)) * std[cols]

    return X, {"outlier_rows": outlier_rows}


def _is_image_dataset(X, dataset_name):
    name = str(dataset_name or "").lower()
    return name in {"cifar10", "cifar", "image"} or np.asarray(X).ndim == 4


def _prepare_image_for_noise(X):
    """Return image array as float32 NCHW in [0, 1], plus original layout."""
    X = np.asarray(X)
    original_shape = X.shape
    original_ndim = X.ndim
    layout = "nchw"

    if X.ndim == 2 and X.shape[1] == 3072:
        X_img = X.reshape(-1, 3, 32, 32)
        layout = "flat_cifar"
    elif X.ndim == 4 and X.shape[1] == 3:
        X_img = X
        layout = "nchw"
    elif X.ndim == 4 and X.shape[-1] == 3:
        X_img = X.transpose(0, 3, 1, 2)
        layout = "nhwc"
    else:
        # Fall back to numeric corruption if the shape is not an image shape.
        return None, original_shape, original_ndim, layout

    orig_dtype = X_img.dtype
    X_img = X_img.astype("float32", copy=True)
    X_img = np.nan_to_num(X_img, nan=0.0, posinf=1.0, neginf=0.0)

    # Only divide by 255 for true uint8 / 0..255 images.  Do NOT divide merely
    # because a small amount of synthetic image noise pushed max above 1.0.
    if np.issubdtype(orig_dtype, np.integer) or float(np.nanmax(X_img)) > 10.0:
        X_img = X_img / 255.0

    X_img = np.clip(X_img, 0.0, 1.0)
    return X_img, original_shape, original_ndim, layout


def _restore_image_layout(X_img, original_shape, layout):
    if layout == "flat_cifar":
        return X_img.reshape(original_shape).astype("float32", copy=False)
    if layout == "nhwc":
        return X_img.transpose(0, 2, 3, 1).astype("float32", copy=False)
    return X_img.astype("float32", copy=False)


def inject_image_feature_noise(X, feature_noise_ratio=0.03, outlier_ratio=0.03, random_state=42):
    """
    CIFAR/image-aware feature corruption.

    The previous tabular noise path flattened images and could create pixel
    values > 1.5.  TorchCIFARClassifier then mistook the already-normalized
    training images for 0..255 uint8 images and divided them by 255, creating a
    train/validation scale mismatch and near-random accuracy.  This function
    preserves image shape and clips corrupted pixels back to [0, 1].
    """
    rng = np.random.default_rng(random_state)
    prepared = _prepare_image_for_noise(X)
    X_img, original_shape, original_ndim, layout = prepared
    if X_img is None:
        return inject_numeric_feature_noise(X, feature_noise_ratio, outlier_ratio, random_state)

    if feature_noise_ratio > 0:
        mask = rng.random(X_img.shape) < float(feature_noise_ratio)
        if mask.any():
            # Mild pixel-level Gaussian noise, appropriate for [0, 1] images.
            X_img[mask] += rng.normal(0.0, 0.08, size=int(mask.sum())).astype("float32")

    n = len(X_img)
    n_out = int(n * float(outlier_ratio))
    outlier_rows = rng.choice(n, size=n_out, replace=False) if n_out > 0 else np.array([], dtype=int)

    # Patch corruption is a more realistic image outlier than arbitrary flat
    # feature spikes.  It also keeps values in the valid pixel range.
    for r in outlier_rows:
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
        X_img[r, :, top:top + h, left:left + w] = patch

    X_img = np.clip(X_img, 0.0, 1.0)
    return _restore_image_layout(X_img, original_shape, layout), {"outlier_rows": outlier_rows}


def inject_dataset_noise(
    X,
    y,
    dataset_name="synthetic",
    label_noise_ratio=0.25,
    feature_noise_ratio=0.03,
    outlier_ratio=0.03,
    random_state=42,
):
    y_noisy, label_idx = inject_label_noise(
        y,
        noise_ratio=label_noise_ratio,
        n_classes=int(np.max(y) + 1),
        random_state=random_state,
    )

    if _is_image_dataset(X, dataset_name):
        X_noisy, info = inject_image_feature_noise(
            X,
            feature_noise_ratio=feature_noise_ratio,
            outlier_ratio=outlier_ratio,
            random_state=random_state + 1,
        )
    else:
        X_noisy, info = inject_numeric_feature_noise(
            X,
            feature_noise_ratio=feature_noise_ratio,
            outlier_ratio=outlier_ratio,
            random_state=random_state + 1,
        )

    info.update({
        "label_noise_indices": label_idx,
        "label_noise_ratio": label_noise_ratio,
        "feature_noise_ratio": feature_noise_ratio,
        "outlier_ratio": outlier_ratio,
        "modality": dataset_name,
    })
    return X_noisy, y_noisy, info
