import numpy as np


def _has_clean_mask(dataset):
    return getattr(dataset, "protected_clean_mask", None) is not None


def _append_clean_mask(dataset, n_new):
    if _has_clean_mask(dataset):
        dataset.protected_clean_mask = np.concatenate([
            np.asarray(dataset.protected_clean_mask, dtype=bool),
            np.zeros(int(n_new), dtype=bool),
        ])


def _slice_raw_df(raw_df, keep):
    if raw_df is None:
        return None
    try:
        return raw_df.iloc[keep].reset_index(drop=True)
    except Exception:
        return raw_df


def apply_operation(dataset, action_name, indices, modifications):
    new_ds = dataset.copy()
    indices = np.asarray(indices, dtype=int)

    if action_name == "noop":
        return new_ds

    if action_name == "remove":
        keep = np.ones(len(new_ds.y_train), dtype=bool)
        keep[indices] = False
        new_ds.X_train = new_ds.X_train[keep]
        if getattr(new_ds, "X_train_clean", None) is not None:
            new_ds.X_train_clean = new_ds.X_train_clean[keep]
        new_ds.y_train = new_ds.y_train[keep]
        if new_ds.y_train_clean is not None:
            new_ds.y_train_clean = new_ds.y_train_clean[keep]
        if _has_clean_mask(new_ds):
            new_ds.protected_clean_mask = np.asarray(new_ds.protected_clean_mask, dtype=bool)[keep]
        new_ds.raw_train_df = _slice_raw_df(new_ds.raw_train_df, keep)
        return new_ds

    if action_name == "relabel":
        if len(indices) > 0:
            new_ds.y_train[indices] = modifications["new_labels"]
        return new_ds

    if action_name == "augment":
        aug_X = modifications.get("augmented_X")
        aug_y = modifications.get("augmented_y")
        if aug_X is None or aug_y is None or len(aug_y) == 0:
            return new_ds
        new_ds.X_train = np.vstack([new_ds.X_train, aug_X])
        if getattr(new_ds, "X_train_clean", None) is not None:
            clean_aug_X = modifications.get("augmented_X_clean", aug_X)
            new_ds.X_train_clean = np.vstack([new_ds.X_train_clean, clean_aug_X])
        new_ds.y_train = np.concatenate([new_ds.y_train, aug_y])
        if new_ds.y_train_clean is not None:
            clean_aug = modifications.get("augmented_y_clean", new_ds.y_train_clean[indices])
            new_ds.y_train_clean = np.concatenate([new_ds.y_train_clean, clean_aug])
        _append_clean_mask(new_ds, len(aug_y))
        return new_ds

    if action_name == "repair":
        if len(indices) > 0:
            new_ds.X_train[indices] = modifications["repaired_X"]
            if getattr(new_ds, "X_train_clean", None) is not None and "repaired_X_clean" in modifications:
                new_ds.X_train_clean[indices] = modifications["repaired_X_clean"]
        return new_ds

    raise ValueError(f"Unknown action: {action_name}")
