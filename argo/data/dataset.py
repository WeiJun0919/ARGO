import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
from sklearn.model_selection import train_test_split


@dataclass
class TabularDataset:
    X_train: object
    y_train: np.ndarray

    X_val: object
    y_val: np.ndarray

    feature_names: Optional[Sequence[str]] = None
    class_names: Optional[Sequence[str]] = None

    # Clean, pre-noise features aligned with the original training rows.
    # This lets examples compare three states fairly:
    #   clean original data -> injected-noise data -> HR/ARGO-cleaned data.
    X_train_clean: object = None

    y_train_clean: Optional[np.ndarray] = None
    y_val_clean: Optional[np.ndarray] = None

    # Boolean mask aligned with current X_train/y_train. True means this row
    # was identified as high-confidence clean by Stage 1 and should be frozen.
    protected_clean_mask: Optional[np.ndarray] = None

    # 20% clean augmentation pool
    X_aug_pool: object = None
    y_aug_pool: Optional[np.ndarray] = None
    y_aug_pool_clean: Optional[np.ndarray] = None

    name: str = "unknown"
    modality: str = "tabular"
    metadata: Optional[Dict[str, Any]] = None

    raw_train_df: object = None
    raw_val_df: object = None
    raw_aug_pool_df: object = None

    label_col: str = "label"

    def copy(self):
        return TabularDataset(
            X_train=self.X_train.copy(),
            y_train=self.y_train.copy(),
            X_val=self.X_val.copy(),
            y_val=self.y_val.copy(),

            X_train_clean=None if self.X_train_clean is None else self.X_train_clean.copy(),

            feature_names=None if self.feature_names is None else list(self.feature_names),
            class_names=None if self.class_names is None else list(self.class_names),

            y_train_clean=None if self.y_train_clean is None else self.y_train_clean.copy(),
            y_val_clean=None if self.y_val_clean is None else self.y_val_clean.copy(),
            protected_clean_mask=(
                None
                if self.protected_clean_mask is None
                else self.protected_clean_mask.copy()
            ),

            X_aug_pool=None if self.X_aug_pool is None else self.X_aug_pool.copy(),
            y_aug_pool=None if self.y_aug_pool is None else self.y_aug_pool.copy(),
            y_aug_pool_clean=None if self.y_aug_pool_clean is None else self.y_aug_pool_clean.copy(),

            name=self.name,
            modality=self.modality,
            metadata=None if self.metadata is None else dict(self.metadata),

            raw_train_df=None if self.raw_train_df is None else self.raw_train_df.copy(),
            raw_val_df=None if self.raw_val_df is None else self.raw_val_df.copy(),
            raw_aug_pool_df=None if self.raw_aug_pool_df is None else self.raw_aug_pool_df.copy(),

            label_col=self.label_col,
        )

    @property
    def n_classes(self):
        arrays = [self.y_train, self.y_val]

        if self.y_train_clean is not None:
            arrays.append(self.y_train_clean)

        if self.y_val_clean is not None:
            arrays.append(self.y_val_clean)

        if self.y_aug_pool is not None:
            arrays.append(self.y_aug_pool)

        if self.y_aug_pool_clean is not None:
            arrays.append(self.y_aug_pool_clean)

        return int(max(int(np.max(a)) for a in arrays if a is not None and len(a) > 0) + 1)

    @property
    def n_features(self):
        shape = self.X_train.shape
        if len(shape) <= 1:
            return 1
        return int(np.prod(shape[1:]))


def from_arrays_with_clean_labels(
    X,
    y_noisy,
    y_clean,
    X_clean=None,
    X_val=None,
    y_val=None,
    test_size=0.2,
    aug_pool_ratio=0.2,
    random_state=42,
    name="array",
    modality="tabular",
    metadata=None,
    feature_names=None,
    class_names=None,
    raw_df=None,
    raw_val_df=None,
    label_col="label",
):
    """
    构造 Dataset。

    新逻辑：

    如果没有外部 X_val / y_val：
        clean full data
        ↓
        80% main data
        20% clean augmentation pool
        ↓
        main data 再划分 train / val

    如果有外部 X_val / y_val：
        X / y_noisy / y_clean 作为主训练池
        同时从主训练池里额外切出 20% 作为 clean augmentation pool
        剩余 80% 作为实际 X_train
        外部 X_val / y_val 保持不变
    """
    y_noisy = np.asarray(y_noisy)
    y_clean = np.asarray(y_clean)
    X_clean = X if X_clean is None else X_clean

    idx = np.arange(len(y_clean))

    # --------------------------------------------------
    # 1. 先划分 main pool 和 clean augmentation pool
    # --------------------------------------------------
    if aug_pool_ratio is not None and aug_pool_ratio > 0:
        main_idx, aug_idx = train_test_split(
            idx,
            test_size=aug_pool_ratio,
            stratify=y_clean,
            random_state=random_state,
        )
    else:
        main_idx = idx
        aug_idx = np.array([], dtype=int)

    X_main = X[main_idx]
    X_clean_main = X_clean[main_idx]
    y_noisy_main = y_noisy[main_idx]
    y_clean_main = y_clean[main_idx]

    # Candidate generation must not see the clean evaluation reference.  The
    # observed (possibly noisy) data form the augmentation pool; clean values
    # are retained only as optional bookkeeping for synthetic evaluations.
    X_aug_pool = X[aug_idx] if len(aug_idx) > 0 else None
    y_aug_pool = y_noisy[aug_idx] if len(aug_idx) > 0 else None
    y_aug_pool_clean = y_clean[aug_idx] if len(aug_idx) > 0 else None

    raw_aug_pool_df = None
    raw_main_df = None

    if raw_df is not None:
        raw_main_df = raw_df.iloc[main_idx].reset_index(drop=True)
        raw_aug_pool_df = raw_df.iloc[aug_idx].reset_index(drop=True) if len(aug_idx) > 0 else None

    # --------------------------------------------------
    # 2. 如果没有外部验证集，从 main pool 再划分 train / val
    # --------------------------------------------------
    if X_val is None or y_val is None:
        main_local_idx = np.arange(len(y_clean_main))

        train_local_idx, val_local_idx = train_test_split(
            main_local_idx,
            test_size=test_size,
            stratify=y_clean_main,
            random_state=random_state,
        )

        X_train = X_main[train_local_idx]
        X_train_clean = X_clean_main[train_local_idx]
        X_val = X_main[val_local_idx]

        y_train = y_noisy_main[train_local_idx]
        y_val = y_clean_main[val_local_idx]

        y_train_clean = y_clean_main[train_local_idx]
        y_val_clean = y_clean_main[val_local_idx]

        raw_train_df = None
        raw_val_df_final = None

        if raw_main_df is not None:
            raw_train_df = raw_main_df.iloc[train_local_idx].reset_index(drop=True)
            raw_val_df_final = raw_main_df.iloc[val_local_idx].reset_index(drop=True)

    # --------------------------------------------------
    # 3. 如果有外部验证集，main pool 全部作为 train
    # --------------------------------------------------
    else:
        X_train = X_main
        X_train_clean = X_clean_main
        y_train = y_noisy_main
        y_train_clean = y_clean_main

        y_val = np.asarray(y_val)
        y_val_clean = y_val

        raw_train_df = None if raw_main_df is None else raw_main_df.reset_index(drop=True)
        raw_val_df_final = raw_val_df

    return TabularDataset(
        X_train=X_train,
        y_train=y_train,

        X_val=X_val,
        y_val=y_val,

        X_train_clean=X_train_clean,

        feature_names=feature_names,
        class_names=class_names,

        y_train_clean=y_train_clean,
        y_val_clean=y_val_clean,

        X_aug_pool=X_aug_pool,
        y_aug_pool=y_aug_pool,
        y_aug_pool_clean=y_aug_pool_clean,

        name=name,
        modality=modality,
        metadata=metadata or {},

        raw_train_df=raw_train_df,
        raw_val_df=raw_val_df_final,
        raw_aug_pool_df=raw_aug_pool_df,

        label_col=label_col,
    )
