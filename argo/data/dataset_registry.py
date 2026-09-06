from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from argo.data.dataset import from_arrays_with_clean_labels
from argo.data.noise import inject_dataset_noise


def _find_file(root, names):
    root = Path(root)
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _detect_label_col(df):
    for c in ["income", "sentiment", "label", "target", "class", "y"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def _encode_labels(y_raw, class_names=None):
    y_raw = pd.Series(y_raw).astype(str).str.strip()

    if class_names is None:
        class_names = list(y_raw.astype("category").cat.categories)

    mapping = {name: i for i, name in enumerate(class_names)}
    unknown = sorted(set(y_raw) - set(mapping))

    if unknown:
        raise ValueError(f"Validation labels contain unseen classes: {unknown}")

    y = y_raw.map(mapping).to_numpy(dtype=int)
    return y, class_names


def _maybe_subsample(X, y, max_samples, random_state=42):
    if max_samples and len(y) > int(max_samples):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=int(max_samples), replace=False)
        return X[idx], y[idx]
    return X, y


def _safe_split_indices(y, test_size, random_state):
    idx = np.arange(len(y))

    try:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
        )

    return train_idx, test_idx


def _build_dataset_with_clean_aug_pool(
    X_clean,
    y_clean,
    X_val,
    y_val,
    meta,
    dataset_name,
    label_noise_ratio,
    feature_noise_ratio,
    outlier_ratio,
    test_size,
    random_state,
    max_samples=None,
    aug_pool_ratio=0.2,
):
    """
    新数据流：

    clean dataset
    ↓
    80% main pool + 20% clean augmentation pool
    ↓
    只对 80% main pool 注入噪声
    ↓
    main pool 进入 RL train / val
    ↓
    augmentation pool 保持干净，只供 augment 使用
    """
    X_clean, y_clean = _maybe_subsample(
        X_clean,
        y_clean,
        max_samples=max_samples,
        random_state=random_state,
    )

    if aug_pool_ratio and aug_pool_ratio > 0:
        main_idx, aug_idx = _safe_split_indices(
            y_clean,
            test_size=aug_pool_ratio,
            random_state=random_state,
        )
    else:
        main_idx = np.arange(len(y_clean))
        aug_idx = np.array([], dtype=int)

    X_main_clean = X_clean[main_idx]
    y_main_clean = y_clean[main_idx]

    X_aug_pool = X_clean[aug_idx] if len(aug_idx) > 0 else None
    y_aug_pool_clean = y_clean[aug_idx] if len(aug_idx) > 0 else None

    X_noisy, y_noisy, noise_info = inject_dataset_noise(
        X_main_clean,
        y_main_clean,
        dataset_name=dataset_name,
        label_noise_ratio=label_noise_ratio,
        feature_noise_ratio=feature_noise_ratio,
        outlier_ratio=outlier_ratio,
        random_state=random_state,
    )

    meta = {**meta, **noise_info}
    meta["aug_pool_ratio"] = aug_pool_ratio
    meta["aug_pool_size"] = 0 if X_aug_pool is None else len(X_aug_pool)

    dataset = from_arrays_with_clean_labels(
        X_noisy,
        y_noisy=y_noisy,
        y_clean=y_main_clean,
        X_clean=X_main_clean,
        X_val=X_val,
        y_val=y_val,
        test_size=test_size,
        random_state=random_state,
        name=dataset_name,
        modality=meta.get("modality", "tabular"),
        metadata=meta,
        feature_names=meta.get("feature_names"),
        class_names=meta.get("class_names"),
        aug_pool_ratio=0.0,
    )

    dataset.X_aug_pool = X_aug_pool
    dataset.y_aug_pool = y_aug_pool_clean
    dataset.y_aug_pool_clean = y_aug_pool_clean

    return dataset


def load_synthetic_clean(random_state=42, max_samples=None):
    n = int(max_samples) if max_samples else 700

    X, y = make_classification(
        n_samples=n,
        n_features=16,
        n_informative=10,
        n_redundant=2,
        n_classes=3,
        weights=[0.65, 0.25, 0.10],
        class_sep=1.25,
        random_state=random_state,
    )

    return X, y, None, None, {
        "modality": "tabular",
        "class_names": ["0", "1", "2"],
        "feature_names": [str(i) for i in range(X.shape[1])],
        "has_user_reference": False,
    }


def _read_adult_file(file):
    if file.suffix.lower() == ".data":
        cols = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        return pd.read_csv(
            file,
            header=None,
            names=cols,
            na_values=["?", " ?"],
            skipinitialspace=True,
        )

    return pd.read_csv(file)


def load_adult_clean(data_root):
    root = Path(data_root)

    train_file = _find_file(
        root,
        ["adult.csv", "adult_clean.csv", "adult.data", "adult_train.csv"],
    )
    val_file = _find_file(
        root,
        ["adult_val.csv", "adult_clean_val.csv"],
    )

    if train_file is None:
        raise FileNotFoundError(
            f"Cannot find Adult data under {root}. "
            f"Expected adult.csv/adult_clean.csv/adult.data/adult_train.csv."
        )

    df_train = _read_adult_file(train_file)
    label_col = _detect_label_col(df_train)

    df_val = None
    if val_file is not None:
        df_val = pd.read_csv(val_file)
        if label_col not in df_val.columns:
            raise ValueError(
                f"Validation file {val_file} does not contain label column '{label_col}'."
            )

    y_train, class_names = _encode_labels(df_train[label_col])

    y_val = None
    if df_val is not None:
        y_val, _ = _encode_labels(
            df_val[label_col],
            class_names=class_names,
        )

    X_train_df = pd.get_dummies(
        df_train.drop(columns=[label_col]),
        dummy_na=True,
    )
    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train_df.to_numpy(dtype=float))

    X_val = None
    if df_val is not None:
        X_val_df = pd.get_dummies(
            df_val.drop(columns=[label_col]),
            dummy_na=True,
        )
        X_val_df = X_val_df.reindex(
            columns=X_train_df.columns,
            fill_value=0,
        )
        X_val_df = X_val_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val = scaler.transform(X_val_df.to_numpy(dtype=float))

    return X_train, y_train, X_val, y_val, {
        "modality": "tabular",
        "feature_names": list(X_train_df.columns),
        "class_names": class_names,
        "label_col": label_col,
        "has_user_reference": df_val is not None,
    }


def load_imdb_clean(data_root, max_features=2000):
    root = Path(data_root)

    train_file = _find_file(
        root,
        ["imdb.csv", "imdb_clean.csv", "IMDB Dataset.csv", "imdb_train.csv"],
    )
    val_file = _find_file(
        root,
        ["imdb_val.csv", "imdb_clean_val.csv"],
    )

    if train_file is None:
        raise FileNotFoundError(
            f"Cannot find IMDB data under {root}. "
            f"Expected imdb.csv/imdb_clean.csv/'IMDB Dataset.csv'/imdb_train.csv."
        )

    df_train = pd.read_csv(train_file)

    text_col = next(
        (c for c in ["review", "text", "sentence", "content"] if c in df_train.columns),
        df_train.columns[0],
    )
    label_col = _detect_label_col(df_train)

    df_val = None
    if val_file is not None:
        df_val = pd.read_csv(val_file)
        if text_col not in df_val.columns or label_col not in df_val.columns:
            raise ValueError(
                f"Validation file {val_file} must contain text column "
                f"'{text_col}' and label column '{label_col}'."
            )

    y_train, class_names = _encode_labels(df_train[label_col])

    y_val = None
    if df_val is not None:
        y_val, _ = _encode_labels(
            df_val[label_col],
            class_names=class_names,
        )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    X_train = vectorizer.fit_transform(
        df_train[text_col].astype(str),
    ).toarray().astype("float32")

    X_val = None
    if df_val is not None:
        X_val = vectorizer.transform(
            df_val[text_col].astype(str),
        ).toarray().astype("float32")

    return X_train, y_train, X_val, y_val, {
        "modality": "text",
        "feature_names": [f"tfidf_{i}" for i in range(X_train.shape[1])],
        "class_names": class_names,
        "label_col": label_col,
        "text_col": text_col,
        "has_user_reference": df_val is not None,
    }


def _load_cifar_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")

    X = batch.get("data")
    y = batch.get("labels") or batch.get("fine_labels")

    return X, np.asarray(y, dtype=int)


def _load_cifar10_raw(data_root):
    root = Path(data_root)

    npz_file = _find_file(root, ["cifar10.npz", "cifar10_clean.npz"])
    if npz_file is not None:
        data = np.load(npz_file, allow_pickle=True)

        X = (
            data["X"]
            if "X" in data
            else data["x"]
            if "x" in data
            else data["X_train"]
        )
        y = (
            data["y"]
            if "y" in data
            else data["labels"]
            if "labels" in data
            else data["y_train"]
        )

        X_val = data["X_val"] if "X_val" in data else None
        y_val = data["y_val"] if "y_val" in data else None

        return (
            np.asarray(X),
            np.asarray(y, dtype=int),
            None if X_val is None else np.asarray(X_val),
            None if y_val is None else np.asarray(y_val, dtype=int),
        )

    batch_dir = root / "cifar-10-batches-py"

    if not batch_dir.exists() and root.name == "cifar-10-batches-py":
        batch_dir = root

    if not batch_dir.exists():
        raise FileNotFoundError(
            f"Cannot find CIFAR-10 under {root}. "
            f"Expected cifar10.npz or cifar-10-batches-py/."
        )

    Xs, ys = [], []

    for i in range(1, 6):
        batch_path = batch_dir / f"data_batch_{i}"
        if batch_path.exists():
            Xb, yb = _load_cifar_batch(batch_path)
            Xs.append(Xb)
            ys.append(yb)

    if not Xs:
        raise FileNotFoundError(
            f"No CIFAR data_batch_1..5 files found under {batch_dir}."
        )

    X = np.vstack(Xs)
    y = np.concatenate(ys).astype(int)

    X_val = None
    y_val = None

    test_batch = batch_dir / "test_batch"
    if test_batch.exists():
        X_val, y_val = _load_cifar_batch(test_batch)

    return X, y, X_val, y_val


def _cifar_to_nchw_float(X):
    X = np.asarray(X)

    if X.ndim == 2 and X.shape[1] == 3072:
        X = X.reshape(-1, 3, 32, 32)

    elif X.ndim == 4 and X.shape[-1] == 3:
        X = X.transpose(0, 3, 1, 2)

    elif X.ndim == 4 and X.shape[1] == 3:
        pass

    else:
        return X.reshape(len(X), -1).astype("float32")

    X = X.astype("float32")

    if X.max() > 1.5:
        X = X / 255.0

    return X


def load_cifar10_clean(
    data_root,
    max_samples=None,
    random_state=42,
    max_val_samples=2000,
):
    """
    Load CIFAR-10 as real image tensors, not ResNet50 embeddings.

    Returned X shape is NCHW float32 in [0, 1], i.e. (N, 3, 32, 32).
    The downstream image model now trains end-to-end on these 32x32 images.
    """
    X_raw, y, X_val_raw, y_val = _load_cifar10_raw(data_root)

    if max_samples and len(y) > int(max_samples):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=int(max_samples), replace=False)
        X_raw = X_raw[idx]
        y = y[idx]

    if (
        X_val_raw is not None
        and y_val is not None
        and max_val_samples
        and len(y_val) > int(max_val_samples)
    ):
        rng = np.random.default_rng(random_state + 17)
        idx = rng.choice(len(y_val), size=int(max_val_samples), replace=False)
        X_val_raw = X_val_raw[idx]
        y_val = y_val[idx]

    X = _cifar_to_nchw_float(X_raw)
    X_val = None if X_val_raw is None else _cifar_to_nchw_float(X_val_raw)

    print(
        f"[CIFAR10] loaded raw 32x32 images: "
        f"train={len(y)} val={0 if y_val is None else len(y_val)}"
    )

    return X, y, X_val, y_val, {
        "modality": "image",
        "feature_names": None,
        "class_names": [str(i) for i in range(10)],
        "has_user_reference": X_val is not None and y_val is not None,
        "feature_extractor": None,
        "backbone": "cifar_resnet18",
        "input_shape": tuple(X.shape[1:]),
        "cifar_val_samples": None if y_val is None else len(y_val),
    }

def build_dataset(
    dataset_name="synthetic",
    data_root="datasets",
    label_noise_ratio=0.25,
    feature_noise_ratio=0.03,
    outlier_ratio=0.03,
    test_size=0.2,
    random_state=42,
    max_samples=None,
    aug_pool_ratio=None,
):
    name = dataset_name.lower()

    if name == "synthetic":
        X_clean, y_clean, X_val, y_val, meta = load_synthetic_clean(
            random_state=random_state,
            max_samples=max_samples,
        )

        # synthetic 已经在 load_synthetic_clean 里使用 max_samples
        local_max_samples = None

    elif name == "adult":
        X_clean, y_clean, X_val, y_val, meta = load_adult_clean(data_root)
        local_max_samples = max_samples

    elif name == "imdb":
        X_clean, y_clean, X_val, y_val, meta = load_imdb_clean(data_root)
        local_max_samples = max_samples

    elif name == "cifar10":
        # CIFAR10 的 max_samples 必须在 ResNet50 提特征前生效
        X_clean, y_clean, X_val, y_val, meta = load_cifar10_clean(
            data_root,
            max_samples=max_samples,
            random_state=random_state,
        )

        # 已经在 load_cifar10_clean 内部采样，不能再次采样
        local_max_samples = None

    else:
        raise ValueError(
            "dataset_name must be one of: synthetic, adult, imdb, cifar10"
        )

    # CIFAR-10 with only a few thousand images is data-limited.  Reserving 20%
    # as a clean augmentation pool means --max-samples 5000 actually trains on
    # only 4000 noisy images.  That made cleaning effects hard to observe and
    # often canceled gains from relabeling.  By default, CIFAR keeps all selected
    # samples in the train pool and ImageAugmentSelector falls back to clean
    # train samples in oracle/simulation experiments.  Other modalities keep the
    # previous 20% augmentation-pool behavior.
    if aug_pool_ratio is None:
        effective_aug_pool_ratio = 0.0 if name == "cifar10" else 0.2
    else:
        effective_aug_pool_ratio = float(aug_pool_ratio)

    return _build_dataset_with_clean_aug_pool(
        X_clean=X_clean,
        y_clean=y_clean,
        X_val=X_val,
        y_val=y_val,
        meta=meta,
        dataset_name=name,
        label_noise_ratio=label_noise_ratio,
        feature_noise_ratio=feature_noise_ratio,
        outlier_ratio=outlier_ratio,
        test_size=test_size,
        random_state=random_state,
        max_samples=local_max_samples,
        aug_pool_ratio=effective_aug_pool_ratio,
    )