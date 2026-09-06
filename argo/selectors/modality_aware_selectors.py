import numpy as np

from argo.selectors.neural_selector import (
    NeuralRemoveSelector,
    NeuralRelabelSelector,
    NeuralAugmentSelector,
    NeuralRepairSelector,
)


def _has_aug_pool(dataset):
    return (
        getattr(dataset, "X_aug_pool", None) is not None
        and getattr(dataset, "y_aug_pool", None) is not None
        and len(dataset.y_aug_pool) > 0
    )


def _select_from_aug_pool(dataset, model, selector, budget, divisor=3, random_state=123):
    """
    从 20% clean augmentation pool 中选择样本。

    返回：
        idx: augmentation pool 中的索引
        mods: apply_operation("augment") 需要的 augmented_X / augmented_y
    """
    if not _has_aug_pool(dataset):
        return None, None

    X_pool = dataset.X_aug_pool
    y_pool = dataset.y_aug_pool
    y_pool_clean = getattr(dataset, "y_aug_pool_clean", None)

    counts = np.bincount(dataset.y_train, minlength=dataset.n_classes)
    pool_scores = 1.0 / (counts[y_pool] + 1e-8)
    pool_scores /= pool_scores.max() + 1e-8
    try:
        proba = model.predict_proba(X_pool)
        confidence = proba.max(axis=1)
        uncertainty = 1.0 - confidence
    except Exception:
        confidence = np.full(len(y_pool), 0.5)
        uncertainty = np.full(len(y_pool), 0.5)
    utility = np.clip(0.55 * pool_scores + 0.30 * uncertainty + 0.15 * confidence, 0.0, 1.0)
    features = np.column_stack([pool_scores, uncertainty, confidence, np.ones(len(y_pool))])
    k = min(max(1, budget // divisor), len(y_pool))
    idx = selector._rank_with_policy(np.arange(len(y_pool)), features, utility, k)

    mods = {
        "augmented_X": X_pool[idx].copy(),
        "augmented_y": y_pool[idx].copy(),
    }

    if y_pool_clean is not None:
        mods["augmented_y_clean"] = y_pool_clean[idx].copy()
    else:
        mods["augmented_y_clean"] = y_pool[idx].copy()

    return idx, mods


class TabularRemoveSelector(NeuralRemoveSelector):
    pass


class TabularRelabelSelector(NeuralRelabelSelector):
    pass


class TabularAugmentSelector(NeuralAugmentSelector):
    """
    表格数据增强：
    优先从 20% clean augmentation pool 中取样本。
    如果没有 augmentation pool，再退回原始 NeuralAugmentSelector。
    """

    def select(self, dataset, model, budget):
        idx, mods = _select_from_aug_pool(
            dataset,
            model,
            self,
            budget=budget,
            divisor=3,
            random_state=101,
        )

        if idx is not None:
            return idx, mods

        return super().select(dataset, model, budget)


class TabularRepairSelector(NeuralRepairSelector):
    pass


class TextRemoveSelector(NeuralRemoveSelector):
    pass


class TextRelabelSelector(NeuralRelabelSelector):
    pass


class TextAugmentSelector(NeuralAugmentSelector):
    """
    文本数据增强：
    TF-IDF 向量不能随便加高斯噪声。
    所以只从 20% clean augmentation pool 中复制样本。
    """

    def select(self, dataset, model, budget):
        idx, mods = _select_from_aug_pool(
            dataset,
            model,
            self,
            budget=budget,
            divisor=3,
            random_state=202,
        )

        if idx is not None:
            return idx, mods

        # 没有 aug_pool 时，保守 fallback：复制干净少数类训练样本，不加噪声
        proba = model.predict_proba(dataset.X_train)
        clean_idx = np.where(proba.argmax(axis=1) == dataset.y_train)[0]

        if len(clean_idx) == 0:
            return np.array([], dtype=int), {
                "augmented_X": dataset.X_train[:0],
                "augmented_y": dataset.y_train[:0],
            }

        counts = np.bincount(dataset.y_train, minlength=dataset.n_classes)
        inv_freq = 1.0 / (counts[dataset.y_train[clean_idx]] + 1e-8)
        inv_freq /= inv_freq.max() + 1e-8
        k = min(max(1, budget // 3), len(clean_idx))
        confidence = proba[clean_idx].max(axis=1)
        utility = np.clip(0.6 * inv_freq + 0.4 * confidence, 0.0, 1.0)
        features = np.column_stack([inv_freq, 1.0 - confidence, confidence, np.ones(len(clean_idx))])
        idx = self._rank_with_policy(clean_idx, features, utility, k)

        mods = {
            "augmented_X": dataset.X_train[idx].copy(),
            "augmented_y": dataset.y_train[idx].copy(),
        }

        if dataset.y_train_clean is not None:
            mods["augmented_y_clean"] = dataset.y_train_clean[idx].copy()

        return idx, mods


class ImageRemoveSelector(NeuralRemoveSelector):
    """
    图像数据 remove：
    小样本 CIFAR 上删除训练图像通常会损失覆盖度，因此默认的 image
    action set 不再包含 remove。保留该类仅供用户显式 --actions remove 时使用。
    """

    def select(self, dataset, model, budget):
        return super().select(dataset, model, max(1, budget // 4))


class ImageRelabelSelector(NeuralRelabelSelector):
    """
    图像数据 relabel：
    CIFAR-10 的主要可控污染是 label noise。相比 remove，优先 relabel
    能保留样本量，5000 张小样本下更容易提升下游分类准确率。
    """

    def select(self, dataset, model, budget):
        # Use the full budget for relabeling. NeuralRelabelSelector already
        # caps the returned set and, when oracle feedback is available, wrong
        # proposals are corrected instead of blindly applied.
        return super().select(dataset, model, max(1, budget))


class ImageAugmentSelector(NeuralAugmentSelector):
    """
    图像数据增强：
    优先从 20% clean augmentation pool 中拿 ResNet50 embedding。
    然后在 embedding 空间加极小扰动。
    """

    def __init__(self):
        super().__init__(noise_scale=0.003)

    def select(self, dataset, model, budget):
        idx, mods = _select_from_aug_pool(
            dataset,
            model,
            self,
            budget=budget,
            divisor=4,
            random_state=303,
        )

        if idx is not None:
            rng = np.random.default_rng(303)
            X_aug = mods["augmented_X"].copy()
            X_aug = X_aug + rng.normal(0, 0.003, size=X_aug.shape)
            if np.asarray(X_aug).ndim >= 3:
                X_aug = np.clip(X_aug, 0.0, 1.0)

            mods["augmented_X"] = X_aug.astype(dataset.X_train.dtype, copy=False)
            return idx, mods

        # 没有 aug_pool 时，fallback 到 clean train samples
        proba = model.predict_proba(dataset.X_train)
        clean_idx = np.where(proba.argmax(axis=1) == dataset.y_train)[0]

        if len(clean_idx) == 0:
            return np.array([], dtype=int), {
                "augmented_X": dataset.X_train[:0],
                "augmented_y": dataset.y_train[:0],
            }

        counts = np.bincount(dataset.y_train, minlength=dataset.n_classes)
        inv_freq = 1.0 / (counts[dataset.y_train[clean_idx]] + 1e-8)
        inv_freq /= inv_freq.max() + 1e-8
        k = min(max(1, budget // 4), len(clean_idx))
        confidence = proba[clean_idx].max(axis=1)
        utility = np.clip(0.6 * inv_freq + 0.4 * confidence, 0.0, 1.0)
        features = np.column_stack([inv_freq, 1.0 - confidence, confidence, np.ones(len(clean_idx))])
        idx = self._rank_with_policy(clean_idx, features, utility, k)

        rng = np.random.default_rng(303)
        X_aug = dataset.X_train[idx].copy()
        X_aug = X_aug + rng.normal(0, 0.003, size=X_aug.shape)
        if np.asarray(X_aug).ndim >= 3:
            X_aug = np.clip(X_aug, 0.0, 1.0)

        mods = {
            "augmented_X": X_aug.astype(dataset.X_train.dtype, copy=False),
            "augmented_y": dataset.y_train[idx].copy(),
        }

        if dataset.y_train_clean is not None:
            mods["augmented_y_clean"] = dataset.y_train_clean[idx].copy()

        return idx, mods


def infer_modality(dataset):
    raw = str(getattr(dataset, "modality", "") or "").lower()
    name = str(getattr(dataset, "name", "") or "").lower()

    if raw in {"tabular", "adult", "synthetic", "csv"} or name in {"adult", "synthetic"}:
        return "tabular"

    if raw in {"text", "imdb"} or name == "imdb":
        return "text"

    if raw in {"image", "cifar10", "cifar"} or name == "cifar10":
        return "image"

    return "tabular"


def build_selectors_for_dataset(dataset):
    modality = infer_modality(dataset)

    if modality == "tabular":
        actions = ["noop", "remove", "augment", "relabel", "repair"]
        selectors = {
            "remove": TabularRemoveSelector(),
            "augment": TabularAugmentSelector(),
            "relabel": TabularRelabelSelector(),
            "repair": TabularRepairSelector(),
        }

    elif modality == "text":
        actions = ["noop", "remove", "augment", "relabel"]
        selectors = {
            "remove": TextRemoveSelector(),
            "augment": TextAugmentSelector(),
            "relabel": TextRelabelSelector(),
        }

    elif modality == "image":
        # Keep remove available for explicit --actions, but run_tabular_demo
        # filters it out by default on CIFAR-10 small-sample runs.
        actions = ["noop", "relabel", "augment", "remove"]
        selectors = {
            "relabel": ImageRelabelSelector(),
            "augment": ImageAugmentSelector(),
            "remove": ImageRemoveSelector(),
        }

    else:
        actions = ["noop", "remove", "augment", "relabel", "repair"]
        selectors = {
            "remove": NeuralRemoveSelector(),
            "augment": NeuralAugmentSelector(),
            "relabel": NeuralRelabelSelector(),
            "repair": NeuralRepairSelector(),
        }

    return actions, selectors, modality
