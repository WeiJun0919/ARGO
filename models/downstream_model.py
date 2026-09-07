import math
import os
import random
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# TorchCIFARClassifier — handles image classifiers
# ─────────────────────────────────────────────────────────────────────────────
class TorchCIFARClassifier:
    """
    End-to-end PyTorch classifier for CIFAR-style image tensors.

    Backbones:
      - resnet50_pretrained: ImageNet-pretrained torchvision ResNet50, full fine-tuning.
        CIFAR images are resized to 224x224 and normalized with ImageNet mean/std.
      - resnet18_pretrained: ImageNet-pretrained torchvision ResNet18, full fine-tuning.
        CIFAR images are resized to 224x224 and normalized with ImageNet mean/std.
      - cifar_resnet18: small CIFAR-style ResNet18 trained from scratch on 32x32 images.

    GPU support:
      - device='auto' uses CUDA when available, otherwise CPU.
      - device='cuda' forces CUDA and raises a clear error if CUDA is unavailable.
      - inputs/labels/model/loss all move to the same device.
      - DataLoader uses pin_memory when CUDA is active.
    """

    def __init__(
        self,
        random_state=42,
        epochs=30,
        batch_size=128,
        lr=0.003,
        weight_decay=5e-4,
        num_workers=0,
        device="auto",
        use_amp=True,
        verbose=True,
        backbone="resnet50_pretrained",
    ):
        self.random_state = random_state
        self.epochs = int(os.environ.get("ARGO_IMAGE_EPOCHS", epochs))
        self.batch_size = int(os.environ.get("ARGO_IMAGE_BATCH_SIZE", batch_size))
        self.lr = float(os.environ.get("ARGO_IMAGE_LR", lr))
        self.weight_decay = float(weight_decay)
        self.num_workers = int(os.environ.get("ARGO_IMAGE_NUM_WORKERS", num_workers))
        self.device = os.environ.get("ARGO_DEVICE", device)
        self.use_amp = bool(use_amp)
        self.verbose = bool(verbose)
        self.backbone = str(os.environ.get("ARGO_IMAGE_BACKBONE", backbone)).lower()
        self.model = None
        self.n_classes_ = None
        self.device_ = None

    def _import_torch(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise ImportError(
                "CIFAR10 end-to-end training requires torch. "
                "Install with: pip install torch torchvision"
            ) from exc
        return torch, nn, F, DataLoader, TensorDataset

    def _resolve_device(self, torch):
        requested = str(self.device or "auto").lower()
        if requested in {"auto", "gpu"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "You requested CUDA/GPU, but torch.cuda.is_available() is False. "
                    "Install a CUDA-enabled PyTorch build and NVIDIA driver, or run with --device cpu."
                )
            return torch.device(requested)
        return torch.device("cpu")

    def _prepare_x(self, X):
        X = np.asarray(X)
        if X.ndim == 2 and X.shape[1] == 3072:
            X = X.reshape(-1, 3, 32, 32)
        elif X.ndim == 4 and X.shape[-1] == 3:
            X = X.transpose(0, 3, 1, 2)
        elif X.ndim == 4 and X.shape[1] == 3:
            pass
        else:
            raise ValueError(
                f"Image model expects CIFAR images with shape (N,3,32,32), "
                f"(N,32,32,3), or flattened 3072; got {X.shape}."
            )
        original_dtype = X.dtype
        X = X.astype("float32", copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        # Robust image scaling:
        # - uint8 / 0..255 images should be divided by 255.
        # - already-normalized [0, 1] images with a few noisy pixels above 1
        #   should be clipped, not divided by 255.
        xmax = float(np.nanmax(X)) if X.size else 0.0
        if np.issubdtype(original_dtype, np.integer) or xmax > 10.0:
            X = X / 255.0
        X = np.clip(X, 0.0, 1.0)
        return np.ascontiguousarray(X)

    def _is_resnet50_pretrained(self):
        return self.backbone in {"resnet50_pretrained", "pretrained_resnet50", "resnet50", "imagenet_resnet50"}

    def _is_pretrained_resnet18(self):
        return self.backbone in {"resnet18_pretrained", "pretrained_resnet18", "resnet18", "imagenet_resnet18"}

    def _make_pretrained_resnet50(self, n_classes):
        torch, nn, F, _, _ = self._import_torch()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except Exception as exc:
            raise ImportError(
                "The resnet50_pretrained backbone requires torchvision. "
                "Install a torchvision build that matches your PyTorch/CUDA version, or run with "
                "--image-backbone cifar_resnet18."
            ) from exc

        try:
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
        except TypeError:
            # Older torchvision API.
            model = resnet50(pretrained=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ImageNet-pretrained ResNet50 weights. The first run may need internet "
                "access to download torchvision's ResNet50 checkpoint. If the machine is offline, either "
                "pre-download the checkpoint into the torch cache or run with --image-backbone cifar_resnet18."
            ) from exc

        model.fc = nn.Linear(model.fc.in_features, n_classes)

        # Option A = full end-to-end fine-tuning, not frozen embedding.
        for p in model.parameters():
            p.requires_grad = True
        return model

    def _make_pretrained_resnet18(self, n_classes):
        torch, nn, F, _, _ = self._import_torch()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise ImportError(
                "The resnet18_pretrained backbone requires torchvision. "
                "Install a torchvision build that matches your PyTorch/CUDA version, or run with "
                "--image-backbone cifar_resnet18."
            ) from exc

        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        except TypeError:
            model = resnet18(pretrained=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ImageNet-pretrained ResNet18 weights. The first run may need internet "
                "access to download torchvision's ResNet18 checkpoint. If the machine is offline, either "
                "pre-download the checkpoint into the torch cache or run with --image-backbone cifar_resnet18."
            ) from exc

        model.fc = nn.Linear(model.fc.in_features, n_classes)

        for p in model.parameters():
            p.requires_grad = True
        return model

    def _make_cifar_resnet18(self, n_classes):
        torch, nn, F, _, _ = self._import_torch()

        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, in_planes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes),
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = out + self.shortcut(x)
                return F.relu(out)

        class CIFARResNet18(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.in_planes = 64
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
                self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
                self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
                self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
                self.fc = nn.Linear(512, num_classes)

            def _make_layer(self, block, planes, blocks, stride):
                strides = [stride] + [1] * (blocks - 1)
                layers = []
                for s in strides:
                    layers.append(block(self.in_planes, planes, s))
                    self.in_planes = planes * block.expansion
                return nn.Sequential(*layers)

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                return self.fc(out)

        return CIFARResNet18(n_classes)

    def _make_model(self, n_classes):
        if self._is_resnet50_pretrained():
            return self._make_pretrained_resnet50(n_classes)
        if self._is_pretrained_resnet18():
            return self._make_pretrained_resnet18(n_classes)
        if self.backbone in {"cifar_resnet18", "resnet18_cifar", "cifar18"}:
            return self._make_cifar_resnet18(n_classes)
        raise ValueError(
            f"Unknown image backbone '{self.backbone}'. Supported: "
            "resnet50_pretrained, resnet18_pretrained, cifar_resnet18."
        )

    def _augment_batch(self, xb, torch):
        mask = torch.rand(xb.size(0), device=xb.device) < 0.5
        xb = xb.clone()
        xb[mask] = torch.flip(xb[mask], dims=[3])

        xb = torch.nn.functional.pad(xb, (4, 4, 4, 4), mode="reflect")
        n = xb.size(0)
        tops = torch.randint(0, 9, (n,), device=xb.device)
        lefts = torch.randint(0, 9, (n,), device=xb.device)
        cropped = [xb[i:i + 1, :, int(tops[i]):int(tops[i]) + 32, int(lefts[i]):int(lefts[i]) + 32] for i in range(n)]
        return torch.cat(cropped, dim=0)

    def _normalize_cifar(self, xb, torch):
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=xb.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=xb.device).view(1, 3, 1, 1)
        return (xb - mean) / std

    def _normalize_imagenet(self, xb, torch):
        mean = torch.tensor([0.485, 0.456, 0.406], device=xb.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=xb.device).view(1, 3, 1, 1)
        return (xb - mean) / std

    def _preprocess_batch(self, xb, torch, train=False):
        if train:
            xb = self._augment_batch(xb, torch)

        if self._is_resnet50_pretrained() or self._is_pretrained_resnet18():
            # Keep torchvision's ImageNet ResNet architecture intact.
            # CIFAR10 32x32 images are resized to 224x224 before ImageNet normalization.
            xb = torch.nn.functional.interpolate(
                xb,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            return self._normalize_imagenet(xb, torch)

        return self._normalize_cifar(xb, torch)

    def _make_scaler(self, torch, amp_enabled):
        if not amp_enabled:
            return None
        if hasattr(torch, "amp"):
            try:
                return torch.amp.GradScaler("cuda", enabled=True)
            except Exception:
                pass
        return torch.cuda.amp.GradScaler(enabled=True)

    def _autocast_context(self, torch, amp_enabled):
        if not amp_enabled:
            return nullcontext()
        if hasattr(torch, "amp"):
            try:
                return torch.amp.autocast("cuda", enabled=True)
            except TypeError:
                try:
                    return torch.amp.autocast(device_type="cuda", enabled=True)
                except TypeError:
                    pass
        return torch.cuda.amp.autocast(enabled=True)

    def fit(self, X, y):
        torch, nn, F, DataLoader, TensorDataset = self._import_torch()

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.benchmark = True

        X = self._prepare_x(X)
        y = np.asarray(y, dtype=np.int64)
        # CIFAR10 has 10 classes. Keeping at least 10 outputs avoids shape changes if a
        # small cleaned subset temporarily misses one class.
        self.n_classes_ = int(max(np.max(y) + 1, 10))

        device = self._resolve_device(torch)
        self.device_ = device
        use_cuda = device.type == "cuda"
        self.model = self._make_model(self.n_classes_).to(device)

        if self.verbose:
            dev_name = torch.cuda.get_device_name(device) if use_cuda else "CPU"
            print(
                f"[TorchCIFARClassifier] device={device} ({dev_name}) | "
                f"backbone={self.backbone} | epochs={self.epochs} "
                f"batch_size={self.batch_size} num_workers={self.num_workers} lr={self.lr}"
            )

        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=use_cuda,
            persistent_workers=(self.num_workers > 0),
            drop_last=False,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs))
        amp_enabled = bool(self.use_amp and use_cuda)
        scaler = self._make_scaler(torch, amp_enabled)

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                xb = self._preprocess_batch(xb, torch, train=True)

                optimizer.zero_grad(set_to_none=True)
                with self._autocast_context(torch, amp_enabled):
                    logits = self.model(xb)
                    loss = criterion(logits, yb)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            scheduler.step()

        return self

    def _predict_logits(self, X, batch_size=None):
        torch, nn, F, DataLoader, TensorDataset = self._import_torch()
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_x(X)
        bs = int(batch_size or self.batch_size)
        device = self.device_
        use_cuda = getattr(device, "type", str(device)) == "cuda"
        ds = TensorDataset(torch.from_numpy(X).float())
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=use_cuda)

        logits_all = []
        self.model.eval()
        amp_enabled = bool(self.use_amp and use_cuda)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=use_cuda)
                xb = self._preprocess_batch(xb, torch, train=False)
                with self._autocast_context(torch, amp_enabled):
                    logits = self.model(xb)
                logits_all.append(logits.detach().cpu().numpy())
        return np.vstack(logits_all)

    def predict_proba(self, X):
        logits = self._predict_logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

    def predict_proba_ensemble(self, X, n_ensemble=5):
        """
        Run n_ensemble forward passes with different random seeds and return
        (ensemble_proba, disagreement_score).

        disagreement_score[i] = fraction of ensemble members whose argmax
        differs from the ensemble's argmax.  0 = perfect agreement, 1 = total
        disagreement across all classes.
        """
        torch, nn, _, _, _ = self._import_torch()
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._predict_prep(X)
        device = self.device_
        use_cuda = getattr(device, "type", str(device)) == "cuda"
        amp_enabled = bool(self.use_amp and use_cuda)
        n_classes = self.model.fc.out_features if hasattr(self.model, "fc") else self.n_classes_
        n_samples = X.shape[0]

        all_logits = np.zeros((n_ensemble, n_samples, n_classes), dtype=np.float32)

        for k in range(n_ensemble):
            torch.manual_seed(self.random_state + k)
            np.random.seed(self.random_state + k)
            if use_cuda:
                torch.cuda.manual_seed_all(self.random_state + k)

            X_t = torch.from_numpy(X).float()
            ds = torch.utils.data.TensorDataset(X_t)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

            logits_list = []
            self.model.eval()
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(device, non_blocking=use_cuda)
                    xb = self._preprocess_batch(xb, torch, train=False)
                    with self._autocast_context(torch, amp_enabled):
                        logits = self.model(xb)
                    logits_list.append(logits.detach().cpu().numpy())
            logits_k = np.vstack(logits_list)
            all_logits[k] = logits_k

        # Average logits -> ensemble probability
        avg_logits = all_logits.mean(axis=0)
        avg_logits = avg_logits - avg_logits.max(axis=1, keepdims=True)
        ensemble_proba = np.exp(avg_logits)
        ensemble_proba = ensemble_proba / (ensemble_proba.sum(axis=1, keepdims=True) + 1e-12)

        # Disagreement: fraction of members whose argmax != ensemble argmax
        ensemble_argmax = ensemble_proba.argmax(axis=1)
        argmax_per_member = all_logits.argmax(axis=2)  # (n_ensemble, n_samples)
        disagreement = (argmax_per_member != ensemble_argmax).mean(axis=0).astype(np.float32)

        return ensemble_proba, disagreement

    def _predict_prep(self, X):
        X = np.asarray(X)
        if X.ndim == 2 and X.shape[1] == 3072:
            X = X.reshape(-1, 3, 32, 32)
        elif X.ndim == 4 and X.shape[-1] == 3:
            X = X.transpose(0, 3, 1, 2)
        X = X.astype("float32", copy=False)
        X = np.clip(X, 0.0, 1.0)
        return np.ascontiguousarray(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return float(accuracy_score(y, self.predict(X)))

    def losses(self, X, y):
        proba = self.predict_proba(X)
        y = np.asarray(y, dtype=int)
        clipped = np.clip(proba, 1e-8, 1.0)
        return -np.log(clipped[np.arange(len(y)), y])


class DownstreamModel:
    """
    Dataset-aware downstream classifier.

    model_type:
      - tabular: RandomForest
      - text: sklearn SGD logistic classifier
      - image: GPU-aware PyTorch image classifier
      - auto: choose by feature shape
    """

    def __init__(
        self,
        random_state=42,
        model_type="auto",
        device="auto",
        image_epochs=30,
        image_batch_size=128,
        image_num_workers=0,
        image_lr=0.003,
        image_backbone="resnet50_pretrained",
    ):
        self.random_state = random_state
        self.model_type = model_type
        self.device = device
        self.image_epochs = image_epochs
        self.image_batch_size = image_batch_size
        self.image_num_workers = image_num_workers
        self.image_lr = image_lr
        self.image_backbone = image_backbone
        self.model = None

    def _make_image_model(self):
        return TorchCIFARClassifier(
            random_state=self.random_state,
            device=self.device,
            epochs=self.image_epochs,
            batch_size=self.image_batch_size,
            num_workers=self.image_num_workers,
            lr=self.image_lr,
            backbone=self.image_backbone,
        )

    def _make_model(self, X):
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        mt = (self.model_type or "auto").lower()

        if mt in {"image", "cifar", "cifar10"}:
            return self._make_image_model()

        if mt in {"text", "linear"}:
            return SGDClassifier(loss="log_loss", max_iter=300, tol=1e-3, alpha=1e-4, random_state=self.random_state)

        if mt in {"tabular", "adult", "synthetic"}:
            return RandomForestClassifier(n_estimators=120, max_depth=None, random_state=self.random_state, n_jobs=-1, warm_start=True)

        if len(X.shape) == 4 and X.shape[1] == 3:
            return self._make_image_model()

        if n_features > 300:
            return SGDClassifier(loss="log_loss", max_iter=300, tol=1e-3, alpha=1e-4, random_state=self.random_state)

        return RandomForestClassifier(n_estimators=120, max_depth=None, random_state=self.random_state, n_jobs=-1, warm_start=True)

    def fit(self, X, y):
        self.model = self._make_model(X)
        self.model.fit(X, y)
        # Record n_estimators so DifferentialDownstreamModel can use it for warm_start.
        if isinstance(self.model, RandomForestClassifier):
            self._rf_n_estimators = self.model.n_estimators
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def losses(self, X, y):
        if hasattr(self.model, "losses"):
            return self.model.losses(X, y)
        proba = self.predict_proba(X)
        y = np.asarray(y, dtype=int)
        clipped = np.clip(proba, 1e-8, 1.0)
        return -np.log(clipped[np.arange(len(y)), y])

    def predict_proba_ensemble(self, X, n_ensemble=5):
        if hasattr(self.model, "predict_proba_ensemble"):
            return self.model.predict_proba_ensemble(X, n_ensemble=n_ensemble)
        # Fallback for non-Torch models: just return single-model proba + zero disagreement
        proba = self.predict_proba(X)
        return proba, np.zeros(len(X))



# -----------------------------------------------------------------------------
# DifferentialDownstreamModel
#
# Dirty-region replay for tabular/text downstream classifiers.
# This intentionally avoids RandomForest warm_start as a dirty-only update,
# because sklearn forests cannot update or unlearn individual rows. Stage 1
# freezes high-confidence clean rows, trains an incremental SGD classifier on
# those rows once, then each cleaning step rebuilds from that clean snapshot and
# replays only the active dirty region.
# -----------------------------------------------------------------------------

class DifferentialDownstreamModel:
    """
    Drop-in model wrapper for Stage-1-clean / dirty-only downstream training.

    dirty_scope="active" supports remove/relabel/repair/augment because deleted
    dirty rows disappear from the replay set. dirty_scope="current" is faster
    but approximate: it only updates rows touched in the current step and cannot
    unlearn an earlier dirty update without a replay.
    """

    def __init__(
        self,
        base_model,
        clean_indices=None,
        use_differential=True,
        clean_epochs=5,
        dirty_epochs=1,
        dirty_scope="active",
        alpha=1e-4,
        random_state=None,
        verbose=False,
        image_ft_epochs=3,
        image_ft_lr_factor=0.05,
        image_ft_batch_size=None,
    ):
        self._base = base_model
        self._use_differential = bool(use_differential)
        self._verbose = bool(verbose)
        self._clean_epochs = max(1, int(clean_epochs))
        self._dirty_epochs = max(1, int(dirty_epochs))
        self._dirty_scope = str(dirty_scope or "active").lower()
        if self._dirty_scope not in {"active", "current"}:
            raise ValueError("dirty_scope must be 'active' or 'current'")
        self._alpha = float(alpha)
        self._random_state = (
            int(random_state)
            if random_state is not None
            else int(getattr(base_model, "random_state", 42))
        )
        self.model_type = getattr(base_model, "model_type", "auto")
        self._clean_indices = (
            set(int(i) for i in clean_indices)
            if clean_indices is not None
            else set()
        )
        self._classes = None
        self._clean_mask = None
        self._clean_snapshot = None
        self._model = None
        self._is_fitted = False
        self._last_dirty_indices = np.array([], dtype=int)
        self._last_active_dirty_count = 0

    def _is_incremental_supported(self):
        mt = str(self.model_type or "auto").lower()
        return mt in {"auto", "tabular", "adult", "synthetic", "text", "linear"}

    def _new_incremental_estimator(self):
        return SGDClassifier(
            loss="log_loss",
            alpha=self._alpha,
            max_iter=1,
            tol=None,
            random_state=self._random_state,
        )

    @staticmethod
    def _safe_rows(X, idx):
        return X[idx]

    @staticmethod
    def _normalise_clean_mask(n, clean_indices=None, clean_mask=None):
        if clean_mask is not None:
            mask = np.asarray(clean_mask, dtype=bool).copy()
            if len(mask) != n:
                raise ValueError(
                    f"clean_mask length {len(mask)} does not match training size {n}"
                )
            return mask
        mask = np.zeros(n, dtype=bool)
        if clean_indices is not None:
            idx = np.asarray(list(clean_indices), dtype=int)
            idx = idx[(idx >= 0) & (idx < n)]
            mask[idx] = True
        return mask

    def _partial_fit_epochs(self, estimator, X_part, y_part, epochs, classes=None):
        if len(y_part) == 0:
            return estimator
        first = not hasattr(estimator, "classes_")
        for _ in range(max(1, int(epochs))):
            if first:
                estimator.partial_fit(X_part, y_part, classes=classes)
                first = False
            else:
                estimator.partial_fit(X_part, y_part)
        return estimator

    def _fallback_full_fit(self, X, y):
        if self._verbose:
            print("[DiffModel] fallback full fit for unsupported model type")
        self._base.fit(X, y)
        self._model = self._base.model
        self._is_fitted = True
        return self

    def pretrain_clean(self, X, y, clean_indices=None, clean_mask=None):
        y = np.asarray(y)
        n = len(y)
        if not self._use_differential or not self._is_incremental_supported():
            return self._fallback_full_fit(X, y)
        if clean_indices is None:
            clean_indices = self._clean_indices
        self._clean_mask = self._normalise_clean_mask(n, clean_indices, clean_mask)
        self._classes = np.arange(int(np.max(y)) + 1, dtype=int)
        clean_idx = np.where(self._clean_mask)[0]
        if len(clean_idx) == 0:
            clean_idx = np.arange(n, dtype=int)
            self._clean_mask = np.ones(n, dtype=bool)
        X_clean = self._safe_rows(X, clean_idx)
        y_clean = y[clean_idx]
        est = self._new_incremental_estimator()
        self._partial_fit_epochs(
            est,
            X_clean,
            y_clean,
            epochs=self._clean_epochs,
            classes=self._classes,
        )
        self._clean_snapshot = deepcopy(est)
        self._model = deepcopy(est)
        self._is_fitted = True
        self._last_dirty_indices = np.array([], dtype=int)
        self._last_active_dirty_count = int(n - len(clean_idx))
        if self._verbose:
            print(
                f"[DiffModel] clean pretrain | clean={len(clean_idx)} "
                f"dirty_initial={n - len(clean_idx)} epochs={self._clean_epochs}"
            )
        return self

    def fit(self, X, y, dirty_indices=None, clean_mask=None, action_name=None, **kwargs):
        y = np.asarray(y)
        n = len(y)
        if not self._use_differential or not self._is_incremental_supported():
            return self._fallback_full_fit(X, y)
        if self._clean_snapshot is None:
            self.pretrain_clean(X, y, clean_mask=clean_mask)
        if clean_mask is not None:
            self._clean_mask = self._normalise_clean_mask(n, clean_mask=clean_mask)
        elif self._clean_mask is None or len(self._clean_mask) != n:
            self._clean_mask = np.zeros(n, dtype=bool)
        current_dirty = (
            np.asarray(dirty_indices, dtype=int)
            if dirty_indices is not None
            else np.array([], dtype=int)
        )
        current_dirty = current_dirty[(current_dirty >= 0) & (current_dirty < n)]
        if self._dirty_scope == "active":
            dirty_idx = np.where(~self._clean_mask)[0]
        else:
            dirty_idx = current_dirty
        est = deepcopy(self._clean_snapshot)
        if len(dirty_idx) > 0:
            self._partial_fit_epochs(
                est,
                self._safe_rows(X, dirty_idx),
                y[dirty_idx],
                epochs=self._dirty_epochs,
            )
        self._model = est
        self._is_fitted = True
        self._last_dirty_indices = current_dirty
        self._last_active_dirty_count = int(len(dirty_idx))
        if self._verbose:
            print(
                f"[DiffModel] dirty replay | scope={self._dirty_scope} "
                f"current={len(current_dirty)} active={len(dirty_idx)} "
                f"epochs={self._dirty_epochs} action={action_name}"
            )
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        pred = self._model.predict(X)
        n_classes = len(self._classes) if self._classes is not None else int(np.max(pred) + 1)
        proba = np.zeros((len(pred), n_classes), dtype=float)
        proba[np.arange(len(pred)), pred.astype(int)] = 1.0
        return proba

    def score(self, X, y):
        return self._model.score(X, y)

    def losses(self, X, y):
        proba = self.predict_proba(X)
        y = np.asarray(y, dtype=int)
        clipped = np.clip(proba, 1e-8, 1.0)
        return -np.log(clipped[np.arange(len(y)), y])

    def predict_proba_ensemble(self, X, n_ensemble=5):
        proba = self.predict_proba(X)
        return proba, np.zeros(len(X))

    @property
    def model(self):
        return self._model

    @property
    def is_differential_enabled(self):
        return self._use_differential

    @property
    def clean_indices(self):
        return self._clean_indices

    @property
    def last_active_dirty_count(self):
        return self._last_active_dirty_count
