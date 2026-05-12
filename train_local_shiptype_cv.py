from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
from torch.utils.data import DataLoader, Dataset

TORCHVISION_IMPORT_ERROR = None
try:
    from torchvision import models as tv_models
except Exception as exc:  # pragma: no cover - environment-specific
    tv_models = None
    TORCHVISION_IMPORT_ERROR = exc

SKLEARN_IMPORT_ERROR = None
try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        top_k_accuracy_score,
    )
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
except Exception as exc:  # pragma: no cover - environment-specific
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    top_k_accuracy_score = None
    StratifiedGroupKFold = None
    StratifiedKFold = None
    SKLEARN_IMPORT_ERROR = exc

try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
except Exception:
    pass


CLASS_SCHEMES: Dict[str, List[str]] = {
    "6": ["Cargo", "Tanker", "Fishing", "Passenger", "Sailing", "Pleasure"],
    "5": ["Cargo", "Tanker", "Fishing", "Passenger", "Leisure"],
    "4": ["CargoTanker", "Fishing", "Passenger", "Leisure"],
}

SCHEME_LABEL_MAP: Dict[str, Dict[str, str | None]] = {
    "6": {
        "Cargo": "Cargo",
        "Tanker": "Tanker",
        "Fishing": "Fishing",
        "Passenger": "Passenger",
        "Sailing": "Sailing",
        "Pleasure": "Pleasure",
    },
    "5": {
        "Cargo": "Cargo",
        "Tanker": "Tanker",
        "Fishing": "Fishing",
        "Passenger": "Passenger",
        "Sailing": "Leisure",
        "Pleasure": "Leisure",
    },
    "4": {
        "Cargo": "CargoTanker",
        "Tanker": "CargoTanker",
        "Fishing": "Fishing",
        "Passenger": "Passenger",
        "Sailing": "Leisure",
        "Pleasure": "Leisure",
    },
}

SUPPORTED_SOURCE_CLASSES = {"Cargo", "Tanker", "Fishing", "Passenger", "Sailing", "Pleasure"}


def format_hms(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_training_defaults(config_path: Path | None) -> Dict[str, Any]:
    def as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        if value is None:
            return bool(default)
        return bool(value)

    def as_float_list(value: Any, default: List[float]) -> List[float]:
        if not isinstance(value, (list, tuple)) or len(value) != len(default):
            return [float(v) for v in default]
        return [as_float(item, fallback) for item, fallback in zip(value, default)]

    defaults = {
        "epochs": 30,
        "batch_size": 64,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "num_workers": 4,
        "image_size": 128,
        "seed": 42,
        "early_stopping_patience": 7,
        "mixed_precision": True,
        "train_val_test_split": [0.8, 0.1, 0.1],
        "bands_order": ["B02", "B03", "B04", "B08"],
    }
    if not config_path or not config_path.exists():
        return defaults

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tr = cfg.get("training", {})
    defaults.update({
        "epochs": as_int(tr.get("epochs", defaults["epochs"]), defaults["epochs"]),
        "batch_size": as_int(tr.get("batch_size", defaults["batch_size"]), defaults["batch_size"]),
        "learning_rate": as_float(tr.get("learning_rate", defaults["learning_rate"]), defaults["learning_rate"]),
        "weight_decay": as_float(tr.get("weight_decay", defaults["weight_decay"]), defaults["weight_decay"]),
        "num_workers": as_int(tr.get("num_workers", defaults["num_workers"]), defaults["num_workers"]),
        "image_size": as_int(tr.get("image_size", defaults["image_size"]), defaults["image_size"]),
        "seed": as_int(tr.get("seed", defaults["seed"]), defaults["seed"]),
        "early_stopping_patience": as_int(
            tr.get("early_stopping_patience", defaults["early_stopping_patience"]),
            defaults["early_stopping_patience"],
        ),
        "mixed_precision": as_bool(tr.get("mixed_precision", defaults["mixed_precision"]), defaults["mixed_precision"]),
        "train_val_test_split": as_float_list(
            tr.get("train_val_test_split", defaults["train_val_test_split"]),
            defaults["train_val_test_split"],
        ),
        "bands_order": cfg.get("bands_order", defaults["bands_order"]),
    })
    return defaults


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    project_root = repo_root.parent
    default_dataset = project_root / "dataset_ship_type"
    default_config = repo_root / "config.yaml"
    default_output = repo_root / "results_local"

    p = argparse.ArgumentParser(description="Local CV training for 6/5/4-class AIS-Sentinel-2 ship-type models.")
    p.add_argument("--dataset-root", type=Path, default=default_dataset, help="Root folder with class subdirectories.")
    p.add_argument("--config", type=Path, default=default_config, help="Optional config.yaml for default hyperparameters.")
    p.add_argument("--class-scheme", choices=["6", "5", "4"], required=True, help="Class scheme to train.")
    p.add_argument(
        "--arch",
        choices=["resnet18", "resnet34", "densenet121", "efficientnet_b0"],
        default="resnet18",
        help="Backbone architecture.",
    )
    p.add_argument(
        "--split-mode",
        choices=["patch", "mmsi_grouped"],
        default="patch",
        help="Cross-validation split mode. 'mmsi_grouped' keeps the same MMSI out of train/val/test at the same time.",
    )
    p.add_argument(
        "--exclude-conflicting-mmsi",
        action="store_true",
        help="Exclude MMSI values that appear under more than one class in the requested class scheme.",
    )
    p.add_argument("--output-root", type=Path, default=default_output, help="Directory to write training outputs.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument(
        "--bands-order",
        nargs="+",
        default=None,
        help="Override input band order, e.g. B02 B03 B04 for RGB-only training.",
    )
    p.add_argument("--folds", type=int, default=None, help="Override inferred number of CV folds.")
    p.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision even if CUDA is available.")
    p.add_argument("--max-samples-per-class", type=int, default=None, help="Optional cap for faster debugging.")
    p.add_argument("--dry-run", action="store_true", help="Only inspect dataset and write summary, do not train.")
    return p


def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_source_label_from_path(path: Path) -> str | None:
    parts = set(path.as_posix().split("/"))
    for cls in SUPPORTED_SOURCE_CLASSES:
        if cls in parts:
            return cls
    parent = path.parent.name
    if parent in SUPPORTED_SOURCE_CLASSES:
        return parent
    return None


def map_label(source_label: str, class_scheme: str) -> str | None:
    return SCHEME_LABEL_MAP[class_scheme].get(source_label)


def extract_mmsi_from_path(path_str: str) -> str | None:
    stem = Path(path_str).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    mmsi = parts[1].strip()
    if not mmsi.isdigit():
        return None
    return mmsi


def collect_samples(dataset_root: Path, class_scheme: str, max_samples_per_class: int | None) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    classes = CLASS_SCHEMES[class_scheme]
    gathered: dict[str, list[str]] = defaultdict(list)
    dropped_other_classes: Counter[str] = Counter()
    skipped_non_tif = 0

    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".tif", ".tiff"}:
            skipped_non_tif += 1
            continue
        if "tci" in path.name.lower():
            continue

        source_label = infer_source_label_from_path(path)
        if source_label is None:
            dropped_other_classes[path.parent.name] += 1
            continue

        mapped = map_label(source_label, class_scheme)
        if mapped is None:
            dropped_other_classes[source_label] += 1
            continue
        gathered[mapped].append(str(path))

    samples_all: list[tuple[str, int]] = []
    kept_counts: dict[str, int] = {}
    for class_name in classes:
        paths = gathered.get(class_name, [])
        random.shuffle(paths)
        if max_samples_per_class is not None:
            paths = paths[:max_samples_per_class]
        kept_counts[class_name] = len(paths)
        label_idx = classes.index(class_name)
        samples_all.extend((p, label_idx) for p in paths)

    meta = {
        "kept_counts": kept_counts,
        "dropped_other_classes": dict(dropped_other_classes),
        "skipped_non_tif": skipped_non_tif,
        "classes": classes,
    }
    return samples_all, meta


def exclude_conflicting_mmsi_samples(
    samples_all: list[tuple[str, int]],
    classes: list[str],
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    mmsi_to_labels: dict[str, set[int]] = defaultdict(set)
    mmsi_to_paths: dict[str, list[tuple[str, int]]] = defaultdict(list)
    missing_mmsi_count = 0

    for path_str, label_idx in samples_all:
        mmsi = extract_mmsi_from_path(path_str)
        if mmsi is None:
            missing_mmsi_count += 1
            continue
        mmsi_to_labels[mmsi].add(label_idx)
        mmsi_to_paths[mmsi].append((path_str, label_idx))

    conflicting_mmsi = sorted([mmsi for mmsi, lbls in mmsi_to_labels.items() if len(lbls) > 1])
    conflicting_set = set(conflicting_mmsi)
    filtered_samples = [
        (path_str, label_idx)
        for path_str, label_idx in samples_all
        if extract_mmsi_from_path(path_str) not in conflicting_set
    ]

    filtered_counts = Counter(label_idx for _, label_idx in filtered_samples)
    kept_counts = {cls: int(filtered_counts.get(idx, 0)) for idx, cls in enumerate(classes)}

    combo_counter: Counter[str] = Counter()
    for mmsi in conflicting_mmsi:
        combo = " | ".join(sorted(classes[idx] for idx in mmsi_to_labels[mmsi]))
        combo_counter[combo] += 1

    excluded_sample_count = len(samples_all) - len(filtered_samples)
    stats = {
        "applied": True,
        "conflicting_mmsi_count": len(conflicting_mmsi),
        "excluded_sample_count": excluded_sample_count,
        "missing_mmsi_count": missing_mmsi_count,
        "conflicting_mmsi_examples": conflicting_mmsi[:20],
        "conflicting_label_combinations": dict(combo_counter),
        "kept_counts_after_exclusion": kept_counts,
    }
    return filtered_samples, stats


def _to_chw(arr: np.ndarray, expected_channels: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected TIFF shape: {arr.shape}")

    axes = list(arr.shape)
    if expected_channels in axes:
        ch_axis = axes.index(expected_channels)
    elif any(s in (3, 4, 8, 12) for s in axes):
        if axes[2] in (3, 4, 8, 12):
            ch_axis = 2
        elif axes[0] in (3, 4, 8, 12):
            ch_axis = 0
        else:
            ch_axis = int(np.argmin(axes))
    else:
        ch_axis = int(np.argmin(axes))

    if ch_axis == 0:
        chw = arr
    elif ch_axis == 2:
        chw = np.transpose(arr, (2, 0, 1))
    elif ch_axis == 1:
        chw = np.transpose(arr, (1, 0, 2))
    else:
        perm = [ch_axis] + [i for i in range(3) if i != ch_axis]
        chw = np.transpose(arr, perm)
    return chw


def read_tif(path: str, expected_channels: int) -> np.ndarray:
    arr = tiff.imread(path)
    chw = _to_chw(arr, expected_channels)
    if chw.shape[0] > expected_channels and expected_channels > 0:
        chw = chw[:expected_channels, ...]
    chw = chw.astype(np.float32)
    chw = np.clip(chw, 0, None)
    return chw


def resize_to_square(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2
        c, _, _ = arr.shape
        out = np.zeros((c, size, size), dtype=arr.dtype)
        for i in range(c):
            out[i] = cv2.resize(arr[i], (size, size), interpolation=cv2.INTER_NEAREST)
        return out
    except Exception:
        c, h, w = arr.shape
        if h == size and w == size:
            return arr
        out = np.zeros((c, size, size), dtype=arr.dtype)
        y0 = max(0, (h - size) // 2)
        x0 = max(0, (w - size) // 2)
        y1 = min(h, y0 + size)
        x1 = min(w, x0 + size)
        crop = arr[:, y0:y1, x0:x1]
        out[:, :crop.shape[1], :crop.shape[2]] = crop
        return out


def adapt_conv2d_input_channels(old_conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_ch,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
        if in_ch == old_conv.in_channels:
            new_conv.weight.copy_(old_conv.weight)
        elif in_ch > old_conv.in_channels:
            new_conv.weight[:, :old_conv.in_channels].copy_(old_conv.weight)
            extra_mean = old_conv.weight.mean(dim=1, keepdim=True)
            repeat_count = in_ch - old_conv.in_channels
            new_conv.weight[:, old_conv.in_channels:].copy_(extra_mean.repeat(1, repeat_count, 1, 1))
        else:
            new_conv.weight.copy_(old_conv.weight[:, :in_ch])
    return new_conv


class S2Dataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        image_size: int,
        norm_stats: Dict[str, np.ndarray],
        expected_channels: int,
        train: bool = True,
    ):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.low = norm_stats["low"].astype(np.float32)
        self.high = norm_stats["high"].astype(np.float32)
        self.expected_channels = expected_channels

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            x = x[:, :, ::-1]
        if random.random() < 0.5:
            x = x[:, ::-1, :]
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            x = np.rot90(x, k, axes=(1, 2)).copy()
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-0.1, 0.1)
            beta = random.uniform(-0.03, 0.03)
            x = np.clip(x * alpha + beta, 0.0, None)
        return x

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        c = x.shape[0]
        low = self.low[:c, None, None]
        high = self.high[:c, None, None]
        x = np.clip(x, low, high)
        denom = np.maximum(high - low, 1e-6)
        return (x - low) / denom

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = read_tif(path, expected_channels=self.expected_channels)
        x = resize_to_square(x, self.image_size)
        if self.train:
            x = self._augment(x)
        x = self._normalize(x)
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(label, dtype=torch.long)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class SmallResNet(nn.Module):
    def __init__(self, layers: list[int], in_ch: int, num_classes: int):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(arch: str, in_ch: int, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        return SmallResNet([2, 2, 2, 2], in_ch, num_classes)
    if arch == "resnet34":
        return SmallResNet([3, 4, 6, 3], in_ch, num_classes)
    if arch == "densenet121":
        if tv_models is None:
            raise RuntimeError(
                "torchvision is required for DenseNet121 but is not available in the current Python environment."
            ) from TORCHVISION_IMPORT_ERROR
        try:
            model = tv_models.densenet121(weights=None)
        except TypeError:
            model = tv_models.densenet121(pretrained=False)
        model.features.conv0 = adapt_conv2d_input_channels(model.features.conv0, in_ch)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        if tv_models is None:
            raise RuntimeError(
                "torchvision is required for EfficientNet-B0 but is not available in the current Python environment."
            ) from TORCHVISION_IMPORT_ERROR
        try:
            model = tv_models.efficientnet_b0(weights=None)
        except TypeError:
            model = tv_models.efficientnet_b0(pretrained=False)
        model.features[0][0] = adapt_conv2d_input_channels(model.features[0][0], in_ch)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


def plot_curves(history: Dict[str, List[float]], out_png: Path) -> None:
    plt.figure()
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred, classes, out_png: Path, normalize: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-6, None)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (norm)" if normalize else ""))
    plt.colorbar()
    tick = np.arange(len(classes))
    plt.xticks(tick, classes, rotation=45, ha="right")
    plt.yticks(tick, classes)
    fmt = ".2f" if normalize else "d"
    thr = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                     color="white" if cm[i, j] > thr else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def classification_report_to_df_dict(y_true, y_pred, classes):
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )


def compute_percentile_stats(
    samples: List[Tuple[str, int]],
    image_size: int,
    per_class_limit: int = 200,
    expected_channels: int = 4,
) -> Dict[str, np.ndarray]:
    by_class = defaultdict(list)
    for p, y in samples:
        by_class[y].append(p)
    arrays = []
    for _, paths in by_class.items():
        random.shuffle(paths)
        for path in paths[:per_class_limit]:
            arr = read_tif(path, expected_channels=expected_channels)
            arr = resize_to_square(arr, image_size)
            arrays.append(arr)
    if not arrays:
        raise RuntimeError("No samples available to compute normalization stats.")
    c = arrays[0].shape[0]
    lows, highs = [], []
    for i in range(c):
        vals = np.concatenate([a[i].ravel() for a in arrays])
        low = np.percentile(vals, 2.0)
        high = np.percentile(vals, 98.0)
        if high <= low:
            high = low + 1e-3
        lows.append(low)
        highs.append(high)
    return {"low": np.array(lows, dtype=np.float32), "high": np.array(highs, dtype=np.float32)}


def make_loaders(
    x_train, x_val, x_test, image_size, norm_stats,
    batch_size, num_workers, expected_channels, device
):
    if device.type != "cuda":
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    else:
        pin_memory = True
        persistent_workers = num_workers > 0

    ds_train = S2Dataset(x_train, image_size, norm_stats, expected_channels, train=True)
    ds_val = S2Dataset(x_val, image_size, norm_stats, expected_channels, train=False)
    ds_test = S2Dataset(x_test, image_size, norm_stats, expected_channels, train=False)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return dl_train, dl_val, dl_test


def train_one_fold(
    fold_dir: Path,
    model: nn.Module,
    device: torch.device,
    dl_train,
    dl_val,
    dl_test,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stopping_patience: int,
    mixed_precision: bool,
    class_weights: torch.Tensor,
) -> Dict[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    mp_enabled = mixed_precision and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=mp_enabled)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for _epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0
        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=mp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            if mp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss = running_loss / max(1, n_train)

        model.eval()
        val_loss = 0.0
        n_val = 0
        correct = 0
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=mp_enabled):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
        val_loss = val_loss / max(1, n_val)
        val_acc = correct / max(1, n_val)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {_epoch:02d}/{epochs} - "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    plot_curves(history, fold_dir / "curves.png")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            y_true_val.extend(y.numpy().tolist())
            y_pred_val.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    y_true_test, y_pred_test, y_prob_test = [], [], []
    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            y_true_test.extend(y.numpy().tolist())
            y_pred_test.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_prob_test.extend(probs.cpu().numpy().tolist())

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "y_true_val": y_true_val,
        "y_pred_val": y_pred_val,
        "y_true_test": y_true_test,
        "y_pred_test": y_pred_test,
        "y_prob_test": y_prob_test,
        "best_state": best_state,
    }


def infer_fold_count(split: list[float], override: int | None) -> int:
    if override is not None:
        return override
    train_frac, val_frac, test_frac = split
    if abs(val_frac - test_frac) < 1e-6 and val_frac > 0:
        k = int(round(1.0 / val_frac))
    else:
        k = 5
    return max(3, min(20, k))


def build_fold_test_indices(
    x_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k: int,
    seed: int,
    split_mode: str,
) -> list[np.ndarray]:
    if split_mode == "patch":
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [test_idx for _, test_idx in skf.split(x_idx, y)]

    if split_mode == "mmsi_grouped":
        if StratifiedGroupKFold is None:
            raise RuntimeError(
                "StratifiedGroupKFold is not available in the installed scikit-learn version. "
                "Update scikit-learn to use MMSI-grouped splitting."
            )
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return [test_idx for _, test_idx in sgkf.split(x_idx, y, groups)]

    raise ValueError(f"Unsupported split mode: {split_mode}")


def validate_group_separation(
    samples_all: list[tuple[str, int]],
    fold_test_indices: list[np.ndarray],
    groups: np.ndarray,
    split_mode: str,
) -> dict[str, Any]:
    fold_summaries: list[dict[str, Any]] = []
    if split_mode != "mmsi_grouped":
        return {
            "split_mode": split_mode,
            "group_separation_checked": False,
            "fold_group_overlap": fold_summaries,
        }

    n_samples = len(samples_all)
    for fold in range(len(fold_test_indices)):
        test_idx = fold_test_indices[fold]
        val_idx = fold_test_indices[(fold + 1) % len(fold_test_indices)]
        train_idx = np.setdiff1d(np.arange(n_samples), np.concatenate([test_idx, val_idx]))

        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        test_groups = set(groups[test_idx].tolist())

        train_val_overlap = sorted(train_groups & val_groups)
        train_test_overlap = sorted(train_groups & test_groups)
        val_test_overlap = sorted(val_groups & test_groups)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise RuntimeError(
                f"MMSI overlap detected in grouped split for fold {fold}: "
                f"train/val={len(train_val_overlap)}, train/test={len(train_test_overlap)}, "
                f"val/test={len(val_test_overlap)}"
            )

        fold_summaries.append(
            {
                "fold": fold,
                "n_train_groups": len(train_groups),
                "n_val_groups": len(val_groups),
                "n_test_groups": len(test_groups),
                "train_val_overlap": 0,
                "train_test_overlap": 0,
                "val_test_overlap": 0,
            }
        )

    return {
        "split_mode": split_mode,
        "group_separation_checked": True,
        "fold_group_overlap": fold_summaries,
    }


def write_dataset_summary(
    out_root: Path,
    samples_all: list[tuple[str, int]],
    meta: dict[str, Any],
    classes: list[str],
    scheme: str,
    groups: np.ndarray | None = None,
    split_mode: str | None = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    unique_groups = int(len(set(groups.tolist()))) if groups is not None else None
    payload = {
        "class_scheme": scheme,
        "classes": classes,
        "n_samples": len(samples_all),
        "kept_counts": meta["kept_counts"],
        "dropped_other_classes": meta["dropped_other_classes"],
        "skipped_non_tif": meta["skipped_non_tif"],
        "split_mode": split_mode,
        "unique_mmsi_count": unique_groups,
        "conflicting_mmsi_exclusion": meta.get("conflicting_mmsi_exclusion"),
    }
    with open(out_root / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(
        [{"class_name": cls, "count": meta["kept_counts"].get(cls, 0)} for cls in classes]
    ).to_csv(out_root / "dataset_class_counts.csv", index=False)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    defaults = load_training_defaults(args.config)
    training = {
        "epochs": args.epochs if args.epochs is not None else defaults["epochs"],
        "batch_size": args.batch_size if args.batch_size is not None else defaults["batch_size"],
        "learning_rate": args.learning_rate if args.learning_rate is not None else defaults["learning_rate"],
        "weight_decay": args.weight_decay if args.weight_decay is not None else defaults["weight_decay"],
        "num_workers": args.num_workers if args.num_workers is not None else defaults["num_workers"],
        "image_size": args.image_size if args.image_size is not None else defaults["image_size"],
        "seed": args.seed if args.seed is not None else defaults["seed"],
        "early_stopping_patience": (
            args.early_stopping_patience if args.early_stopping_patience is not None
            else defaults["early_stopping_patience"]
        ),
        "mixed_precision": False if args.no_mixed_precision else bool(defaults["mixed_precision"]),
        "train_val_test_split": defaults["train_val_test_split"],
        "bands_order": args.bands_order if args.bands_order is not None else defaults["bands_order"],
    }

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    set_determinism(training["seed"])
    classes = CLASS_SCHEMES[args.class_scheme]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"local_cv_{args.class_scheme}class_{args.arch}_{args.split_mode}_{ts}"
    out_root = args.output_root / run_name

    samples_all, meta = collect_samples(args.dataset_root, args.class_scheme, args.max_samples_per_class)
    if not samples_all:
        raise RuntimeError("No valid samples found for the requested class scheme.")

    if args.exclude_conflicting_mmsi:
        filtered_samples, exclusion_stats = exclude_conflicting_mmsi_samples(samples_all, classes)
        if not filtered_samples:
            raise RuntimeError("All samples were removed by conflicting-MMSI exclusion.")
        samples_all = filtered_samples
        meta["kept_counts"] = exclusion_stats["kept_counts_after_exclusion"]
        meta["conflicting_mmsi_exclusion"] = exclusion_stats
    else:
        meta["conflicting_mmsi_exclusion"] = {"applied": False}

    group_values = []
    missing_group_examples = []
    for path_str, _ in samples_all:
        mmsi = extract_mmsi_from_path(path_str)
        if mmsi is None:
            if len(missing_group_examples) < 10:
                missing_group_examples.append(path_str)
            group_values.append("__MISSING__")
        else:
            group_values.append(mmsi)
    groups = np.array(group_values, dtype=object)

    if args.split_mode == "mmsi_grouped" and missing_group_examples:
        raise RuntimeError(
            "Failed to extract MMSI from some filenames required for grouped splitting. "
            f"Examples: {missing_group_examples}"
        )

    write_dataset_summary(out_root, samples_all, meta, classes, args.class_scheme, groups=groups, split_mode=args.split_mode)
    print(f"Collected {len(samples_all)} samples for {args.class_scheme}-class scheme.")
    print("Counts:", meta["kept_counts"])
    if meta.get("conflicting_mmsi_exclusion", {}).get("applied"):
        stats = meta["conflicting_mmsi_exclusion"]
        print(
            "Conflicting-MMSI exclusion:",
            {
                "conflicting_mmsi_count": stats["conflicting_mmsi_count"],
                "excluded_sample_count": stats["excluded_sample_count"],
            },
        )
    print(f"Split mode: {args.split_mode}")
    print(f"Unique MMSI count: {len(set(groups.tolist()))}")

    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-learn is required for full training but is not installed in the current Python environment. "
            "Install the project requirements first, then rerun the training command."
        ) from SKLEARN_IMPORT_ERROR

    test_arr = read_tif(samples_all[0][0], expected_channels=len(training["bands_order"]))
    in_ch = test_arr.shape[0]
    if len(training["bands_order"]) != in_ch:
        warnings.warn(
            f"bands_order length ({len(training['bands_order'])}) != image channels ({in_ch}). "
            f"Proceeding with {in_ch} channels."
        )

    num_classes = len(classes)
    k = infer_fold_count(training["train_val_test_split"], args.folds)
    print(
        f"K-fold setup: K={k} (target split ~ "
        f"{training['train_val_test_split'][0]}/{training['train_val_test_split'][1]}/{training['train_val_test_split'][2]})"
    )

    x_idx = np.arange(len(samples_all))
    y = np.array([s[1] for s in samples_all])
    fold_test_indices = build_fold_test_indices(
        x_idx=x_idx,
        y=y,
        groups=groups,
        k=k,
        seed=training["seed"],
        split_mode=args.split_mode,
    )
    split_check = validate_group_separation(samples_all, fold_test_indices, groups, args.split_mode)
    with open(out_root / "split_check.json", "w", encoding="utf-8") as f:
        json.dump(split_check, f, indent=2)
    if args.dry_run:
        print(f"Fold split verified for mode: {args.split_mode}")
        print(f"Dry-run complete. Summary written to: {out_root}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    total_train_start = perf_counter()
    cv_results = []
    agg_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    oof_rows: list[dict[str, Any]] = []

    for fold in range(k):
        fold_start = perf_counter()

        test_idx = fold_test_indices[fold]
        val_idx = fold_test_indices[(fold + 1) % k]
        train_idx = np.setdiff1d(np.arange(len(samples_all)), np.concatenate([test_idx, val_idx]))

        x_train = [(samples_all[i][0], samples_all[i][1]) for i in train_idx]
        x_val = [(samples_all[i][0], samples_all[i][1]) for i in val_idx]
        x_test = [(samples_all[i][0], samples_all[i][1]) for i in test_idx]

        train_labels = [lbl for _, lbl in x_train]
        cnt = Counter(train_labels)
        total = sum(cnt.values())
        class_weights = torch.tensor(
            [total / max(1, cnt.get(c, 1)) for c in range(num_classes)],
            dtype=torch.float32,
        )

        norm_stats = compute_percentile_stats(
            x_train,
            image_size=training["image_size"],
            per_class_limit=200,
            expected_channels=in_ch,
        )

        dl_train, dl_val, dl_test = make_loaders(
            x_train, x_val, x_test,
            training["image_size"], norm_stats,
            training["batch_size"], training["num_workers"],
            in_ch, device
        )

        model = build_model(args.arch, in_ch=in_ch, num_classes=num_classes).to(device)
        fold_dir = out_root / f"fold_{fold:02d}"
        out = train_one_fold(
            fold_dir=fold_dir,
            model=model,
            device=device,
            dl_train=dl_train,
            dl_val=dl_val,
            dl_test=dl_test,
            epochs=training["epochs"],
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
            early_stopping_patience=training["early_stopping_patience"],
            mixed_precision=training["mixed_precision"],
            class_weights=class_weights,
        )

        fold_elapsed = perf_counter() - fold_start
        print(f"[Fold {fold}] {format_hms(fold_elapsed)} ({fold_elapsed:.1f}s)")

        if out["best_state"] is not None:
            torch.save(out["best_state"], fold_dir / "best_model.pth")

        y_true = np.array(out["y_true_test"])
        y_pred = np.array(out["y_pred_test"])
        y_prob = np.array(out["y_prob_test"])

        acc = accuracy_score(y_true, y_pred)
        top2 = top_k_accuracy_score(y_true, y_prob, k=min(2, num_classes), labels=list(range(num_classes)))
        rep_dict = classification_report_to_df_dict(y_true, y_pred, classes)

        with open(fold_dir / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(rep_dict, f, indent=2)
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "fold": fold,
                "test_accuracy": float(acc),
                "test_top2_accuracy": float(top2),
                "best_val_loss": float(out["best_val_loss"]),
                "arch": args.arch,
                "class_scheme": args.class_scheme,
            }, f, indent=2)

        save_confusion_matrix(y_true, y_pred, classes, fold_dir / "confusion_matrix.png", normalize=False)
        save_confusion_matrix(y_true, y_pred, classes, fold_dir / "confusion_matrix_norm.png", normalize=True)
        agg_confusion += confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        for local_idx, sample_idx in enumerate(test_idx):
            src_path, label_idx = samples_all[sample_idx]
            row = {
                "fold": fold,
                "filepath": src_path,
                "true_label_idx": int(label_idx),
                "true_label": classes[label_idx],
                "pred_label_idx": int(y_pred[local_idx]),
                "pred_label": classes[int(y_pred[local_idx])],
            }
            probs = y_prob[local_idx]
            for c_idx, c_name in enumerate(classes):
                row[f"prob_{c_name}"] = float(probs[c_idx])
            row["correct"] = bool(int(y_pred[local_idx]) == int(label_idx))
            oof_rows.append(row)

        cv_results.append({
            "fold": fold,
            "n_train": len(x_train),
            "n_val": len(x_val),
            "n_test": len(x_test),
            "train_time_sec": round(fold_elapsed, 2),
            "train_time_hms": format_hms(fold_elapsed),
            "acc": float(acc),
            "top2": float(top2),
            "macro_f1": float(rep_dict["macro avg"]["f1-score"]),
            "weighted_f1": float(rep_dict["weighted avg"]["f1-score"]),
        })

    df_cv = pd.DataFrame(cv_results)
    df_cv.to_csv(out_root / "cv_summary.csv", index=False)

    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(out_root / "oof_predictions.csv", index=False)
    oof_df[oof_df["correct"] == False].to_csv(out_root / "test_misclassified.csv", index=False)

    total_train_elapsed = perf_counter() - total_train_start
    cv_stats = {
        "arch": args.arch,
        "class_scheme": args.class_scheme,
        "split_mode": args.split_mode,
        "classes": classes,
        "folds": len(cv_results),
        "acc_mean": float(df_cv["acc"].mean()),
        "acc_std": float(df_cv["acc"].std(ddof=0)),
        "macro_f1_mean": float(df_cv["macro_f1"].mean()),
        "macro_f1_std": float(df_cv["macro_f1"].std(ddof=0)),
        "weighted_f1_mean": float(df_cv["weighted_f1"].mean()),
        "weighted_f1_std": float(df_cv["weighted_f1"].std(ddof=0)),
        "top2_mean": float(df_cv["top2"].mean()),
        "top2_std": float(df_cv["top2"].std(ddof=0)),
        "total_train_time_sec": float(round(total_train_elapsed, 2)),
        "total_train_time_hms": format_hms(total_train_elapsed),
        "dataset_root": str(args.dataset_root),
    }
    with open(out_root / "cv_stats.json", "w", encoding="utf-8") as f:
        json.dump(cv_stats, f, indent=2)

    with open(out_root / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Dataset root: {args.dataset_root}\n")
        f.write(f"Class scheme: {args.class_scheme}\n")
        f.write(f"Split mode: {args.split_mode}\n")
        f.write(f"Classes: {classes}\n")
        f.write(f"Architecture: {args.arch}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Samples: {len(samples_all)}\n")
        f.write(f"Unique MMSI: {len(set(groups.tolist()))}\n")
        f.write(f"Counts: {meta['kept_counts']}\n")
        if meta.get("conflicting_mmsi_exclusion", {}).get("applied"):
            stats = meta["conflicting_mmsi_exclusion"]
            f.write(
                "Conflicting-MMSI exclusion: "
                f"{stats['conflicting_mmsi_count']} MMSI removed, "
                f"{stats['excluded_sample_count']} samples excluded\n"
            )
        f.write(f"Total training time: {format_hms(total_train_elapsed)} ({total_train_elapsed:.1f}s)\n")
        for rec in cv_results:
            f.write(f"Fold {rec['fold']}: {rec['train_time_hms']} ({rec['train_time_sec']}s)\n")

    plt.figure(figsize=(6, 5))
    plt.imshow(agg_confusion, interpolation="nearest")
    plt.title("Aggregate Confusion Matrix")
    plt.colorbar()
    tick = np.arange(len(classes))
    plt.xticks(tick, classes, rotation=45, ha="right")
    plt.yticks(tick, classes)
    thr = agg_confusion.max() / 2 if agg_confusion.size else 0.0
    for i in range(agg_confusion.shape[0]):
        for j in range(agg_confusion.shape[1]):
            plt.text(j, i, str(agg_confusion[i, j]), ha="center", va="center",
                     color="white" if agg_confusion[i, j] > thr else "black")
    plt.tight_layout()
    plt.savefig(out_root / "aggregate_confusion.png", dpi=160)
    plt.close()

    save_confusion_matrix(
        oof_df["true_label_idx"].to_numpy(),
        oof_df["pred_label_idx"].to_numpy(),
        classes,
        out_root / "aggregate_confusion_norm.png",
        normalize=True,
    )

    print(f"Done. Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
