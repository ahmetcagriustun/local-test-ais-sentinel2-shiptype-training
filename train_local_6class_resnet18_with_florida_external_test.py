from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

# Import torch before the base trainer to avoid intermittent DLL init failures on Windows.
import train_local_shiptype_cv as base

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        top_k_accuracy_score,
    )
except Exception as exc:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "scikit-learn is required for the Florida external test training script."
    ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    project_root = repo_root.parent
    default_denmark = project_root / "dataset_ship_type"
    default_florida = project_root / "noaa_accessais_sentinel2_testset" / "data" / "patches" / "florida_6class_by_type"
    default_config = repo_root / "config.yaml"
    if not default_config.exists():
        default_config = repo_root / "config.example.yaml"
    default_output = repo_root / "results_local_external_test_runs"

    parser = argparse.ArgumentParser(
        description="Train one CV fold of a 6-class ResNet18 on Denmark patches and report Denmark/Florida test results separately."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_denmark, help="Denmark dataset root.")
    parser.add_argument("--florida-root", type=Path, default=default_florida, help="Florida external-test dataset root.")
    parser.add_argument("--config", type=Path, default=default_config, help="Optional config.yaml for defaults.")
    parser.add_argument("--output-root", type=Path, default=default_output, help="Directory to write outputs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--exclude-conflicting-mmsi", action="store_true")
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--fold-index", type=int, default=0, help="Which fold to train/evaluate (0-based).")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def summarize_split(samples: list[tuple[str, int]], classes: list[str]) -> dict[str, int]:
    counter = Counter(label for _, label in samples)
    return {class_name: int(counter.get(idx, 0)) for idx, class_name in enumerate(classes)}


def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def report_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str],
    out_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    top2 = top_k_accuracy_score(y_true, y_prob, k=min(2, len(classes)), labels=list(range(len(classes))))
    rep_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    with open(out_dir / f"{prefix}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(rep_dict, f, indent=2)

    metrics = {
        "accuracy": float(acc),
        "top2_accuracy": float(top2),
        "macro_f1": float(rep_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(rep_dict["weighted avg"]["f1-score"]),
        "n_samples": int(len(y_true)),
    }
    with open(out_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    base.save_confusion_matrix(y_true, y_pred, classes, out_dir / f"{prefix}_confusion_matrix.png", normalize=False)
    base.save_confusion_matrix(y_true, y_pred, classes, out_dir / f"{prefix}_confusion_matrix_norm.png", normalize=True)

    pred_rows = []
    for idx in range(len(y_true)):
        row = {
            "true_label_idx": int(y_true[idx]),
            "true_label": classes[int(y_true[idx])],
            "pred_label_idx": int(y_pred[idx]),
            "pred_label": classes[int(y_pred[idx])],
            "correct": bool(int(y_true[idx]) == int(y_pred[idx])),
        }
        for class_idx, class_name in enumerate(classes):
            row[f"prob_{class_name}"] = float(y_prob[idx, class_idx])
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(out_dir / f"{prefix}_predictions.csv", index=False)
    return metrics


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    defaults = base.load_training_defaults(args.config)
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
        "bands_order": defaults["bands_order"],
    }

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Denmark dataset root not found: {args.dataset_root}")
    if not args.florida_root.exists():
        raise FileNotFoundError(f"Florida dataset root not found: {args.florida_root}")

    base.set_determinism(training["seed"])
    classes = base.CLASS_SCHEMES["6"]
    split_mode = "patch"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"denmark_6class_resnet18_fold{args.fold_index:02d}_with_florida_external_{ts}"
    out_root = args.output_root / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    denmark_samples, denmark_meta = base.collect_samples(args.dataset_root, "6", args.max_samples_per_class)
    if not denmark_samples:
        raise RuntimeError("No Denmark samples found for the 6-class scheme.")

    if args.exclude_conflicting_mmsi:
        denmark_samples, exclusion_stats = base.exclude_conflicting_mmsi_samples(denmark_samples, classes)
        denmark_meta["kept_counts"] = exclusion_stats["kept_counts_after_exclusion"]
        denmark_meta["conflicting_mmsi_exclusion"] = exclusion_stats
    else:
        denmark_meta["conflicting_mmsi_exclusion"] = {"applied": False}

    florida_samples, florida_meta = base.collect_samples(args.florida_root, "6", None)
    if not florida_samples:
        raise RuntimeError("No Florida samples found for the 6-class scheme.")

    group_values = []
    for path_str, _ in denmark_samples:
        mmsi = base.extract_mmsi_from_path(path_str)
        group_values.append(mmsi if mmsi is not None else "__MISSING__")
    groups = np.array(group_values, dtype=object)

    x_idx = np.arange(len(denmark_samples))
    y = np.array([label for _, label in denmark_samples], dtype=int)
    k = base.infer_fold_count(training["train_val_test_split"], override=None)
    fold_test_indices = base.build_fold_test_indices(
        x_idx=x_idx,
        y=y,
        groups=groups,
        k=k,
        seed=training["seed"],
        split_mode=split_mode,
    )
    if args.fold_index < 0 or args.fold_index >= len(fold_test_indices):
        raise ValueError(f"fold-index must be between 0 and {len(fold_test_indices) - 1}, got {args.fold_index}")

    test_idx = fold_test_indices[args.fold_index]
    val_idx = fold_test_indices[(args.fold_index + 1) % len(fold_test_indices)]
    train_idx = np.setdiff1d(np.arange(len(denmark_samples)), np.concatenate([test_idx, val_idx]))

    x_train = [denmark_samples[i] for i in train_idx]
    x_val = [denmark_samples[i] for i in val_idx]
    x_test = [denmark_samples[i] for i in test_idx]
    x_florida = florida_samples

    split_summary = {
        "denmark_train_counts": summarize_split(x_train, classes),
        "denmark_val_counts": summarize_split(x_val, classes),
        "denmark_test_counts": summarize_split(x_test, classes),
        "florida_test_counts": summarize_split(x_florida, classes),
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "florida_test_size": len(x_florida),
        "split_ratio_target": training["train_val_test_split"],
        "fold_index": args.fold_index,
        "fold_count": k,
    }
    with open(out_root / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    if args.dry_run:
        print(json.dumps(split_summary, indent=2))
        print(f"Dry-run complete. Outputs written to: {out_root}")
        return

    in_ch = base.read_tif(x_train[0][0], expected_channels=len(training["bands_order"])).shape[0]
    print("Computing normalization statistics from Denmark training patches...")
    norm_stats = base.compute_percentile_stats(
        x_train,
        image_size=training["image_size"],
        per_class_limit=200,
        expected_channels=in_ch,
    )
    print("Normalization statistics ready.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Denmark split sizes -> train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")
    print(f"Florida external test size -> {len(x_florida)}")
    print("Building Denmark and Florida data loaders...")

    dl_train, dl_val, dl_test = base.make_loaders(
        x_train,
        x_val,
        x_test,
        training["image_size"],
        norm_stats,
        training["batch_size"],
        training["num_workers"],
        in_ch,
        device,
    )
    _, _, dl_florida = base.make_loaders(
        x_train,
        x_val,
        x_florida,
        training["image_size"],
        norm_stats,
        training["batch_size"],
        training["num_workers"],
        in_ch,
        device,
    )
    print("Data loaders ready.")

    train_labels = [label for _, label in x_train]
    cnt = Counter(train_labels)
    total = sum(cnt.values())
    class_weights = torch.tensor(
        [total / max(1, cnt.get(c, 1)) for c in range(len(classes))],
        dtype=torch.float32,
    )

    model = base.build_model("resnet18", in_ch=in_ch, num_classes=len(classes)).to(device)
    print("Starting training loop...")

    train_start = perf_counter()
    out = base.train_one_fold(
        fold_dir=out_root,
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
    train_elapsed = perf_counter() - train_start
    print(f"Training completed in {base.format_hms(train_elapsed)} ({train_elapsed:.1f}s)")

    if out["best_state"] is not None:
        torch.save(out["best_state"], out_root / "best_model.pth")
        model.load_state_dict(out["best_state"])

    denmark_y_true = np.array(out["y_true_test"])
    denmark_y_pred = np.array(out["y_pred_test"])
    denmark_y_prob = np.array(out["y_prob_test"])
    florida_y_true, florida_y_pred, florida_y_prob = evaluate_loader(model, dl_florida, device)

    denmark_metrics = report_metrics(
        y_true=denmark_y_true,
        y_pred=denmark_y_pred,
        y_prob=denmark_y_prob,
        classes=classes,
        out_dir=out_root,
        prefix="denmark_test",
    )
    florida_metrics = report_metrics(
        y_true=florida_y_true,
        y_pred=florida_y_pred,
        y_prob=florida_y_prob,
        classes=classes,
        out_dir=out_root,
        prefix="florida_test",
    )

    summary = {
        "run_name": run_name,
        "class_scheme": "6",
        "architecture": "resnet18",
        "epochs": training["epochs"],
        "split_mode": split_mode,
        "fold_index": args.fold_index,
        "fold_count": k,
        "split_ratio_target": training["train_val_test_split"],
        "dataset_root": str(args.dataset_root),
        "florida_root": str(args.florida_root),
        "exclude_conflicting_mmsi": bool(args.exclude_conflicting_mmsi),
        "denmark_meta": denmark_meta,
        "florida_meta": florida_meta,
        "split_summary": split_summary,
        "train_time_sec": round(train_elapsed, 2),
        "train_time_hms": base.format_hms(train_elapsed),
        "denmark_test_metrics": denmark_metrics,
        "florida_test_metrics": florida_metrics,
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_root / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Architecture: resnet18\n")
        f.write(f"Epochs: {training['epochs']}\n")
        f.write(f"Fold index: {args.fold_index} / {k - 1}\n")
        f.write(f"Split ratio target: {training['train_val_test_split']}\n")
        f.write(f"Denmark train/val/test: {len(x_train)}/{len(x_val)}/{len(x_test)}\n")
        f.write(f"Florida external test: {len(x_florida)}\n")
        f.write(f"Training time: {base.format_hms(train_elapsed)} ({train_elapsed:.1f}s)\n")
        f.write(
            f"Denmark test -> acc={denmark_metrics['accuracy']:.4f}, "
            f"macro_f1={denmark_metrics['macro_f1']:.4f}, "
            f"weighted_f1={denmark_metrics['weighted_f1']:.4f}\n"
        )
        f.write(
            f"Florida test -> acc={florida_metrics['accuracy']:.4f}, "
            f"macro_f1={florida_metrics['macro_f1']:.4f}, "
            f"weighted_f1={florida_metrics['weighted_f1']:.4f}\n"
        )

    print(f"Done. Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
