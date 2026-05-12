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

import train_local_shiptype_cv as base

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        top_k_accuracy_score,
    )
except Exception as exc:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "scikit-learn is required for the Florida 4-class evaluation script."
    ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    project_root = repo_root.parent
    default_denmark = project_root / "dataset_ship_type"
    default_florida = project_root / "noaa_accessais_sentinel2_testset" / "data" / "patches" / "florida_6class_by_type"
    default_run_root = repo_root / "results_local_resnet18_runs" / "local_cv_4class_resnet18_patch_20260502_002403"
    default_config = repo_root / "config.example.yaml"
    default_output = repo_root / "results_florida_4class_eval_runs"

    parser = argparse.ArgumentParser(
        description="Evaluate the best 4-class ResNet18 checkpoint on remapped Florida patches using CPU only."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_denmark, help="Denmark dataset root.")
    parser.add_argument("--florida-root", type=Path, default=default_florida, help="Florida 6-class patch root.")
    parser.add_argument("--run-root", type=Path, default=default_run_root, help="4-class ResNet18 CV run directory.")
    parser.add_argument("--config", type=Path, default=default_config, help="Config/example config path.")
    parser.add_argument("--output-root", type=Path, default=default_output, help="Directory to write evaluation outputs.")
    parser.add_argument("--fold-index", type=int, default=None, help="Optional explicit fold index. Defaults to best fold by test accuracy.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for CPU evaluation.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. CPU-only default is 0.")
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
            x = x.to(device, non_blocking=False)
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

    with open(out_dir / "florida_4class_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(rep_dict, f, indent=2)

    metrics = {
        "accuracy": float(acc),
        "top2_accuracy": float(top2),
        "macro_f1": float(rep_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(rep_dict["weighted avg"]["f1-score"]),
        "n_samples": int(len(y_true)),
    }
    with open(out_dir / "florida_4class_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    base.save_confusion_matrix(y_true, y_pred, classes, out_dir / "florida_4class_confusion_matrix.png", normalize=False)
    base.save_confusion_matrix(y_true, y_pred, classes, out_dir / "florida_4class_confusion_matrix_norm.png", normalize=True)

    rows = []
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
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "florida_4class_predictions.csv", index=False)
    return metrics


def select_best_fold(run_root: Path, explicit_fold_index: int | None) -> tuple[int, Path, dict[str, Any]]:
    cv_summary = run_root / "cv_summary.csv"
    if not cv_summary.exists():
        raise FileNotFoundError(f"cv_summary.csv not found under run root: {run_root}")

    df = pd.read_csv(cv_summary)
    if explicit_fold_index is not None:
        row = df.loc[df["fold"] == explicit_fold_index]
        if row.empty:
            raise ValueError(f"Requested fold {explicit_fold_index} not found in {cv_summary}")
        best_row = row.iloc[0]
    else:
        best_row = df.sort_values(["acc", "macro_f1", "weighted_f1"], ascending=[False, False, False]).iloc[0]

    fold_index = int(best_row["fold"])
    fold_dir = run_root / f"fold_{fold_index:02d}"
    checkpoint = fold_dir / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return fold_index, checkpoint, best_row.to_dict()


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Denmark dataset root not found: {args.dataset_root}")
    if not args.florida_root.exists():
        raise FileNotFoundError(f"Florida patch root not found: {args.florida_root}")
    if not args.run_root.exists():
        raise FileNotFoundError(f"4-class run root not found: {args.run_root}")

    training = base.load_training_defaults(args.config if args.config.exists() else None)
    batch_size = args.batch_size if args.batch_size is not None else int(training["batch_size"])
    classes = base.CLASS_SCHEMES["4"]
    split_mode = "patch"

    fold_index, checkpoint_path, best_row = select_best_fold(args.run_root, args.fold_index)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"florida_4class_eval_resnet18_fold{fold_index:02d}_{ts}"
    out_root = args.output_root / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    denmark_samples, denmark_meta = base.collect_samples(args.dataset_root, "4", None)
    if not denmark_samples:
        raise RuntimeError("No Denmark samples found for the 4-class scheme.")
    florida_samples, florida_meta = base.collect_samples(args.florida_root, "4", None)
    if not florida_samples:
        raise RuntimeError("No remapped Florida samples found for the 4-class scheme.")

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
        seed=int(training["seed"]),
        split_mode=split_mode,
    )

    test_idx = fold_test_indices[fold_index]
    val_idx = fold_test_indices[(fold_index + 1) % len(fold_test_indices)]
    train_idx = np.setdiff1d(np.arange(len(denmark_samples)), np.concatenate([test_idx, val_idx]))

    x_train = [denmark_samples[i] for i in train_idx]
    x_val = [denmark_samples[i] for i in val_idx]
    x_florida = florida_samples

    split_summary = {
        "selected_fold_index": fold_index,
        "selected_fold_metrics_from_cv": best_row,
        "denmark_train_counts": summarize_split(x_train, classes),
        "denmark_val_counts": summarize_split(x_val, classes),
        "florida_test_counts": summarize_split(x_florida, classes),
        "train_size": len(x_train),
        "val_size": len(x_val),
        "florida_test_size": len(x_florida),
        "split_ratio_target": training["train_val_test_split"],
    }
    with open(out_root / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    in_ch = base.read_tif(x_train[0][0], expected_channels=len(training["bands_order"])).shape[0]
    print(f"Selected checkpoint: {checkpoint_path}")
    print(f"Computing normalization statistics from Denmark fold {fold_index:02d} train split...")
    norm_stats = base.compute_percentile_stats(
        x_train,
        image_size=int(training["image_size"]),
        per_class_limit=200,
        expected_channels=in_ch,
    )
    print("Normalization statistics ready.")

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Florida remapped 4-class test size -> {len(x_florida)}")
    print("Building CPU data loader for Florida evaluation...")
    _, _, dl_florida = base.make_loaders(
        x_train,
        x_val,
        x_florida,
        int(training["image_size"]),
        norm_stats,
        batch_size,
        int(args.num_workers),
        in_ch,
        device,
    )
    print("Data loader ready.")

    model = base.build_model("resnet18", in_ch=in_ch, num_classes=len(classes)).to(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    start = perf_counter()
    y_true, y_pred, y_prob = evaluate_loader(model, dl_florida, device)
    elapsed = perf_counter() - start
    print(f"CPU evaluation completed in {base.format_hms(elapsed)} ({elapsed:.1f}s)")

    metrics = report_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        classes=classes,
        out_dir=out_root,
    )

    summary = {
        "run_name": run_name,
        "mode": "cpu_eval_only",
        "class_scheme": "4",
        "architecture": "resnet18",
        "selected_fold_index": fold_index,
        "selected_checkpoint": str(checkpoint_path),
        "device": str(device),
        "dataset_root": str(args.dataset_root),
        "florida_root": str(args.florida_root),
        "run_root": str(args.run_root),
        "denmark_meta": denmark_meta,
        "florida_meta": florida_meta,
        "split_summary": split_summary,
        "eval_time_sec": round(elapsed, 2),
        "eval_time_hms": base.format_hms(elapsed),
        "florida_4class_metrics": metrics,
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_root / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write("Mode: CPU evaluation only\n")
        f.write("Architecture: resnet18\n")
        f.write("Class scheme: 4\n")
        f.write(f"Selected fold: {fold_index}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Florida remapped 4-class test size: {len(x_florida)}\n")
        f.write(f"Evaluation time: {base.format_hms(elapsed)} ({elapsed:.1f}s)\n")
        f.write(
            f"Florida 4-class -> acc={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}\n"
        )

    print(f"Done. Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
