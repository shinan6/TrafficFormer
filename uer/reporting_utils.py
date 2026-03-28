"""Metrics computation, file writing, and cross-fold aggregation."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, label_names):
    """Compute classification metrics including per-class report."""
    all_labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=all_labels)),
        "per_class": classification_report(
            y_true, y_pred, target_names=label_names,
            labels=all_labels, output_dict=True, zero_division=0,
        ),
    }


def write_metrics_json(path, metrics):
    """Write metrics dict to JSON file."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def write_confusion_matrix_csv(path, y_true, y_pred, label_names):
    """Write confusion matrix as CSV (rows=true, cols=predicted, sklearn convention)."""
    all_labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + label_names)
        for i, row in enumerate(cm):
            w.writerow([label_names[i]] + [int(x) for x in row])


def write_predictions_tsv(path, y_true, y_pred, label_names):
    """Write per-sample predictions to TSV with human-readable label names."""
    with open(path, "w") as f:
        f.write("true_label\tpred_label\ttrue_id\tpred_id\n")
        for t, p in zip(y_true, y_pred):
            f.write(f"{label_names[t]}\t{label_names[p]}\t{t}\t{p}\n")


def aggregate_fold_metrics(fold_metrics_list):
    """Aggregate metrics across k folds: mean, std, and per-fold values."""
    if not fold_metrics_list:
        return {}
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    agg = {}
    for key in keys:
        values = [m[key] for m in fold_metrics_list]
        agg[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": values,
        }
    return agg
