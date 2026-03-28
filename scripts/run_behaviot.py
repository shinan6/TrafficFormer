#!/usr/bin/env python3
"""End-to-end BehavIoT runner with k-fold cross-validation and TensorBoard logging.

Usage:
    python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target activity_type_label
    python3 scripts/run_behaviot.py --config configs/behaviot/smoke.json --target device_type_label --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.behaviot_data_gen import (
    load_config, load_manifest, build_label_map,
    extract_all_features, build_kfold_splits, write_fold_tsvs,
    cap_samples_per_class,
)
from uer.reporting_utils import aggregate_fold_metrics


VALID_TARGETS = ["device_type_label", "device_label", "activity_type_label", "activity_label"]


def resolve_checkpoint(zip_path, model_path):
    """Extract pretrained model from zip if not already present. Renames to model_path."""
    model_path = Path(model_path)
    if model_path.exists():
        print(f"Checkpoint already exists: {model_path}")
        return str(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting checkpoint from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bin_names = [n for n in zf.namelist() if n.endswith(".bin") or ".bin-" in n]
        bin_names = [n for n in bin_names if not n.startswith("__MACOSX")]
        if not bin_names:
            raise FileNotFoundError(f"No .bin file found in {zip_path}")
        zf.extract(bin_names[0], str(model_path.parent))
        extracted = model_path.parent / bin_names[0]
        if extracted != model_path:
            extracted.rename(model_path)
    print(f"Checkpoint extracted to: {model_path}")
    return str(model_path)


def build_classifier_command(config, data_dir, results_dir, pretrained_path,
                             id_to_label_path, tb_log_dir):
    """Build the run_classifier.py subprocess command for one fold."""
    return [
        sys.executable, "fine-tuning/run_classifier.py",
        "--vocab_path", config["vocab_path"],
        "--config_path", config["config_path"],
        "--train_path", str(Path(data_dir) / "train_dataset.tsv"),
        "--dev_path", str(Path(data_dir) / "valid_dataset.tsv"),
        "--test_path", str(Path(data_dir) / "test_dataset.tsv"),
        "--pretrained_model_path", pretrained_path,
        "--output_model_path", str(Path(results_dir) / "finetuned_model.bin"),
        "--results_dir", str(results_dir),
        "--id_to_label_path", str(id_to_label_path),
        "--tb_log_dir", str(tb_log_dir),
        "--epochs_num", str(config["epochs_num"]),
        "--earlystop", str(config["earlystop"]),
        "--batch_size", str(config["batch_size"]),
        "--seq_length", str(config["seq_length"]),
        "--learning_rate", str(config["learning_rate"]),
        "--seed", str(config["seed"]),
        "--embedding", "word_pos_seg",
        "--encoder", "transformer",
        "--mask", "fully_visible",
    ]


def write_environment_json(path):
    """Write environment metadata for reproducibility."""
    import platform
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
            env["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    Path(path).write_text(json.dumps(env, indent=2))


def main():
    parser = argparse.ArgumentParser(description="BehavIoT k-fold cross-validation runner")
    parser.add_argument("--config", required=True, help="Path to BehavIoT config JSON")
    parser.add_argument("--target", required=True, choices=VALID_TARGETS,
                        help="Target label column from the manifest CSV")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print resolved config and exit without training")
    args = parser.parse_args()

    config = load_config(args.config)
    target = args.target

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["results_root"]) / target / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = {**config, "target_label_column": target, "run_dir": str(run_dir),
                "timestamp": timestamp}
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2))

    if args.dry_run:
        print(f"[DRY RUN] Target: {target}")
        print(f"[DRY RUN] Run dir: {run_dir}")
        print(f"[DRY RUN] Config: {json.dumps(resolved, indent=2)}")
        return

    pretrained_path = resolve_checkpoint(config["pretrained_model_zip"],
                                         config["pretrained_model_path"])
    write_environment_json(run_dir / "environment.json")

    rows = load_manifest(config["manifest_csv"])
    print(f"Loaded {len(rows)} manifest rows")

    cache_dir = Path(config["generated_dataset_root"]) / "cache"
    features, skipped = extract_all_features(rows, config, cache_dir=str(cache_dir))

    extractable_rows = [r for r in rows if r["pcap_path"] in features]
    print(f"Extractable samples: {len(extractable_rows)} / {len(rows)}")

    label_to_id = build_label_map(extractable_rows, target, config["min_samples_per_class"])
    id_to_label = {v: k for k, v in label_to_id.items()}
    print(f"Classes for {target}: {len(label_to_id)}")

    dropped = build_label_map(extractable_rows, target, min_samples=1)
    dropped_classes = {k for k in dropped if k not in label_to_id}
    if dropped_classes:
        print(f"Dropped classes (below min_samples={config['min_samples_per_class']}): {dropped_classes}")

    (run_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2))
    id_to_label_path = run_dir / "id_to_label.json"
    id_to_label_path.write_text(json.dumps(id_to_label, indent=2))

    working_rows = extractable_rows
    if config.get("max_samples_per_class"):
        working_rows = cap_samples_per_class(
            extractable_rows, target, label_to_id,
            config["max_samples_per_class"], config["seed"],
        )
        print(f"Capped to {len(working_rows)} samples (max {config['max_samples_per_class']}/class)")

    folds = build_kfold_splits(working_rows, target, label_to_id,
                               config["n_folds"], config["seed"])
    print(f"Built {len(folds)} folds")

    log_path = run_dir / "run.log"
    fold_metrics = []

    with open(log_path, "w") as log_file:
        for fold_idx, fold_splits in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx}/{len(folds)-1}")
            print(f"{'='*60}")

            data_dir = Path(config["generated_dataset_root"]) / target / f"fold_{fold_idx}"
            write_fold_tsvs(fold_splits, features, label_to_id, target, str(data_dir),
                            max_tokens=config["seq_length"])
            train_count = sum(1 for _ in open(data_dir / "train_dataset.tsv")) - 1
            test_count = sum(1 for _ in open(data_dir / "test_dataset.tsv")) - 1
            print(f"  Train: {train_count}, Test: {test_count}")

            fold_results_dir = run_dir / f"fold_{fold_idx}"
            fold_results_dir.mkdir(parents=True, exist_ok=True)
            tb_log_dir = run_dir / "tb_logs" / f"fold_{fold_idx}"
            cmd = build_classifier_command(
                config, str(data_dir), str(fold_results_dir),
                pretrained_path, str(id_to_label_path), str(tb_log_dir),
            )

            log_file.write(f"\n{'='*60}\nFold {fold_idx}\n{'='*60}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n\n")
            log_file.flush()

            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            log_file.write(result.stdout)
            log_file.flush()
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

            if result.returncode != 0:
                print(f"WARNING: Fold {fold_idx} failed with return code {result.returncode}")
                continue

            metrics_path = fold_results_dir / "metrics.json"
            if metrics_path.exists():
                fold_metrics.append(json.loads(metrics_path.read_text()))
                print(f"  Fold {fold_idx} macro_f1: {fold_metrics[-1]['macro_f1']:.4f}")

    n_expected = config["n_folds"]
    n_succeeded = len(fold_metrics)

    if n_succeeded < n_expected:
        print(f"\nERROR: Only {n_succeeded}/{n_expected} folds succeeded. "
              f"Check run.log for details.")

    if fold_metrics:
        aggregated = aggregate_fold_metrics(fold_metrics)
        aggregated["folds_succeeded"] = n_succeeded
        aggregated["folds_expected"] = n_expected
        (run_dir / "aggregated_metrics.json").write_text(json.dumps(aggregated, indent=2))
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS ({n_succeeded}/{n_expected} folds)")
        print(f"{'='*60}")
        for key in ["accuracy", "macro_f1", "weighted_f1"]:
            m = aggregated[key]
            print(f"  {key}: {m['mean']:.4f} +/- {m['std']:.4f}")
        print(f"\nTensorBoard: tensorboard --logdir {run_dir / 'tb_logs'}")
    else:
        print("ERROR: No fold metrics collected. Check run.log for errors.")

    print(f"\nAll results saved to: {run_dir}")

    if n_succeeded < n_expected:
        sys.exit(1)


if __name__ == "__main__":
    main()
