"""BehavIoT data generation: config, manifest, feature extraction, k-fold splits, TSV writing."""

import csv
import json
import random
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split


REQUIRED_CONFIG_KEYS = [
    "manifest_csv", "n_folds", "results_root", "generated_dataset_root",
    "pretrained_model_zip", "pretrained_model_path", "vocab_path",
    "config_path", "payload_length", "start_index",
    "seq_length", "learning_rate", "batch_size", "epochs_num",
    "earlystop", "seed", "min_samples_per_class",
]


def load_config(path):
    """Load and validate a BehavIoT JSON config."""
    with open(path) as f:
        cfg = json.load(f)
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return cfg


def load_manifest(csv_path):
    """Load the BehavIoT manifest CSV as a list of dicts."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def build_label_map(rows, target_column, min_samples=1):
    """Build a sorted label-to-id mapping, filtering classes below min_samples."""
    counts = Counter(r[target_column] for r in rows)
    labels = sorted(label for label, count in counts.items() if count >= min_samples)
    return {label: idx for idx, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_single_pcap(args):
    """Worker function for multiprocessing. Module-level for pickling.

    Calls get_feature_flow() with packets_num=999999 so all packets from all
    flows are included. seq_length truncation happens at tokenization time.
    """
    pcap_path, payload_length, start_index = args
    try:
        from data_generation.finetuning_data_gen import get_feature_flow
        result = get_feature_flow(pcap_path, payload_length, packets_num=999999,
                                  start_index=start_index)
        if result == -1:
            return pcap_path, None, "extraction_returned_-1"
        return pcap_path, result[0], None
    except Exception as e:
        return pcap_path, None, str(e)


def extract_all_features(manifest_rows, config, cache_dir=None, n_workers=32):
    """Extract bigram features from all pcaps with multiprocessing and caching.

    Args:
        manifest_rows: List of dicts with 'pcap_path' key.
        config: Dict with 'payload_length', 'start_index'.
        cache_dir: If set, cache extracted features to this directory.
        n_workers: Number of parallel workers (1 = serial, for testing).

    Returns:
        (features_dict, skipped_list) where features_dict maps pcap_path to
        datagram string and skipped_list contains dicts with pcap_path + reason.
    """
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_file = cache_dir / "features_cache.json"
        skipped_file = cache_dir / "skipped_samples.csv"
        if cache_file.exists():
            with open(cache_file) as f:
                features = json.load(f)
            skipped = []
            if skipped_file.exists():
                with open(skipped_file) as f:
                    skipped = list(csv.DictReader(f))
            print(f"Loaded {len(features)} cached features, {len(skipped)} previously skipped")
            return features, skipped

    args_list = [
        (r["pcap_path"], config["payload_length"], config["start_index"])
        for r in manifest_rows
    ]

    features = {}
    skipped = []

    if n_workers > 1 and len(args_list) > 1:
        from multiprocessing import Pool
        with Pool(min(n_workers, len(args_list))) as pool:
            for pcap_path, datagram, error in pool.imap_unordered(_extract_single_pcap, args_list):
                if error:
                    skipped.append({"pcap_path": pcap_path, "reason": error})
                else:
                    features[pcap_path] = datagram
    else:
        for a in args_list:
            pcap_path, datagram, error = _extract_single_pcap(a)
            if error:
                skipped.append({"pcap_path": pcap_path, "reason": error})
            else:
                features[pcap_path] = datagram

    print(f"Extracted {len(features)} features, skipped {len(skipped)}")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(features, f)
        if skipped:
            with open(skipped_file, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["pcap_path", "reason"])
                w.writeheader()
                w.writerows(skipped)

    return features, skipped


# ---------------------------------------------------------------------------
# K-fold splitting
# ---------------------------------------------------------------------------

def _validate_split_feasibility(rows, target_column, label_to_id, n_folds):
    """Check that every class has enough samples for the requested fold count."""
    counts = Counter(r[target_column] for r in rows if r[target_column] in label_to_id)
    problems = {label: c for label, c in counts.items() if c < n_folds}
    if problems:
        raise ValueError(
            f"Split infeasible: these classes have fewer samples than n_folds={n_folds}: "
            f"{problems}. Increase min_samples_per_class or reduce n_folds."
        )


def build_kfold_splits(rows, target_column, label_to_id, n_folds, seed):
    """Build stratified k-fold splits with inner train/dev split.

    Validates feasibility before generating folds. For each fold:
    test = one fold (20%), train+dev = remaining folds split 90/10.

    Returns:
        List of (train_rows, dev_rows, test_rows) tuples, one per fold.
    """
    valid_rows = [r for r in rows if r[target_column] in label_to_id]
    _validate_split_feasibility(valid_rows, target_column, label_to_id, n_folds)
    labels = [r[target_column] for r in valid_rows]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(valid_rows, labels):
        test_rows = [valid_rows[i] for i in test_idx]
        train_all = [valid_rows[i] for i in train_idx]
        train_labels = [r[target_column] for r in train_all]
        train_rows, dev_rows = train_test_split(
            train_all, test_size=0.1, random_state=seed, stratify=train_labels
        )
        folds.append((train_rows, dev_rows, test_rows))
    return folds


def write_fold_tsvs(fold_splits, features, label_to_id, target_column, output_dir):
    """Write train/valid/test TSV files for one fold.

    TSV format matches run_classifier.py: header 'label\\ttext_a',
    then one row per sample with numeric label and bigram datagram string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(["train", "valid", "test"], fold_splits):
        tsv_path = output_dir / f"{name}_dataset.tsv"
        with open(tsv_path, "w") as f:
            f.write("label\ttext_a\n")
            for r in rows:
                if r["pcap_path"] not in features:
                    continue
                label_id = label_to_id[r[target_column]]
                f.write(f"{label_id}\t{features[r['pcap_path']]}\n")


def cap_samples_per_class(rows, target_column, label_to_id, max_per_class, seed=42):
    """Limit samples per class to max_per_class via random sampling."""
    rng = random.Random(seed)
    by_class = {}
    for r in rows:
        label = r[target_column]
        if label in label_to_id:
            by_class.setdefault(label, []).append(r)
    capped = []
    for label in sorted(by_class):
        class_rows = by_class[label]
        if len(class_rows) > max_per_class:
            class_rows = rng.sample(class_rows, max_per_class)
        capped.extend(class_rows)
    return capped
