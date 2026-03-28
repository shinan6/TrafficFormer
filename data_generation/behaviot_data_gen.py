"""BehavIoT data generation: config, manifest, feature extraction, k-fold splits, TSV writing."""

import csv
import json
import math
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

    Pre-filters the pcap to remove non-IP packets (e.g., EAPOL) that would
    crash get_feature_flow's random_ip_port() when it expects IP in the first packet.
    """
    pcap_path, payload_length, start_index = args
    try:
        import os
        import sys
        import tempfile
        # finetuning_data_gen.py uses `from utils import *` expecting data_generation/ on path
        data_gen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if data_gen_dir not in sys.path:
            sys.path.insert(0, data_gen_dir)

        import scapy.all as scapy_mod
        packets = scapy_mod.rdpcap(pcap_path)
        ip_packets = scapy_mod.PacketList([p for p in packets if scapy_mod.IP in p])
        if len(ip_packets) < 3:
            return pcap_path, None, f"fewer_than_3_ip_packets ({len(ip_packets)})"

        # Write filtered pcap to temp file for get_feature_flow
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            scapy_mod.wrpcap(tmp_path, ip_packets)

            from data_generation.finetuning_data_gen import get_feature_flow
            result = get_feature_flow(tmp_path, payload_length, packets_num=999999,
                                      start_index=start_index)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if result == -1:
            return pcap_path, None, "extraction_returned_-1"
        return pcap_path, result[0], None
    except Exception as e:
        return pcap_path, None, str(e)


def load_device_macs(metadata_path):
    """Load device MAC addresses from device_behaviot.txt.

    File format: device_name mac_address manufacturer (space-separated)
    Returns: dict mapping device_label to lowercase MAC address.
    """
    macs = {}
    with open(metadata_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                macs[parts[0]] = parts[1].lower()
    return macs


def _extract_by_device_mac(args):
    """Extract bigram features from packets matching a device MAC address.

    Bypasses flowcontainer entirely — reads pcap with scapy, filters to IP
    packets involving the device MAC, anonymizes, and converts to bigrams.
    No minimum packet count; short samples are padded by the tokenizer.
    """
    pcap_path, payload_length, start_index, device_mac = args
    try:
        import binascii
        import os
        import sys
        import scapy.all as scapy_mod

        data_gen_dir = os.path.dirname(os.path.abspath(__file__))
        if data_gen_dir not in sys.path:
            sys.path.insert(0, data_gen_dir)

        packets = scapy_mod.rdpcap(pcap_path)
        device_mac = device_mac.lower()

        # Filter to IP packets involving the device MAC
        device_packets = scapy_mod.PacketList([
            p for p in packets
            if scapy_mod.IP in p and scapy_mod.Ether in p
            and (p[scapy_mod.Ether].src.lower() == device_mac
                 or p[scapy_mod.Ether].dst.lower() == device_mac)
        ])

        if len(device_packets) == 0:
            return pcap_path, None, "no_matching_packets"

        from data_generation.finetuning_data_gen import random_ip_port, random_tls_randomtime
        from utils import bigram_generation

        # Anonymize
        device_packets = random_ip_port(device_packets)
        if device_packets == -1:
            return pcap_path, None, "anonymization_failed"
        device_packets = random_tls_randomtime(device_packets)

        # Convert to bigrams
        flow_data_string = ""
        No_ether = not hasattr(device_packets[0], "type")

        for packet in device_packets:
            packet_data = packet.copy()
            data = binascii.hexlify(bytes(packet_data))
            if No_ether:
                packet_string = data.decode()
                packet_string = "c49a025996f8e46f13e2e3ae0800" + packet_string
                packet_string = packet_string[start_index:start_index + 2 * payload_length]
            else:
                packet_string = data.decode()[start_index:start_index + 2 * payload_length]

            flow_data_string += "[SEP] "
            flow_data_string += bigram_generation(
                packet_string.strip(),
                token_len=len(packet_string.strip()),
                flag=True,
            )

        return pcap_path, flow_data_string, None
    except Exception as e:
        return pcap_path, None, str(e)


def compute_class_weights(class_counts, method="inverse_sqrt"):
    """Compute class weights for weighted NLLLoss.

    Args:
        class_counts: dict mapping class_name to sample count
        method: "inverse_sqrt" or "none"

    Returns:
        List of float weights ordered by sorted class name, or None if method="none".
    """
    if method == "none":
        return None
    n_total = sum(class_counts.values())
    n_classes = len(class_counts)
    weights = []
    for label in sorted(class_counts.keys()):
        n_c = class_counts[label]
        if method == "inverse_sqrt":
            weights.append(math.sqrt(n_total / (n_classes * n_c)))
        else:
            raise ValueError(f"Unknown weighting method: {method}")
    return weights


def extract_all_features(manifest_rows, config, cache_dir=None, n_workers=32,
                         device_macs=None):
    """Extract bigram features from all pcaps with multiprocessing and caching.

    If device_macs is provided, uses MAC-filtered single-device extraction
    (bypasses flowcontainer). Otherwise uses the legacy all-flows extraction.

    Args:
        manifest_rows: List of dicts with 'pcap_path' and optionally 'device_label'.
        config: Dict with 'payload_length', 'start_index'.
        cache_dir: If set, cache extracted features to this directory.
        n_workers: Number of parallel workers (1 = serial, for testing).
        device_macs: Optional dict mapping device_label to MAC address.

    Returns:
        (features_dict, skipped_list) where features_dict maps pcap_path to
        datagram string and skipped_list contains dicts with pcap_path + reason.
    """
    # Build a cache key from extraction parameters + manifest content to detect stale caches
    import hashlib
    pcap_paths = sorted(r["pcap_path"] for r in manifest_rows)
    extraction_mode = "mac_filtered" if device_macs else "all_flows"
    cache_key = hashlib.md5(json.dumps({
        "pcap_paths": pcap_paths,
        "payload_length": config["payload_length"],
        "start_index": config["start_index"],
        "extraction_mode": extraction_mode,
    }, sort_keys=True).encode()).hexdigest()[:16]

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_file = cache_dir / "features_cache.json"
        cache_key_file = cache_dir / "cache_key.txt"
        skipped_file = cache_dir / "skipped_samples.csv"
        if cache_file.exists():
            stored_key = cache_key_file.read_text().strip() if cache_key_file.exists() else ""
            if stored_key == cache_key:
                with open(cache_file) as f:
                    features = json.load(f)
                skipped = []
                if skipped_file.exists():
                    with open(skipped_file) as f:
                        skipped = list(csv.DictReader(f))
                print(f"Loaded {len(features)} cached features, {len(skipped)} previously skipped")
                return features, skipped
            else:
                print(f"Cache key mismatch (config or manifest changed). Re-extracting.")

    if device_macs:
        args_list = [
            (r["pcap_path"], config["payload_length"], config["start_index"],
             device_macs.get(r.get("device_label", ""), ""))
            for r in manifest_rows
        ]
        extract_fn = _extract_by_device_mac
    else:
        args_list = [
            (r["pcap_path"], config["payload_length"], config["start_index"])
            for r in manifest_rows
        ]
        extract_fn = _extract_single_pcap

    features = {}
    skipped = []

    if n_workers > 1 and len(args_list) > 1:
        from multiprocessing import Pool
        with Pool(min(n_workers, len(args_list))) as pool:
            for pcap_path, datagram, error in pool.imap_unordered(extract_fn, args_list):
                if error:
                    skipped.append({"pcap_path": pcap_path, "reason": error})
                else:
                    features[pcap_path] = datagram
    else:
        for a in args_list:
            pcap_path, datagram, error = extract_fn(a)
            if error:
                skipped.append({"pcap_path": pcap_path, "reason": error})
            else:
                features[pcap_path] = datagram

    print(f"Extracted {len(features)} features, skipped {len(skipped)}")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(features, f)
        cache_key_file.write_text(cache_key)
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
    if len(counts) < 2:
        raise ValueError(
            f"Split infeasible: need at least 2 classes but found {len(counts)}. "
            f"Check min_samples_per_class filter or manifest content."
        )
    problems = {label: c for label, c in counts.items() if c < n_folds}
    if problems:
        raise ValueError(
            f"Split infeasible: these classes have fewer samples than n_folds={n_folds}: "
            f"{problems}. Increase min_samples_per_class or reduce n_folds."
        )
    total_train = sum(counts.values()) * (n_folds - 1) // n_folds
    min_dev = max(2, len(counts))
    if int(total_train * 0.1) < 1:
        raise ValueError(
            f"Split infeasible: training set too small for 10% dev split "
            f"({total_train} samples, {len(counts)} classes)."
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
        try:
            train_rows, dev_rows = train_test_split(
                train_all, test_size=0.1, random_state=seed, stratify=train_labels
            )
        except ValueError:
            # Stratified inner split infeasible (some classes have too few samples
            # for 10% dev). Fall back to non-stratified random split.
            print(f"WARNING: Stratified inner dev split infeasible for fold "
                  f"(likely classes with <=1 sample in train). Using random split.")
            train_rows, dev_rows = train_test_split(
                train_all, test_size=0.1, random_state=seed
            )
        folds.append((train_rows, dev_rows, test_rows))
    return folds


def write_fold_tsvs(fold_splits, features, label_to_id, target_column, output_dir,
                    max_tokens=None):
    """Write train/valid/test TSV files for one fold.

    TSV format matches run_classifier.py: header 'label\\ttext_a',
    then one row per sample with numeric label and bigram datagram string.

    If max_tokens is set, truncate each datagram string to approximately that many
    space-separated tokens. This avoids slow tokenization when seq_length << full
    datagram length. The full strings remain in the feature cache for optionality.
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
                datagram = features[r["pcap_path"]]
                if max_tokens and datagram:
                    # Truncate to ~max_tokens space-separated tokens
                    # Each token is ~5 chars ("XX "), so max_tokens*5 chars is a safe cutoff
                    char_limit = max_tokens * 5
                    if len(datagram) > char_limit:
                        datagram = datagram[:char_limit].rsplit(" ", 1)[0]
                f.write(f"{label_id}\t{datagram}\n")


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
