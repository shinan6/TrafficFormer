# BehavIoT Multi-Label 5-Fold Cross-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt TrafficFormer to fine-tune and evaluate on all four BehavIoT label columns (`device_type_label`, `device_label`, `activity_type_label`, `activity_label`) from `/mnt/data/behavoiot/pcap_vs_label.csv`, using 5-fold stratified cross-validation with pretrained initialization, producing per-fold and aggregated metrics with TensorBoard logging.

**Architecture:** A config-driven BehavIoT pipeline runs alongside the existing TrafficFormer code. A shared data generator reads the manifest CSV, extracts bigram features from all pcaps once using the existing `get_feature_flow()` with no packet limit (all flows mixed, `seq_length=320` truncates at tokenization), caches results via 32-worker multiprocessing, then for any target label column builds stratified 5-fold splits and writes per-fold TrafficFormer-compatible TSV files. A runner script resolves the pretrained checkpoint from `/home/shinanliu/pre-trained_model.bin.zip`, trains `fine-tuning/run_classifier.py` per fold with TensorBoard logging, and aggregates cross-fold metrics. The `--target` CLI argument selects which of the four label columns to classify.

**Tech Stack:** Python 3.12, pandas, scapy, flowcontainer, scikit-learn (`StratifiedKFold`), PyTorch 2.10 + CUDA 12.8, TensorBoard, JSON configs, existing TrafficFormer/UER code.

**Hardware:** 2x NVIDIA RTX 5090 (32GB), 32 CPU cores.

**Design doc:** `docs/plans/2026-03-28-behaviot-multi-label-kfold-design.md`

---

### Key Design Decisions

1. **All flows mixed:** `get_feature_flow()` already reads all packets from the pcap via `scapy.rdpcap()`. Passing `packets_num=999999` removes the 5-packet limit. `seq_length=320` truncates at tokenization.
2. **No class imbalance handling:** Stratified splits + macro F1. Clean baseline.
3. **Same hyperparameters for all 4 targets:** Fair comparison.
4. **Checkpoint rename:** Zip contains `nomoe_bertflow_pre-trained_model.bin-120000`, extracted and renamed to `models/pretrained/pre-trained_model.bin`.
5. **Observability:** File artifacts (metrics.json, confusion_matrix.csv, predictions.tsv) + TensorBoard (train loss, dev/test metrics).

---

### File Map

**New files:**

| File | Responsibility |
|---|---|
| `tests/__init__.py` | Makes `tests/` a Python package for `unittest` discovery |
| `tests/test_behaviot_data_gen.py` | Tests config, manifest, feature extraction, splits, TSV generation |
| `tests/test_reporting_utils.py` | Tests metric computation, file writing, cross-fold aggregation |
| `tests/test_behaviot_runner.py` | Tests runner dry-run, command construction, checkpoint resolution |
| `data_generation/behaviot_data_gen.py` | Config, manifest, feature extraction (reuses `get_feature_flow`), k-fold splits, TSV writing |
| `uer/reporting_utils.py` | Metrics computation, file writing, cross-fold aggregation |
| `scripts/run_behaviot.py` | End-to-end runner with k-fold orchestration |
| `configs/behaviot/smoke.json` | Quick validation config (2 folds, 10 samples/class) |
| `configs/behaviot/full.json` | Full experiment config (5 folds, all samples) |
| `.gitignore` | Exclude generated data, results, and caches |

**Modified files:**

| File | Change |
|---|---|
| `fine-tuning/run_classifier.py:257` | Return `y_true, y_pred` from `evaluate()` |
| `fine-tuning/run_classifier.py:~290` | Add `--results_dir`, `--id_to_label_path`, `--tb_log_dir` optional args |
| `fine-tuning/run_classifier.py:362-379` | Add TensorBoard `SummaryWriter` for train loss and dev macro_f1 |
| `fine-tuning/run_classifier.py:388` | Capture test result, save reports + TB test metrics when `--results_dir` set |

**Label column summary** (7514 samples total):

| Target Column | Classes | Min/Max Samples | Notes |
|---|---|---|---|
| `device_type_label` | 9 | 39 – 1966 | |
| `device_label` | 50 | 6 – 603 | All classes ≥5 samples |
| `activity_type_label` | 11 | 90 – 1530 | |
| `activity_label` | 55 | 2 – 548 | `start` (2 samples) dropped by `min_samples_per_class=5` |

---

### Task 1: Project Scaffolding, Config, and Manifest Parsing

**Files:**
- Create: `tests/__init__.py`
- Create: `.gitignore`
- Create: `configs/behaviot/smoke.json`
- Create: `configs/behaviot/full.json`
- Create: `data_generation/behaviot_data_gen.py`
- Create: `tests/test_behaviot_data_gen.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty file) and `tests/test_behaviot_data_gen.py`:

```python
import csv
import json
import tempfile
import unittest
from pathlib import Path

from data_generation.behaviot_data_gen import (
    load_config,
    load_manifest,
    build_label_map,
)

SAMPLE_MANIFEST = [
    {"pcap_path": "/tmp/a.pcap", "device_type_label": "plug", "device_label": "amazon-plug", "activity_type_label": "off", "activity_label": "lan_off"},
    {"pcap_path": "/tmp/b.pcap", "device_type_label": "bulb", "device_label": "tp-bulb", "activity_type_label": "on", "activity_label": "lan_on"},
    {"pcap_path": "/tmp/c.pcap", "device_type_label": "plug", "device_label": "amazon-plug", "activity_type_label": "off", "activity_label": "lan_off"},
    {"pcap_path": "/tmp/d.pcap", "device_type_label": "bulb", "device_label": "tp-bulb", "activity_type_label": "on", "activity_label": "lan_on"},
    {"pcap_path": "/tmp/e.pcap", "device_type_label": "plug", "device_label": "gosund-plug", "activity_type_label": "off", "activity_label": "wan_off"},
    {"pcap_path": "/tmp/f.pcap", "device_type_label": "bulb", "device_label": "gosund-bulb", "activity_type_label": "audio", "activity_label": "alexa_audio"},
]


def _write_manifest(path, rows=SAMPLE_MANIFEST):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / "config.json"
        self.valid_config = {
            "manifest_csv": "/tmp/m.csv", "n_folds": 5,
            "results_root": "/tmp/r", "generated_dataset_root": "/tmp/g",
            "pretrained_model_zip": "/tmp/z.zip",
            "pretrained_model_path": "/tmp/m.bin",
            "vocab_path": "v.txt", "config_path": "c.json",
            "payload_length": 64, "start_index": 76,
            "seq_length": 320, "learning_rate": 6e-5, "batch_size": 128,
            "epochs_num": 4, "earlystop": 4, "seed": 42,
            "min_samples_per_class": 5, "max_samples_per_class": None,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_valid_config(self):
        self.config_path.write_text(json.dumps(self.valid_config))
        cfg = load_config(str(self.config_path))
        self.assertEqual(cfg["n_folds"], 5)

    def test_load_config_missing_key_raises(self):
        bad = {k: v for k, v in self.valid_config.items() if k != "n_folds"}
        self.config_path.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            load_config(str(self.config_path))


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "manifest.csv"
        _write_manifest(self.csv_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_manifest_returns_all_rows(self):
        rows = load_manifest(str(self.csv_path))
        self.assertEqual(len(rows), 6)

    def test_load_manifest_has_expected_columns(self):
        rows = load_manifest(str(self.csv_path))
        for col in ["pcap_path", "device_type_label", "device_label",
                     "activity_type_label", "activity_label"]:
            self.assertIn(col, rows[0])

    def test_build_label_map_sorted_alphabetically(self):
        rows = load_manifest(str(self.csv_path))
        lmap = build_label_map(rows, "activity_type_label")
        self.assertEqual(list(lmap.keys()), sorted(lmap.keys()))

    def test_build_label_map_assigns_contiguous_ids(self):
        rows = load_manifest(str(self.csv_path))
        lmap = build_label_map(rows, "device_type_label")
        self.assertEqual(lmap, {"bulb": 0, "plug": 1})

    def test_build_label_map_filters_rare_classes(self):
        rows = load_manifest(str(self.csv_path))
        lmap = build_label_map(rows, "activity_type_label", min_samples=2)
        self.assertIn("off", lmap)       # 3 samples
        self.assertIn("on", lmap)        # 2 samples
        self.assertNotIn("audio", lmap)  # 1 sample
        self.assertEqual(len(lmap), 2)

    def test_build_label_map_works_for_any_column(self):
        rows = load_manifest(str(self.csv_path))
        for col in ["device_type_label", "device_label",
                     "activity_type_label", "activity_label"]:
            lmap = build_label_map(rows, col)
            self.assertGreater(len(lmap), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'data_generation.behaviot_data_gen'`.

- [ ] **Step 3: Write minimal implementation**

Create `data_generation/behaviot_data_gen.py`:

```python
"""BehavIoT data generation: config, manifest, feature extraction, k-fold splits, TSV writing."""

import csv
import json
from collections import Counter
from pathlib import Path


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
```

Create `.gitignore`:

```
generated/
results/behaviot/
models/pretrained/
__pycache__/
*.pyc
```

Create `configs/behaviot/smoke.json`:

```json
{
    "manifest_csv": "/mnt/data/behavoiot/pcap_vs_label.csv",
    "n_folds": 2,
    "results_root": "results/behaviot",
    "generated_dataset_root": "generated/behaviot",
    "pretrained_model_zip": "/home/shinanliu/pre-trained_model.bin.zip",
    "pretrained_model_path": "models/pretrained/pre-trained_model.bin",
    "vocab_path": "models/encryptd_vocab.txt",
    "config_path": "models/bert/base_config.json",
    "payload_length": 64,
    "start_index": 76,
    "seq_length": 320,
    "learning_rate": 6e-5,
    "batch_size": 64,
    "epochs_num": 2,
    "earlystop": 2,
    "seed": 42,
    "max_samples_per_class": 10,
    "min_samples_per_class": 2
}
```

Create `configs/behaviot/full.json`:

```json
{
    "manifest_csv": "/mnt/data/behavoiot/pcap_vs_label.csv",
    "n_folds": 5,
    "results_root": "results/behaviot",
    "generated_dataset_root": "generated/behaviot",
    "pretrained_model_zip": "/home/shinanliu/pre-trained_model.bin.zip",
    "pretrained_model_path": "models/pretrained/pre-trained_model.bin",
    "vocab_path": "models/encryptd_vocab.txt",
    "config_path": "models/bert/base_config.json",
    "payload_length": 64,
    "start_index": 76,
    "seq_length": 320,
    "learning_rate": 6e-5,
    "batch_size": 128,
    "epochs_num": 4,
    "earlystop": 4,
    "seed": 42,
    "max_samples_per_class": null,
    "min_samples_per_class": 5
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen -v
```

Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add tests/__init__.py tests/test_behaviot_data_gen.py \
  data_generation/behaviot_data_gen.py \
  configs/behaviot/smoke.json configs/behaviot/full.json .gitignore
git commit -m "feat: add BehavIoT config, manifest parsing, and project scaffolding"
```


### Task 2: Feature Extraction with Caching

**Files:**
- Modify: `data_generation/behaviot_data_gen.py`
- Modify: `tests/test_behaviot_data_gen.py`

**Note:** `get_feature_flow()` already reads ALL packets from the pcap (all flows mixed via `scapy.rdpcap()`). By passing `packets_num=999999`, we remove the 5-packet limit. `seq_length=320` truncates at tokenization. No new wrapper function needed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_behaviot_data_gen.py`:

```python
from unittest.mock import patch
from data_generation.behaviot_data_gen import extract_all_features


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name) / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    @patch("data_generation.behaviot_data_gen._extract_single_pcap")
    def test_extract_returns_features_dict(self, mock_extract):
        mock_extract.side_effect = lambda args: (args[0], "45 00 3c [SEP] 45 00 28", None)
        rows = [{"pcap_path": "/tmp/a.pcap"}, {"pcap_path": "/tmp/b.pcap"}]
        config = {"payload_length": 64, "start_index": 76}
        features, skipped = extract_all_features(rows, config, n_workers=1)
        self.assertEqual(len(features), 2)
        self.assertEqual(len(skipped), 0)
        self.assertIn("/tmp/a.pcap", features)

    @patch("data_generation.behaviot_data_gen._extract_single_pcap")
    def test_extract_skips_failed_pcaps(self, mock_extract):
        mock_extract.side_effect = [
            ("/tmp/ok.pcap", "45 00 3c", None),
            ("/tmp/fail.pcap", None, "extraction_returned_-1"),
        ]
        rows = [{"pcap_path": "/tmp/ok.pcap"}, {"pcap_path": "/tmp/fail.pcap"}]
        config = {"payload_length": 64, "start_index": 76}
        features, skipped = extract_all_features(rows, config, n_workers=1)
        self.assertEqual(len(features), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["pcap_path"], "/tmp/fail.pcap")

    @patch("data_generation.behaviot_data_gen._extract_single_pcap")
    def test_extract_caches_results(self, mock_extract):
        mock_extract.side_effect = lambda args: (args[0], "45 00", None)
        rows = [{"pcap_path": "/tmp/a.pcap"}]
        config = {"payload_length": 64, "start_index": 76}
        f1, _ = extract_all_features(rows, config, cache_dir=str(self.cache_dir), n_workers=1)
        call_count_after_first = mock_extract.call_count
        f2, _ = extract_all_features(rows, config, cache_dir=str(self.cache_dir), n_workers=1)
        self.assertEqual(mock_extract.call_count, call_count_after_first)  # cache hit
        self.assertEqual(f1, f2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen.TestFeatureExtraction -v
```

Expected: FAIL with `ImportError` for `extract_all_features`.

- [ ] **Step 3: Write minimal implementation**

Add to `data_generation/behaviot_data_gen.py`:

```python
def _extract_single_pcap(args):
    """Worker function for multiprocessing. Module-level for pickling.

    Calls get_feature_flow() with packets_num=999999 so all packets from all flows
    are included. seq_length truncation happens at tokenization time.
    """
    pcap_path, payload_length, start_index = args
    try:
        from data_generation.finetuning_data_gen import get_feature_flow
        result = get_feature_flow(pcap_path, payload_length, packets_num=999999,
                                  start_index=start_index)
        if result == -1:
            return pcap_path, None, "extraction_returned_-1"
        return pcap_path, result[0], None  # result[0] is the datagram string
    except Exception as e:
        return pcap_path, None, str(e)


def extract_all_features(manifest_rows, config, cache_dir=None, n_workers=32):
    """Extract bigram features from all pcaps with multiprocessing and caching.

    Uses get_feature_flow() with no packet limit — all packets from all flows
    in each pcap are mixed into a single bigram datagram string.

    Args:
        manifest_rows: List of dicts with 'pcap_path' key.
        config: Dict with 'payload_length', 'start_index'.
        cache_dir: If set, cache extracted features to this directory.
        n_workers: Number of parallel workers (1 = serial, for testing).

    Returns:
        (features_dict, skipped_list) where features_dict maps pcap_path to datagram string.
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
        for args in args_list:
            pcap_path, datagram, error = _extract_single_pcap(args)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add data_generation/behaviot_data_gen.py tests/test_behaviot_data_gen.py
git commit -m "feat: add BehavIoT feature extraction with multiprocessing and caching"
```


### Task 3: Stratified 5-Fold Splitting and TSV Generation

**Files:**
- Modify: `data_generation/behaviot_data_gen.py`
- Modify: `tests/test_behaviot_data_gen.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_behaviot_data_gen.py`:

```python
from collections import Counter
from data_generation.behaviot_data_gen import (
    build_kfold_splits,
    write_fold_tsvs,
    cap_samples_per_class,
)


class TestKFoldSplit(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"pcap_path": f"/tmp/{i}.pcap", "label": "A" if i % 2 == 0 else "B"}
            for i in range(20)
        ]
        self.label_to_id = {"A": 0, "B": 1}

    def test_returns_correct_number_of_folds(self):
        folds = build_kfold_splits(self.rows, "label", self.label_to_id, n_folds=5, seed=42)
        self.assertEqual(len(folds), 5)

    def test_each_fold_has_train_dev_test(self):
        folds = build_kfold_splits(self.rows, "label", self.label_to_id, n_folds=5, seed=42)
        for train, dev, test in folds:
            self.assertGreater(len(train), 0)
            self.assertGreater(len(dev), 0)
            self.assertGreater(len(test), 0)

    def test_no_overlap_between_splits(self):
        folds = build_kfold_splits(self.rows, "label", self.label_to_id, n_folds=5, seed=42)
        for train, dev, test in folds:
            train_p = {r["pcap_path"] for r in train}
            dev_p = {r["pcap_path"] for r in dev}
            test_p = {r["pcap_path"] for r in test}
            self.assertEqual(len(train_p & test_p), 0)
            self.assertEqual(len(dev_p & test_p), 0)
            self.assertEqual(len(train_p & dev_p), 0)

    def test_all_samples_covered_across_test_folds(self):
        folds = build_kfold_splits(self.rows, "label", self.label_to_id, n_folds=5, seed=42)
        all_test = set()
        for _, _, test in folds:
            all_test.update(r["pcap_path"] for r in test)
        all_valid = {r["pcap_path"] for r in self.rows if r["label"] in self.label_to_id}
        self.assertEqual(all_test, all_valid)

    def test_filters_rows_with_invalid_labels(self):
        rows = self.rows + [{"pcap_path": "/tmp/unknown.pcap", "label": "UNKNOWN"}]
        folds = build_kfold_splits(rows, "label", self.label_to_id, n_folds=2, seed=42)
        all_paths = set()
        for train, dev, test in folds:
            for r in train + dev + test:
                all_paths.add(r["pcap_path"])
        self.assertNotIn("/tmp/unknown.pcap", all_paths)


class TestTSVGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name) / "fold_0"

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_fold_tsvs_creates_three_files(self):
        train = [{"pcap_path": "/tmp/a.pcap", "lbl": "A"}, {"pcap_path": "/tmp/b.pcap", "lbl": "B"}]
        dev = [{"pcap_path": "/tmp/c.pcap", "lbl": "A"}]
        test = [{"pcap_path": "/tmp/d.pcap", "lbl": "B"}]
        features = {"/tmp/a.pcap": "45 00", "/tmp/b.pcap": "45 01",
                    "/tmp/c.pcap": "45 02", "/tmp/d.pcap": "45 03"}
        label_to_id = {"A": 0, "B": 1}
        write_fold_tsvs((train, dev, test), features, label_to_id, "lbl", str(self.output_dir))
        self.assertTrue((self.output_dir / "train_dataset.tsv").exists())
        self.assertTrue((self.output_dir / "valid_dataset.tsv").exists())
        self.assertTrue((self.output_dir / "test_dataset.tsv").exists())

    def test_tsv_format_matches_run_classifier_expectations(self):
        train = [{"pcap_path": "/tmp/a.pcap", "lbl": "A"}]
        dev = [{"pcap_path": "/tmp/b.pcap", "lbl": "B"}]
        test = [{"pcap_path": "/tmp/c.pcap", "lbl": "A"}]
        features = {"/tmp/a.pcap": "45 00", "/tmp/b.pcap": "45 01", "/tmp/c.pcap": "45 02"}
        label_to_id = {"A": 0, "B": 1}
        write_fold_tsvs((train, dev, test), features, label_to_id, "lbl", str(self.output_dir))
        with open(self.output_dir / "train_dataset.tsv") as f:
            lines = f.readlines()
        self.assertEqual(lines[0].strip(), "label\ttext_a")
        parts = lines[1].strip().split("\t")
        self.assertEqual(parts[0], "0")      # label A -> id 0
        self.assertEqual(parts[1], "45 00")  # datagram string

    def test_skips_pcaps_without_features(self):
        train = [{"pcap_path": "/tmp/a.pcap", "lbl": "A"},
                 {"pcap_path": "/tmp/missing.pcap", "lbl": "B"}]
        features = {"/tmp/a.pcap": "45 00"}
        label_to_id = {"A": 0, "B": 1}
        write_fold_tsvs((train, [], []), features, label_to_id, "lbl", str(self.output_dir))
        with open(self.output_dir / "train_dataset.tsv") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # header + 1 row


class TestCapSamples(unittest.TestCase):
    def test_caps_large_classes(self):
        rows = [{"pcap_path": f"/tmp/a{i}.pcap", "lbl": "A"} for i in range(10)]
        rows += [{"pcap_path": f"/tmp/b{i}.pcap", "lbl": "B"} for i in range(5)]
        label_to_id = {"A": 0, "B": 1}
        capped = cap_samples_per_class(rows, "lbl", label_to_id, max_per_class=3, seed=42)
        counts = Counter(r["lbl"] for r in capped)
        self.assertEqual(counts["A"], 3)
        self.assertEqual(counts["B"], 3)

    def test_no_cap_when_under_max(self):
        rows = [{"pcap_path": f"/tmp/{i}.pcap", "lbl": "A"} for i in range(3)]
        label_to_id = {"A": 0}
        capped = cap_samples_per_class(rows, "lbl", label_to_id, max_per_class=10, seed=42)
        self.assertEqual(len(capped), 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen.TestKFoldSplit tests.test_behaviot_data_gen.TestTSVGeneration tests.test_behaviot_data_gen.TestCapSamples -v
```

Expected: FAIL with `ImportError` for missing functions.

- [ ] **Step 3: Write minimal implementation**

Add to `data_generation/behaviot_data_gen.py`:

```python
import random
from sklearn.model_selection import StratifiedKFold, train_test_split


def build_kfold_splits(rows, target_column, label_to_id, n_folds, seed):
    """Build stratified k-fold splits with inner train/dev split.

    For each fold: test = one fold, train+dev = remaining folds (90/10 inner split).

    Returns:
        List of (train_rows, dev_rows, test_rows) tuples, one per fold.
    """
    valid_rows = [r for r in rows if r[target_column] in label_to_id]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add data_generation/behaviot_data_gen.py tests/test_behaviot_data_gen.py
git commit -m "feat: add stratified 5-fold splitting and TSV generation"
```


### Task 4: Reporting Utilities

**Files:**
- Create: `uer/reporting_utils.py`
- Create: `tests/test_reporting_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reporting_utils.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from uer.reporting_utils import (
    compute_metrics,
    write_metrics_json,
    write_confusion_matrix_csv,
    write_predictions_tsv,
    aggregate_fold_metrics,
)


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        m = compute_metrics(y_true, y_pred, ["A", "B", "C"])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)

    def test_imperfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        m = compute_metrics(y_true, y_pred, ["A", "B"])
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertIn("per_class", m)

    def test_returns_all_expected_keys(self):
        m = compute_metrics([0, 1], [0, 1], ["A", "B"])
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
            self.assertIn(key, m)


class TestWriteFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_metrics_json_roundtrips(self):
        metrics = {"accuracy": 0.95, "macro_f1": 0.9}
        write_metrics_json(self.d / "m.json", metrics)
        loaded = json.loads((self.d / "m.json").read_text())
        self.assertEqual(loaded["accuracy"], 0.95)

    def test_write_confusion_matrix_has_labels(self):
        write_confusion_matrix_csv(self.d / "cm.csv", [0, 0, 1, 1], [0, 1, 1, 0], ["A", "B"])
        text = (self.d / "cm.csv").read_text()
        self.assertIn("A", text)
        self.assertIn("B", text)
        lines = text.strip().split("\n")
        self.assertEqual(len(lines), 3)  # header + 2 rows

    def test_write_predictions_tsv(self):
        write_predictions_tsv(self.d / "p.tsv", [0, 1, 0], [0, 0, 1], ["A", "B"])
        lines = (self.d / "p.tsv").read_text().strip().split("\n")
        self.assertEqual(len(lines), 4)  # header + 3 rows
        self.assertIn("true_label", lines[0])


class TestAggregation(unittest.TestCase):
    def test_aggregate_computes_mean_and_std(self):
        folds = [
            {"accuracy": 0.8, "macro_precision": 0.7, "macro_recall": 0.75,
             "macro_f1": 0.72, "weighted_f1": 0.78},
            {"accuracy": 0.9, "macro_precision": 0.85, "macro_recall": 0.88,
             "macro_f1": 0.86, "weighted_f1": 0.89},
        ]
        agg = aggregate_fold_metrics(folds)
        self.assertAlmostEqual(agg["accuracy"]["mean"], 0.85)
        self.assertEqual(len(agg["accuracy"]["per_fold"]), 2)
        self.assertIn("std", agg["accuracy"])
        self.assertGreater(agg["accuracy"]["std"], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_reporting_utils -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'uer.reporting_utils'`.

- [ ] **Step 3: Write minimal implementation**

Create `uer/reporting_utils.py`:

```python
"""Metrics computation, file writing, and cross-fold aggregation for BehavIoT experiments."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, label_names):
    """Compute classification metrics. Returns a dict of scalar metrics plus per-class report."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, target_names=label_names,
            output_dict=True, zero_division=0,
        ),
    }


def write_metrics_json(path, metrics):
    """Write metrics dict to JSON file."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def write_confusion_matrix_csv(path, y_true, y_pred, label_names):
    """Write confusion matrix as CSV with row/column label headers."""
    cm = confusion_matrix(y_true, y_pred)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_reporting_utils -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add uer/reporting_utils.py tests/test_reporting_utils.py
git commit -m "feat: add reporting utilities and cross-fold aggregation"
```


### Task 5: Modify run_classifier.py for Result Saving and TensorBoard

**Files:**
- Modify: `fine-tuning/run_classifier.py`

- [ ] **Step 1: Verify current code lacks new features**

Run:

```bash
cd /home/shinanliu/TrafficFormer
grep -n "results_dir\|id_to_label_path\|tb_log_dir\|SummaryWriter" fine-tuning/run_classifier.py
```

Expected: no matches.

- [ ] **Step 2: Run syntax check on current code**

```bash
cd /home/shinanliu/TrafficFormer
python3 -c "import ast; ast.parse(open('fine-tuning/run_classifier.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Apply four modifications**

**Modification A** — Change `evaluate()` return at line 257.

Replace:
```python
    return f1_score(y_true,y_pred,average='macro'), confusion
```

With:
```python
    return f1_score(y_true,y_pred,average='macro'), confusion, y_true, y_pred
```

**Modification B** — Add new arguments. Insert before line 292 (`args = parser.parse_args()`):

```python
    parser.add_argument("--results_dir", type=str, default=None,
                        help="If set, save metrics/predictions/confusion_matrix to this directory.")
    parser.add_argument("--id_to_label_path", type=str, default=None,
                        help="Path to id_to_label.json for human-readable metric names.")
    parser.add_argument("--tb_log_dir", type=str, default=None,
                        help="If set, write TensorBoard logs to this directory.")
```

**Modification C** — Add TensorBoard logging to training loop. After line 297 (`set_seed(args.seed)`), insert:

```python
    tb_writer = None
    if args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
```

Inside the training loop (line 362-379), add TensorBoard writes. After line 368 (the `total_loss` print statement), insert:

```python
                if tb_writer:
                    global_step = (epoch - 1) * (instances_num // batch_size) + (i + 1)
                    tb_writer.add_scalar("train/loss", total_loss / args.report_steps, global_step)
```

After line 372 (`result = evaluate(args, read_dataset(args, args.dev_path))`), insert:

```python
        if tb_writer:
            tb_writer.add_scalar("dev/macro_f1", result[0], epoch)
```

**Modification D** — Capture and save test results. Replace lines 382–389:

```python
    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path), True)
```

With:

```python
    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        result = evaluate(args, read_dataset(args, args.test_path), True)

        if args.results_dir:
            from pathlib import Path as _Path
            from uer.reporting_utils import (
                compute_metrics, write_metrics_json,
                write_confusion_matrix_csv, write_predictions_tsv,
            )
            rd = _Path(args.results_dir)
            rd.mkdir(parents=True, exist_ok=True)
            y_true_list, y_pred_list = result[2], result[3]
            label_names = [str(i) for i in range(args.labels_num)]
            if args.id_to_label_path:
                import json as _json
                id_to_label = _json.loads(_Path(args.id_to_label_path).read_text())
                label_names = [id_to_label[str(i)] for i in range(len(id_to_label))]
            metrics = compute_metrics(y_true_list, y_pred_list, label_names)
            write_metrics_json(rd / "metrics.json", metrics)
            write_confusion_matrix_csv(rd / "confusion_matrix.csv",
                                       y_true_list, y_pred_list, label_names)
            write_predictions_tsv(rd / "predictions.tsv",
                                  y_true_list, y_pred_list, label_names)
            if tb_writer:
                for key in ["accuracy", "macro_f1", "weighted_f1"]:
                    tb_writer.add_scalar(f"test/{key}", metrics[key])
            print(f"Results saved to {rd}")

    if tb_writer:
        tb_writer.close()
```

- [ ] **Step 4: Verify syntax after modification**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -c "import ast; ast.parse(open('fine-tuning/run_classifier.py').read()); print('OK')"
grep -c "results_dir\|tb_writer\|SummaryWriter" fine-tuning/run_classifier.py
```

Expected: `OK` and count ≥ 6.

- [ ] **Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add fine-tuning/run_classifier.py
git commit -m "feat: add --results_dir and TensorBoard logging to run_classifier.py"
```


### Task 6: End-to-End Runner with K-Fold Orchestration

**Files:**
- Create: `scripts/run_behaviot.py`
- Create: `tests/test_behaviot_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_behaviot_runner.py`:

```python
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class TestCheckpointResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.zip_path = self.d / "model.zip"
        self.bin_content = b"fake model data"
        with zipfile.ZipFile(self.zip_path, "w") as zf:
            zf.writestr("nomoe_bertflow_pre-trained_model.bin-120000", self.bin_content)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extracts_and_renames_bin_from_zip(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_behaviot import resolve_checkpoint
        model_path = self.d / "extracted" / "model.bin"
        resolve_checkpoint(str(self.zip_path), str(model_path))
        self.assertTrue(model_path.exists())
        self.assertEqual(model_path.read_bytes(), self.bin_content)

    def test_skips_if_already_exists(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_behaviot import resolve_checkpoint
        model_path = self.d / "existing" / "model.bin"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"already here")
        resolve_checkpoint(str(self.zip_path), str(model_path))
        self.assertEqual(model_path.read_bytes(), b"already here")


class TestRunnerDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.config = {
            "manifest_csv": "/mnt/data/behavoiot/pcap_vs_label.csv",
            "n_folds": 2,
            "results_root": str(self.d / "results"),
            "generated_dataset_root": str(self.d / "generated"),
            "pretrained_model_zip": "/home/shinanliu/pre-trained_model.bin.zip",
            "pretrained_model_path": str(self.d / "model.bin"),
            "vocab_path": "models/encryptd_vocab.txt",
            "config_path": "models/bert/base_config.json",
            "payload_length": 64, "start_index": 76,
            "seq_length": 320, "learning_rate": 6e-5, "batch_size": 128,
            "epochs_num": 2, "earlystop": 2, "seed": 42,
            "min_samples_per_class": 5, "max_samples_per_class": 10,
        }
        self.config_path = self.d / "config.json"
        self.config_path.write_text(json.dumps(self.config))

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_exits_cleanly(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_behaviot.py",
             "--config", str(self.config_path),
             "--target", "activity_type_label",
             "--dry_run"],
            capture_output=True, text=True,
            cwd="/home/shinanliu/TrafficFormer",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("[DRY RUN]", result.stdout)

    def test_dry_run_prints_target_column(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_behaviot.py",
             "--config", str(self.config_path),
             "--target", "device_type_label",
             "--dry_run"],
            capture_output=True, text=True,
            cwd="/home/shinanliu/TrafficFormer",
        )
        self.assertIn("device_type_label", result.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_runner -v
```

Expected: FAIL because `scripts/run_behaviot.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/run_behaviot.py`:

```python
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

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["results_root"]) / target / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    resolved = {**config, "target_label_column": target, "run_dir": str(run_dir),
                "timestamp": timestamp}
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2))

    if args.dry_run:
        print(f"[DRY RUN] Target: {target}")
        print(f"[DRY RUN] Run dir: {run_dir}")
        print(f"[DRY RUN] Config: {json.dumps(resolved, indent=2)}")
        return

    # Resolve pretrained checkpoint
    pretrained_path = resolve_checkpoint(config["pretrained_model_zip"],
                                         config["pretrained_model_path"])

    # Write environment info
    write_environment_json(run_dir / "environment.json")

    # Load manifest and extract features (cached — only runs once)
    rows = load_manifest(config["manifest_csv"])
    print(f"Loaded {len(rows)} manifest rows")

    cache_dir = Path(config["generated_dataset_root"]) / "cache"
    features, skipped = extract_all_features(rows, config, cache_dir=str(cache_dir))

    # Filter to rows with extracted features
    extractable_rows = [r for r in rows if r["pcap_path"] in features]
    print(f"Extractable samples: {len(extractable_rows)} / {len(rows)}")

    # Build label map from extractable rows
    label_to_id = build_label_map(extractable_rows, target, config["min_samples_per_class"])
    id_to_label = {v: k for k, v in label_to_id.items()}
    print(f"Classes for {target}: {len(label_to_id)}")

    # Save label maps
    (run_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2))
    id_to_label_path = run_dir / "id_to_label.json"
    id_to_label_path.write_text(json.dumps(id_to_label, indent=2))

    # Cap samples per class if configured
    working_rows = extractable_rows
    if config.get("max_samples_per_class"):
        working_rows = cap_samples_per_class(
            extractable_rows, target, label_to_id,
            config["max_samples_per_class"], config["seed"],
        )
        print(f"Capped to {len(working_rows)} samples (max {config['max_samples_per_class']}/class)")

    # Build k-fold splits
    folds = build_kfold_splits(working_rows, target, label_to_id,
                               config["n_folds"], config["seed"])
    print(f"Built {len(folds)} folds")

    # Run each fold
    log_path = run_dir / "run.log"
    fold_metrics = []

    with open(log_path, "w") as log_file:
        for fold_idx, fold_splits in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx}/{len(folds)-1}")
            print(f"{'='*60}")

            # Write per-fold TSVs
            data_dir = Path(config["generated_dataset_root"]) / target / f"fold_{fold_idx}"
            write_fold_tsvs(fold_splits, features, label_to_id, target, str(data_dir))
            train_count = sum(1 for _ in open(data_dir / "train_dataset.tsv")) - 1
            test_count = sum(1 for _ in open(data_dir / "test_dataset.tsv")) - 1
            print(f"  Train: {train_count}, Test: {test_count}")

            # Build and run classifier
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

            # Collect fold metrics
            metrics_path = fold_results_dir / "metrics.json"
            if metrics_path.exists():
                fold_metrics.append(json.loads(metrics_path.read_text()))
                print(f"  Fold {fold_idx} macro_f1: {fold_metrics[-1]['macro_f1']:.4f}")

    # Aggregate cross-fold results
    if fold_metrics:
        aggregated = aggregate_fold_metrics(fold_metrics)
        (run_dir / "aggregated_metrics.json").write_text(json.dumps(aggregated, indent=2))
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS ({len(fold_metrics)} folds)")
        print(f"{'='*60}")
        for key in ["accuracy", "macro_f1", "weighted_f1"]:
            m = aggregated[key]
            print(f"  {key}: {m['mean']:.4f} +/- {m['std']:.4f}")
        print(f"\nTensorBoard: tensorboard --logdir {run_dir / 'tb_logs'}")
    else:
        print("WARNING: No fold metrics collected. Check run.log for errors.")

    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_runner -v
python3 scripts/run_behaviot.py --config configs/behaviot/smoke.json --target activity_type_label --dry_run
```

Expected:
- unit tests PASS
- dry run prints resolved config with `activity_type_label` and exits

- [ ] **Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add scripts/run_behaviot.py tests/test_behaviot_runner.py
git commit -m "feat: add end-to-end BehavIoT runner with k-fold orchestration and TensorBoard"
```


### Task 7: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Verify current README lacks BehavIoT section**

Run:

```bash
cd /home/shinanliu/TrafficFormer
grep -n "BehavIoT\|run_behaviot\|behaviot" README.md
```

Expected: no matches.

- [ ] **Step 2: Add BehavIoT workflow section**

Append a new section to `README.md`:

````markdown
## BehavIoT Multi-Label Fine-Tuning

Fine-tune TrafficFormer on the BehavIoT dataset with 5-fold stratified cross-validation across four label targets.

**Data source:** `/mnt/data/behavoiot/pcap_vs_label.csv` (7514 samples, 4 label columns)

| Target Column | Classes | Description |
|---|---|---|
| `device_type_label` | 9 | Device category (plug, bulb, camera, ...) |
| `device_label` | 50 | Specific device model |
| `activity_type_label` | 11 | Activity category (on, off, audio, ...) |
| `activity_label` | 55 | Specific activity (android_lan_on, alexa_audio, ...) |

### Quick smoke test (2 folds, 10 samples/class)

```bash
python3 scripts/run_behaviot.py --config configs/behaviot/smoke.json --target activity_type_label
```

### Full experiment (5-fold CV, all samples)

```bash
python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target device_type_label
python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target device_label
python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target activity_type_label
python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target activity_label
```

### TensorBoard

```bash
tensorboard --logdir results/behaviot/<target>/<timestamp>/tb_logs
```

### Output artifacts

Each run creates `results/behaviot/<target>/<timestamp>/` containing:

- `resolved_config.json`, `environment.json` — config and environment
- `label_to_id.json`, `id_to_label.json` — label mappings
- `fold_0/` through `fold_4/` — per-fold: `metrics.json`, `confusion_matrix.csv`, `predictions.tsv`, `finetuned_model.bin`
- `tb_logs/` — TensorBoard logs (train loss, dev/test metrics per fold)
- `aggregated_metrics.json` — mean +/- std across folds
- `run.log` — full training output
````

- [ ] **Step 3: Verify documentation added**

Run:

```bash
cd /home/shinanliu/TrafficFormer
grep -c "BehavIoT\|run_behaviot" README.md
```

Expected: >= 5 matches.

- [ ] **Step 4: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add README.md
git commit -m "docs: add BehavIoT multi-label k-fold workflow"
```


### Task 8: Smoke Experiment

**Files:**
- Use: `configs/behaviot/smoke.json`
- Output: `results/behaviot/activity_type_label/<timestamp>/`

- [ ] **Step 1: Run all unit tests**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen tests.test_reporting_utils tests.test_behaviot_runner -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run the smoke pipeline**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 scripts/run_behaviot.py --config configs/behaviot/smoke.json --target activity_type_label
```

Expected:
- Feature extraction runs (or loads from cache)
- 2 folds train and evaluate
- Aggregated metrics printed
- TensorBoard logs created

- [ ] **Step 3: Verify artifacts**

Run:

```bash
cd /home/shinanliu/TrafficFormer
latest=$(find results/behaviot/activity_type_label -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
echo "Run dir: $latest"
ls -la "$latest"/
ls -la "$latest"/fold_0/
ls -la "$latest"/tb_logs/
cat "$latest"/aggregated_metrics.json
```

Expected files:
- `resolved_config.json`, `environment.json`, `label_to_id.json`, `id_to_label.json`
- `fold_0/metrics.json`, `fold_0/confusion_matrix.csv`, `fold_0/predictions.tsv`
- `fold_1/` with same structure
- `tb_logs/fold_0/`, `tb_logs/fold_1/`
- `aggregated_metrics.json` with mean/std
- `run.log`

- [ ] **Step 4: Verify TensorBoard loads**

Run:

```bash
cd /home/shinanliu/TrafficFormer
latest=$(find results/behaviot/activity_type_label -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('$latest/tb_logs/fold_0')
ea.Reload()
print('Tags:', ea.Tags())
print('TensorBoard logs OK')
"
```

Expected: prints available tags (train/loss, dev/macro_f1).


### Task 9: Full Experiments (All 4 Label Targets)

**Files:**
- Use: `configs/behaviot/full.json`
- Output: `results/behaviot/<target>/<timestamp>/` for each target

- [ ] **Step 1: Run all 4 label experiments**

Run each sequentially (feature extraction is cached after the first run):

```bash
cd /home/shinanliu/TrafficFormer
python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target device_type_label
python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target device_label
python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target activity_type_label
python3 scripts/run_behaviot.py --config configs/behaviot/full.json --target activity_label
```

Expected: each completes 5-fold training and evaluation.

- [ ] **Step 2: Verify all results exist**

Run:

```bash
cd /home/shinanliu/TrafficFormer
for target in device_type_label device_label activity_type_label activity_label; do
    latest=$(find "results/behaviot/$target" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
    if [ -n "$latest" ]; then
        echo "=== $target ==="
        cat "$latest/aggregated_metrics.json" | python3 -c "
import json,sys; d=json.load(sys.stdin)
for k in ['accuracy','macro_f1','weighted_f1']:
    print(f'  {k}: {d[k][\"mean\"]:.4f} +/- {d[k][\"std\"]:.4f}')
"
    else
        echo "=== $target === MISSING"
    fi
done
```

Expected: summary table with mean +/- std for all 4 targets.

- [ ] **Step 3: Launch TensorBoard for comparison**

Run:

```bash
tensorboard --logdir results/behaviot/ --port 6006
```

Expected: TensorBoard UI shows all folds across all targets.

- [ ] **Step 4: Final test verification**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen tests.test_reporting_utils tests.test_behaviot_runner -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit any code fixes discovered during experiments**

```bash
cd /home/shinanliu/TrafficFormer
git status --short
# Stage and commit only code/config changes, NOT generated/ or results/ directories
```
