# BehavIoT Activity-Type Fine-Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Adapt TrafficFormer to fine-tune and evaluate on BehavIoT `activity_type_label` using `/mnt/data/behavoiot/pcap_vs_label.csv` as the single source of truth, with one original `.pcap` as one sample, stratified train/valid/test splits, pretrained initialization, and saved reports/logs.

**Architecture:** Add a config-driven BehavIoT path alongside the existing TrafficFormer pipeline. A new BehavIoT dataset generator will read the manifest CSV, build stratified splits, convert each source `.pcap` into the existing datagram representation, and write TrafficFormer-compatible TSV files plus label metadata. A wrapper runner will resolve the official pretrained checkpoint if missing, launch data preparation and fine-tuning, and write run artifacts into `results/behaviot_activity_type/<timestamp>/`. The existing folder-based generators stay intact.

**Tech Stack:** Python 3.12, pandas, scapy, flowcontainer, scikit-learn, PyTorch, JSON configs, existing TrafficFormer/UER code, shell subprocess orchestration.

---

### Task 1: Add BehavIoT Config And Manifest Parsing

**Files:**
- Create: `tests/test_behaviot_finetuning_data_gen.py`
- Create: `data_generation/behaviot_finetuning_data_gen.py`
- Create: `configs/behaviot/activity_type_smoke.json`
- Create: `configs/behaviot/activity_type_full.json`

**Step 1: Write the failing tests**

Create `tests/test_behaviot_finetuning_data_gen.py` with a tiny temporary manifest fixture and tests for:
- loading a JSON config file
- validating required config keys
- loading `/mnt/data/behavoiot/pcap_vs_label.csv`-shaped data
- extracting the target label column `activity_type_label`
- producing a deterministic label-to-id mapping for all 11 classes

Example test skeleton:

```python
import json
import tempfile
import unittest
from pathlib import Path

from data_generation.behaviot_finetuning_data_gen import (
    load_behaviot_config,
    load_manifest,
    build_label_map,
)


class BehavIoTConfigTests(unittest.TestCase):
    def test_build_label_map_orders_labels_stably(self):
        rows = [
            {"pcap_path": "/tmp/a.pcap", "activity_type_label": "off"},
            {"pcap_path": "/tmp/b.pcap", "activity_type_label": "on"},
            {"pcap_path": "/tmp/c.pcap", "activity_type_label": "audio"},
        ]
        label_to_id = build_label_map(rows, target_column="activity_type_label")
        self.assertEqual(set(label_to_id), {"audio", "off", "on"})
        self.assertEqual(label_to_id["audio"], 0)
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_finetuning_data_gen -v
```

Expected: FAIL with `ModuleNotFoundError` or missing function errors for `behaviot_finetuning_data_gen`.

**Step 3: Write minimal implementation**

Create `data_generation/behaviot_finetuning_data_gen.py` with:
- `load_behaviot_config(path)`
- `load_manifest(csv_path)`
- `build_label_map(rows, target_column)`
- config validation for:
  - `manifest_csv`
  - `target_label_column`
  - `results_root`
  - `generated_dataset_root`
  - `pretrained_model_path`
  - `train_ratio`
  - `valid_ratio`
  - `test_ratio`
- default assumptions:
  - manifest source is `/mnt/data/behavoiot/pcap_vs_label.csv`
  - target column is `activity_type_label`
  - one `.pcap` equals one sample

Create these JSON configs:
- `configs/behaviot/activity_type_smoke.json`
- `configs/behaviot/activity_type_full.json`

The smoke config should point to the same manifest but include a per-class sample cap for fast validation. The full config should use the entire manifest.

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_finetuning_data_gen -v
```

Expected: PASS.

**Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add tests/test_behaviot_finetuning_data_gen.py data_generation/behaviot_finetuning_data_gen.py configs/behaviot/activity_type_smoke.json configs/behaviot/activity_type_full.json
git commit -m "feat: add BehavIoT config and manifest parsing"
```


### Task 2: Generate Stratified BehavIoT Train/Valid/Test TSVs

**Files:**
- Modify: `data_generation/behaviot_finetuning_data_gen.py`
- Create: `tests/test_behaviot_tsv_generation.py`

**Step 1: Write the failing test**

Create `tests/test_behaviot_tsv_generation.py` that:
- monkeypatches or injects a fake feature extractor in place of `get_feature_flow`
- feeds a temporary manifest with at least 2 samples per class
- verifies the generator:
  - performs stratified train/valid/test split
  - writes `train_dataset.tsv`, `valid_dataset.tsv`, `test_dataset.tsv`
  - writes `label_to_id.json`, `id_to_label.json`
  - writes split manifests such as `train_manifest.csv`, `valid_manifest.csv`, `test_manifest.csv`

Example test intent:

```python
self.assertTrue((output_dir / "train_dataset.tsv").exists())
self.assertTrue((output_dir / "label_to_id.json").exists())
self.assertEqual(header, ["label", "text_a"])
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_tsv_generation -v
```

Expected: FAIL because split/build functions do not exist yet.

**Step 3: Write minimal implementation**

Extend `data_generation/behaviot_finetuning_data_gen.py` with functions like:
- `build_stratified_splits(rows, target_column, train_ratio, valid_ratio, test_ratio, seed)`
- `generate_behaviot_dataset(config)`
- `materialize_split_tsv(split_rows, output_dir, label_to_id, feature_extractor)`

Implementation rules:
- Use `activity_type_label` from the manifest, not folder names.
- Keep one original `.pcap` as one sample.
- Reuse the existing `get_feature_flow()` logic from `data_generation/finetuning_data_gen.py` to produce the datagram string.
- Skip rows whose pcap feature extraction returns `-1`, but log them to `skipped_samples.csv`.
- Save numeric labels in TSV, plus label maps for decoding predictions.

Write outputs under a config-selected directory such as:

```text
/home/shinanliu/TrafficFormer/generated/behaviot_activity_type/<run_name>/
```

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_tsv_generation -v
```

Expected: PASS.

**Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add tests/test_behaviot_tsv_generation.py data_generation/behaviot_finetuning_data_gen.py
git commit -m "feat: generate BehavIoT stratified TSV datasets"
```


### Task 3: Resolve Or Download The Official Pretrained Checkpoint

**Files:**
- Create: `tests/test_behaviot_runner.py`
- Create: `scripts/run_behaviot_activity_type.py`

**Step 1: Write the failing test**

Create `tests/test_behaviot_runner.py` with a dry-run test that verifies:
- the runner loads the JSON config
- it resolves an existing pretrained path when present
- it falls back to extracting from `/home/shinanliu/pre-trained_model.bin.zip` when the local checkpoint is missing
- it writes a resolved run config into the chosen results directory

Use a `--dry_run` flag so the test does not download or train.

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_runner -v
```

Expected: FAIL because the runner script and resolution helpers do not exist.

**Step 3: Write minimal implementation**

Create `scripts/run_behaviot_activity_type.py` with logic to:
- load the selected config
- compute a timestamped run directory under:
  - `results/behaviot_activity_type/<timestamp>/`
- resolve `pretrained_model_path`
- if missing, extract from the local zip archive at `/home/shinanliu/pre-trained_model.bin.zip` into:
  - `models/pretrained/pre-trained_model.bin`
- write `resolved_config.json` into the run directory
- support `--dry_run`

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_runner -v
python3 scripts/run_behaviot_activity_type.py --config configs/behaviot/activity_type_smoke.json --dry_run
```

Expected:
- unit test PASS
- dry run prints resolved paths and exits without training

**Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add tests/test_behaviot_runner.py scripts/run_behaviot_activity_type.py
git commit -m "feat: add config-driven BehavIoT runner and checkpoint resolution"
```


### Task 4: Save Full Reports, Predictions, And Logs From Fine-Tuning

**Files:**
- Create: `uer/reporting_utils.py`
- Modify: `fine-tuning/run_classifier.py`
- Create: `tests/test_reporting_utils.py`

**Step 1: Write the failing test**

Create `tests/test_reporting_utils.py` that passes fake:
- `y_true`
- `y_pred`
- logits
- label names
- sample identifiers

and asserts that the report writer creates:
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `predictions.tsv`

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_reporting_utils -v
```

Expected: FAIL because `uer.reporting_utils` does not exist.

**Step 3: Write minimal implementation**

Create `uer/reporting_utils.py` with helpers such as:
- `compute_metrics(y_true, y_pred, label_names)`
- `write_confusion_matrix(path, confusion, label_names)`
- `write_predictions(path, sample_ids, y_true, y_pred, logits, label_names)`
- `write_metrics_json(path, metrics_dict)`

Modify `fine-tuning/run_classifier.py` to add optional CLI args:
- `--results_dir`
- `--label_names_path`
- `--sample_manifest_path`

Then:
- during dev/test evaluation, save metrics and reports if `results_dir` is set
- write human-readable logs to stdout as before
- keep old behavior intact when new args are omitted

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_reporting_utils -v
```

Expected: PASS.

**Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add uer/reporting_utils.py fine-tuning/run_classifier.py tests/test_reporting_utils.py
git commit -m "feat: save fine-tuning reports and prediction artifacts"
```


### Task 5: Wire The BehavIoT Runner End To End

**Files:**
- Modify: `scripts/run_behaviot_activity_type.py`
- Modify: `data_generation/behaviot_finetuning_data_gen.py`
- Modify: `configs/behaviot/activity_type_smoke.json`
- Modify: `configs/behaviot/activity_type_full.json`

**Step 1: Write the failing integration check**

Add or extend `tests/test_behaviot_runner.py` to verify that the runner, in dry-run mode, constructs:
- dataset-generation outputs
- `fine-tuning/run_classifier.py` command
- `results/behaviot_activity_type/<timestamp>/run.log`
- `results/behaviot_activity_type/<timestamp>/resolved_config.json`

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_runner -v
```

Expected: FAIL because the runner does not yet orchestrate both data generation and training.

**Step 3: Write minimal implementation**

Complete `scripts/run_behaviot_activity_type.py` so it:
- calls the BehavIoT TSV generator
- captures skipped samples
- launches `fine-tuning/run_classifier.py` with:
  - `--train_path`
  - `--dev_path`
  - `--test_path`
  - `--pretrained_model_path`
  - `--vocab_path models/encryptd_vocab.txt`
  - `--config_path models/bert/base_config.json`
  - `--results_dir <run_dir>`
  - `--label_names_path <generated label map>`
  - `--sample_manifest_path <generated split manifest>`
- tees subprocess stdout/stderr to:
  - `results/.../run.log`

Also write:
- `environment.json` with Python version, torch version, GPU summary, config path, manifest path

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_runner -v
python3 scripts/run_behaviot_activity_type.py --config configs/behaviot/activity_type_smoke.json --dry_run
```

Expected:
- PASS
- dry run shows end-to-end command construction with resolved output paths

**Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add scripts/run_behaviot_activity_type.py data_generation/behaviot_finetuning_data_gen.py configs/behaviot/activity_type_smoke.json configs/behaviot/activity_type_full.json tests/test_behaviot_runner.py
git commit -m "feat: wire BehavIoT activity-type training pipeline"
```


### Task 6: Document The BehavIoT Workflow

**Files:**
- Modify: `README.md`

**Step 1: Write the failing documentation checklist**

Create a short checklist in your scratchpad and verify the README currently does not explain:
- BehavIoT manifest source
- config paths
- smoke run command
- full run command
- output artifacts location

**Step 2: Run the checklist**

Run:

```bash
cd /home/shinanliu/TrafficFormer
rg -n "BehavIoT|activity_type_label|run_behaviot_activity_type|results/behaviot_activity_type" README.md
```

Expected: no complete BehavIoT workflow section yet.

**Step 3: Write minimal implementation**

Add a new README section that documents:
- manifest source: `/mnt/data/behavoiot/pcap_vs_label.csv`
- target label: `activity_type_label`
- smoke command
- full command
- pretrained model download behavior
- saved outputs:
  - generated TSVs
  - label maps
  - metrics JSON
  - confusion matrix CSV
  - per-class metrics CSV
  - predictions TSV
  - run log

**Step 4: Run the checklist again**

Run:

```bash
cd /home/shinanliu/TrafficFormer
rg -n "BehavIoT|activity_type_label|run_behaviot_activity_type|results/behaviot_activity_type" README.md
```

Expected: matches the new documentation.

**Step 5: Commit**

```bash
cd /home/shinanliu/TrafficFormer
git add README.md
git commit -m "docs: add BehavIoT activity-type workflow"
```


### Task 7: Run A Smoke Experiment And Verify Artifacts

**Files:**
- Use: `configs/behaviot/activity_type_smoke.json`
- Output: `results/behaviot_activity_type/<timestamp>/`

**Step 1: Run the smoke pipeline**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m pip install -r requirements.txt
python3 scripts/run_behaviot_activity_type.py --config configs/behaviot/activity_type_smoke.json
```

Expected:
- generated TSVs exist
- pretrained checkpoint exists
- training completes
- test evaluation runs

**Step 2: Verify report artifacts**

Run:

```bash
cd /home/shinanliu/TrafficFormer
find results/behaviot_activity_type -maxdepth 2 -type f | sort | tail -n 40
```

Expected files in the latest run directory:
- `run.log`
- `resolved_config.json`
- `environment.json`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `predictions.tsv`

**Step 3: Verify the metrics file is machine-readable**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 - <<'PY'
import json, pathlib
latest = sorted(pathlib.Path("results/behaviot_activity_type").glob("*"))[-1]
print(json.loads((latest / "metrics.json").read_text())["macro_f1"])
PY
```

Expected: prints a numeric macro F1.

**Step 4: Commit code changes only**

```bash
cd /home/shinanliu/TrafficFormer
git status --short
```

Expected: code/docs/config files staged or committed; do **not** commit large generated datasets or result directories.


### Task 8: Run The Full BehavIoT Experiment

**Files:**
- Use: `configs/behaviot/activity_type_full.json`
- Output: `results/behaviot_activity_type/<timestamp>/`

**Step 1: Launch the full run**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 scripts/run_behaviot_activity_type.py --config configs/behaviot/activity_type_full.json
```

Expected: end-to-end data generation, training, and test evaluation on all 11 `activity_type_label` classes.

**Step 2: Verify final outputs**

Run:

```bash
cd /home/shinanliu/TrafficFormer
latest=$(find results/behaviot_activity_type -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
echo "$latest"
ls -lah "$latest"
```

Expected:
- final checkpoint
- metrics and confusion matrix
- predictions TSV
- run log
- copied config/label map

**Step 3: Summarize the run**

Run:

```bash
cd /home/shinanliu/TrafficFormer
latest=$(find results/behaviot_activity_type -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
python3 - <<'PY'
import json, pathlib, os
latest = pathlib.Path(os.environ["LATEST"])
print((latest / "metrics.json").read_text())
PY
```

Run with:

```bash
export LATEST=$(find /home/shinanliu/TrafficFormer/results/behaviot_activity_type -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
```

Expected: JSON summary ready to paste into a final report.

**Step 4: Final verification**

Run:

```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_finetuning_data_gen tests.test_behaviot_tsv_generation tests.test_behaviot_runner tests.test_reporting_utils -v
```

Expected: all custom BehavIoT tests PASS after the full run work is complete.
