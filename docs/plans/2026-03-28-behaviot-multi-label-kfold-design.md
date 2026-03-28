# BehavIoT Multi-Label 5-Fold CV — Experiment Design

**Date:** 2026-03-28
**Status:** Approved

## Goal

Fine-tune TrafficFormer on all four BehavIoT label columns with 5-fold stratified cross-validation, producing per-fold and aggregated metrics with TensorBoard logging.

## Experiment Design

**Independent variable:** Target label column (4 levels)
**Dependent variables:** Macro F1, accuracy, weighted F1 (mean ± std across 5 folds)
**Control variables:** Model architecture, hyperparameters, preprocessing, pretrained checkpoint

### Label Targets

| Target Column | Classes | Sample Range | Notes |
|---|---|---|---|
| `device_type_label` | 9 | 39 – 1966 | |
| `device_label` | 50 | 6 – 603 | |
| `activity_type_label` | 11 | 90 – 1530 | |
| `activity_label` | 55 | 2 – 548 | `start` (2 samples) dropped by min_samples filter |

### Data Source

- Manifest: `/mnt/data/behavoiot/pcap_vs_label.csv` (7514 samples)
- Pretrained checkpoint: `/home/shinanliu/pre-trained_model.bin.zip` → extracted and renamed to `models/pretrained/pre-trained_model.bin`

## Key Design Decisions

### 1. Multi-flow pcap handling: Mix all flows

Each BehavIoT pcap is a ~38-second capture session that may contain multiple TCP/UDP flows. Rather than picking one flow, **all flows are mixed into a single sample**: packets from every flow are concatenated with `[SEP]` separators into one bigram datagram string.

Implementation: New `get_all_flows_feature()` wrapper in `behaviot_data_gen.py` that reuses low-level utilities (`random_ip_port`, `random_tls_randomtime`, `bigram_generation`) from `finetuning_data_gen.py` but iterates all flows. Original `get_feature_flow()` is not modified.

### 2. No packet limit at extraction

No `packets_num` cap during extraction. `seq_length=320` truncation at tokenization time handles length.

### 3. Class imbalance: No special handling

Baseline approach. Stratified splits preserve class ratios. Macro F1 surfaces per-class issues. No class weights, no oversampling, no augmentation.

### 4. Same hyperparameters for all 4 targets

Fair comparison across targets:
- `seq_length=320`, `learning_rate=6e-5`, `batch_size=128`
- `epochs_num=4`, `earlystop=4`, `seed=42`
- Embedding: `word_pos_seg`, Encoder: `transformer`, Mask: `fully_visible`

### 5. Rare class filter: `min_samples_per_class=5`

Classes with fewer than 5 samples are dropped. Only affects `start` (2 samples) in `activity_label`.

### 6. Pretrained checkpoint: Rename on extraction

Zip contains `nomoe_bertflow_pre-trained_model.bin-120000`. Extracted and renamed to `models/pretrained/pre-trained_model.bin` for clean config paths.

### 7. Feature extraction: 32-worker multiprocessing with caching

32 workers (matching CPU core count). Results cached to `generated/behaviot/cache/features_cache.json`. Runs once (~5 min), reused across all targets and folds.

### 8. Observability: File artifacts + TensorBoard

- **Files:** `metrics.json`, `confusion_matrix.csv`, `predictions.tsv` per fold; `aggregated_metrics.json` per target
- **TensorBoard:** Training loss per step, dev macro_f1 per epoch, test metrics per fold. Logs at `results/behaviot/<target>/<timestamp>/tb_logs/`

## Architecture Changes

### New code

| Component | File | Description |
|---|---|---|
| Config/manifest parsing | `data_generation/behaviot_data_gen.py` | `load_config()`, `load_manifest()`, `build_label_map()` |
| All-flows feature extraction | `data_generation/behaviot_data_gen.py` | `get_all_flows_feature()` — wrapper using existing utilities |
| Cached multiprocessing extraction | `data_generation/behaviot_data_gen.py` | `extract_all_features()` — 32 workers, JSON cache |
| K-fold splitting | `data_generation/behaviot_data_gen.py` | `build_kfold_splits()`, `write_fold_tsvs()`, `cap_samples_per_class()` |
| Reporting utilities | `uer/reporting_utils.py` | `compute_metrics()`, `write_*()`, `aggregate_fold_metrics()` |
| Runner | `scripts/run_behaviot.py` | End-to-end: config → features → splits → training → aggregation |
| Configs | `configs/behaviot/smoke.json`, `full.json` | Smoke (2 folds, 10/class) and full (5 folds, all) |

### Modifications to existing code

| File | Change | Backward Compatible |
|---|---|---|
| `fine-tuning/run_classifier.py:257` | `evaluate()` returns `y_true, y_pred` (appended to tuple) | Yes — callers use `result[0]` |
| `fine-tuning/run_classifier.py:~290` | Add `--results_dir`, `--id_to_label_path` optional args | Yes — default `None` |
| `fine-tuning/run_classifier.py:388` | Capture test result, save reports when `--results_dir` set | Yes — gated |
| `fine-tuning/run_classifier.py` | Add `SummaryWriter` TensorBoard logging | Yes — gated behind `--results_dir` |

### Not modified

`finetuning_data_gen.py`, `pretrain_data_gen.py`, `utils.py`, UER framework core.

## CV Strategy

- 5-fold `StratifiedKFold(shuffle=True, random_state=42)`
- Each fold: test = 1 fold (20%), train+dev = 4 folds split 90/10
- Per fold: ~72% train, ~8% dev (early stopping), ~20% test
- All test folds are disjoint and cover the full dataset

## Validation Pyramid

| Level | Enabled | Config |
|---|---|---|
| L0: Static Checks | Yes | Device consistency, precision, optimizer, TensorBoard logging, 15 advisory |
| L1: Runtime Validation | Yes | Real pcap data, smoke config, 5-min budget |
| L2: E2E Pipeline | Yes | 3-5 steps per stage |

## Evaluation Structure

- Trainer owns **when**: dev eval every epoch, test eval after early stopping
- Per-fold outputs: `metrics.json`, `confusion_matrix.csv`, `predictions.tsv`, `finetuned_model.bin`, TensorBoard logs
- Per-target outputs: `aggregated_metrics.json` (mean ± std)
- Failure handling: failed fold is logged and skipped, runner continues, warns if <5 folds completed

## Hardware

- 2x NVIDIA RTX 5090 (32GB each)
- 32 CPU cores
- PyTorch 2.10 + CUDA 12.8

## Output Directory Structure

```
results/behaviot/<target>/<timestamp>/
├── resolved_config.json
├── environment.json
├── label_to_id.json
├── id_to_label.json
├── tb_logs/                    # TensorBoard logs
├── fold_0/
│   ├── metrics.json
│   ├── confusion_matrix.csv
│   ├── predictions.tsv
│   └── finetuned_model.bin
├── fold_1/ ... fold_4/
├── aggregated_metrics.json
└── run.log
```
