# BehavIoT Multi-Target 5-Fold Cross-Validation Plan

## Goal Description

Adapt TrafficFormer to fine-tune and evaluate on four separate BehavIoT classification targets (`device_type_label`, `device_label`, `activity_type_label`, `activity_label`) from `/mnt/data/behavoiot/pcap_vs_label.csv`, using 5-fold stratified cross-validation with pretrained initialization. Each target is an independent single-label multi-class classification experiment using identical hyperparameters. The pipeline produces per-fold metrics, TensorBoard logs, and cross-fold aggregated results.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Feature extraction caches all valid pcap bigrams and logs failures
  - Positive Tests (expected to PASS):
    - Extracting features from a valid pcap returns a non-empty bigram datagram string
    - Re-running extraction with a populated cache directory returns the same features without calling `get_feature_flow()`
    - Skipped samples are written to `skipped_samples.csv` with pcap path and reason
  - Negative Tests (expected to FAIL):
    - Extracting from an IPv6-only pcap returns a skip entry, not a crash
    - Extracting from a pcap with fewer than 3 packets returns a skip entry
  - AC-1.1: Feature extraction uses `get_feature_flow()` with `packets_num=999999` (no packet limit; `seq_length=320` truncates at tokenization)
    - Positive: Datagram string contains bigrams from all packets in the pcap
    - Negative: Datagram string is not truncated to 5 packets at extraction time

- AC-2: Stratified 5-fold splits with feasibility validation
  - Positive Tests (expected to PASS):
    - 5-fold split produces exactly 5 (train, dev, test) tuples
    - Each sample appears in exactly one test fold across all folds
    - No sample appears in both train and test within any fold
    - No sample appears in both dev and test within any fold
    - Train/dev inner split uses 90/10 ratio with stratification
  - Negative Tests (expected to FAIL):
    - A class with fewer than `n_folds` samples after extraction causes a clear error, not a silent split
    - A label not in `label_to_id` does not appear in any fold
  - AC-2.1: Split feasibility is validated before fold generation
    - Positive: Pipeline fails fast with descriptive error if any class has fewer extractable samples than `n_folds`
    - Negative: Pipeline does not silently produce empty or degenerate folds

- AC-3: `run_classifier.py` modifications are backward-compatible
  - Positive Tests (expected to PASS):
    - `evaluate()` returns `(macro_f1, confusion, y_true, y_pred)` — a 4-tuple
    - Existing callers accessing `result[0]` for macro_f1 still work
    - When `--results_dir` is omitted, behavior is identical to unmodified code
    - When `--results_dir` is set, `metrics.json`, `confusion_matrix.csv`, and `predictions.tsv` are written
    - When `--tb_log_dir` is set, TensorBoard event files are written with `train/loss`, `dev/macro_f1`, and `test/*` scalars
  - Negative Tests (expected to FAIL):
    - Passing an invalid `--results_dir` path causes an error before training starts
    - Omitting `--id_to_label_path` still works (labels shown as numeric IDs)

- AC-4: Per-fold and aggregated reporting
  - Positive Tests (expected to PASS):
    - Each fold directory contains `metrics.json` with keys: `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`, `per_class`
    - Each fold directory contains `confusion_matrix.csv` with label headers (rows=true, cols=predicted, sklearn convention)
    - Each fold directory contains `predictions.tsv` with columns: `true_label`, `pred_label`, `true_id`, `pred_id`
    - `aggregated_metrics.json` contains `mean`, `std`, and `per_fold` arrays for each metric key
    - TensorBoard logs at `tb_logs/fold_N/` load successfully and contain expected scalar tags
  - Negative Tests (expected to FAIL):
    - `compute_metrics()` with mismatched array lengths raises an error
    - `aggregate_fold_metrics()` with an empty list returns an empty dict, not a crash

- AC-5: End-to-end runner orchestration
  - Positive Tests (expected to PASS):
    - `--dry_run` creates `resolved_config.json` and exits without training
    - Runner resolves checkpoint from zip (extracts `nomoe_bertflow_pre-trained_model.bin-120000`, renames to `pre-trained_model.bin`)
    - Runner skips extraction if checkpoint already exists
    - Runner creates `environment.json` with Python version, torch version, GPU info
    - Runner writes `label_to_id.json` and `id_to_label.json` before training
    - `run.log` contains full stdout/stderr from all fold training runs
    - Runner prints aggregated mean±std for accuracy, macro_f1, weighted_f1
  - Negative Tests (expected to FAIL):
    - `--target invalid_column` is rejected by argparse
    - A config missing required keys raises `ValueError`

- AC-6: All four classification targets produce valid results
  - Positive Tests (expected to PASS):
    - `device_type_label` (10 classes) completes 5-fold CV and produces aggregated metrics
    - `device_label` (49 classes after filtering) completes 5-fold CV
    - `activity_type_label` (11 classes) completes 5-fold CV
    - `activity_label` (54 classes after min_samples=5 filter) completes 5-fold CV
    - All 4 targets use identical hyperparameters (lr=6e-5, batch=128, epochs=4, earlystop=4, seq_length=320)
  - Negative Tests (expected to FAIL):
    - A target where post-extraction filtering drops all classes below threshold raises an error

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation includes: config-driven pipeline, cached 32-worker feature extraction, stratified 5-fold CV with feasibility checks, per-fold + aggregated file reporting, TensorBoard integration, end-to-end runner with dry-run mode, checkpoint resolution from zip, documentation, and successful runs on all 4 label targets.

### Lower Bound (Minimum Acceptable Scope)
The implementation includes: config-driven pipeline, serial feature extraction with caching, stratified 5-fold CV, per-fold file reporting with aggregation, end-to-end runner, and at least one successful target run (activity_type_label).

### Allowed Choices
- Can use: existing `get_feature_flow()` with large `packets_num`, `StratifiedKFold` from scikit-learn, `SummaryWriter` from `torch.utils.tensorboard`, `subprocess` for training orchestration, JSON/CSV/TSV file formats
- Cannot use: external logging services (WandB, MLflow), new deep learning frameworks, modifications to UER core modules, grouped CV splitting

## Feasibility Hints and Suggestions

### Conceptual Approach

```
1. Load config + manifest CSV
2. Extract features from all pcaps (cached, 32 workers)
   - get_feature_flow(pcap, payload_length=64, packets_num=999999, start_index=76)
   - Cache to generated/behaviot/cache/features_cache.json
3. For target column T:
   a. Filter to extractable rows
   b. Build label_to_id (sorted, min_samples filtered)
   c. Validate split feasibility
   d. Build 5-fold StratifiedKFold splits
   e. For each fold:
      - Write train/valid/test TSVs
      - Run run_classifier.py as subprocess
      - Collect metrics.json
   f. Aggregate fold metrics → aggregated_metrics.json
```

### Relevant References
- `data_generation/finetuning_data_gen.py` — `get_feature_flow()` function (line 158), bigram extraction from pcap
- `data_generation/utils.py` — `bigram_generation()` (line 66), `write_dataset_tsv()` (line 130)
- `fine-tuning/run_classifier.py` — `evaluate()` (line 201), `main()` (line 260), argument parsing (line 261)
- `uer/opts.py` — `finetune_opts()` (line 66) defines existing CLI args
- `models/bert/base_config.json` — BERT-base config (768 hidden, 12 heads, 12 layers)
- `models/encryptd_vocab.txt` — vocabulary file (60,005 tokens)

## Dependencies and Sequence

### Milestones

1. **Data Infrastructure**: Config parsing, manifest loading, feature extraction, caching
   - Phase A: Config schema, loader, validator
   - Phase B: Manifest loading, label map construction
   - Phase C: Feature extraction with multiprocessing and caching

2. **Split and Dataset Generation**: K-fold splitting with feasibility checks, TSV materialization
   - Phase A: Stratified k-fold split logic with feasibility validation
   - Phase B: TSV file generation per fold

3. **Training Integration**: run_classifier.py modifications, reporting utilities
   - Phase A: Reporting utilities module (metrics, files, aggregation)
   - Phase B: run_classifier.py modifications (results saving, TensorBoard)

4. **Orchestration**: End-to-end runner, documentation
   - Phase A: Runner script with checkpoint resolution and k-fold loop
   - Phase B: Documentation

5. **Validation**: Smoke test, full experiments
   - Phase A: Smoke experiment (activity_type_label, 2 folds)
   - Phase B: Full experiments (all 4 targets, 5 folds each)

Milestone 2 depends on Milestone 1. Milestone 3 can proceed in parallel with Milestone 2 (reporting utilities are independent). Milestone 4 depends on Milestones 1-3. Milestone 5 depends on Milestone 4.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Project scaffolding: tests/__init__.py, .gitignore, config JSON files, config loader/validator, manifest loader, label map builder | AC-1, AC-5 | coding | - |
| task2 | Feature extraction: `_extract_single_pcap` worker (calls `get_feature_flow` with `packets_num=999999`), `extract_all_features` with 32-worker multiprocessing and JSON caching | AC-1, AC-1.1 | coding | task1 |
| task3 | K-fold splitting: `build_kfold_splits` with StratifiedKFold and feasibility validation, `write_fold_tsvs`, `cap_samples_per_class` | AC-2, AC-2.1 | coding | task1 |
| task4 | Reporting utilities: `compute_metrics`, `write_metrics_json`, `write_confusion_matrix_csv`, `write_predictions_tsv`, `aggregate_fold_metrics` | AC-4 | coding | - |
| task5 | Modify run_classifier.py: add `--results_dir`, `--id_to_label_path`, `--tb_log_dir` args; extend `evaluate()` return; add TensorBoard logging; save results after test eval | AC-3 | coding | task4 |
| task6 | End-to-end runner: `resolve_checkpoint`, `build_classifier_command`, k-fold orchestration loop, aggregation, dry-run mode | AC-5 | coding | task2, task3, task4, task5 |
| task7 | Review plan correctness: verify label counts, split feasibility, config consistency, and reporting conventions | AC-2, AC-4 | analyze | task6 |
| task8 | Documentation: README section for BehavIoT workflow | AC-5 | coding | task6 |
| task9 | Smoke experiment: run activity_type_label with 2 folds, verify all artifacts | AC-4, AC-5 | coding | task8 |
| task10 | Full experiments: run all 4 label targets with 5 folds each, verify aggregated results | AC-6 | coding | task9 |

## Claude-Codex Deliberation

### Agreements
- Post-extraction filtering before label-map creation and fold construction is correct
- Extending `run_classifier.py` with optional output-path args and returning `y_true`/`y_pred` is a low-risk integration
- Per-fold artifacts plus per-target aggregation with TensorBoard logging is appropriate
- Identical hyperparameters across all 4 targets is a defensible baseline
- No class imbalance handling is acceptable for a first pass when macro F1 is the headline metric

### Resolved Disagreements

- **"Multi-label" terminology**: Codex correctly noted these are four separate single-label multi-class classification tasks, not multi-label. Renamed to "multi-target" throughout. Claude agreed.

- **Smoke config feasibility**: Codex identified that `max_samples_per_class=10` with 2 folds makes stratified inner dev splits infeasible for high-class targets (dev_size < num_classes). Resolution: smoke config uses all samples (no cap) with 2 folds and 2 epochs for speed. Smoke test runs only on activity_type_label.

- **Split-feasibility checks**: Codex required explicit validation before fold generation. Resolution: `build_kfold_splits` validates that every class has ≥ `n_folds` extractable samples and the inner dev split is feasible. Fails fast with descriptive error.

- **Label count correction**: Codex verified actual counts differ from draft (10/49/11/55 vs draft's 9/50/11/55). Corrected in plan and design doc.

- **Import safety**: Codex claimed importing `finetuning_data_gen.py` triggers scapy socket init. Claude verified via exploration that module-level imports are static and don't trigger socket operations. The plan uses lazy imports inside `_extract_single_pcap` regardless, making this a non-issue.

- **Single JSON cache**: Codex flagged scalability risk. Claude assessed: with `packets_num=999999` and full-pcap extraction, cache could reach ~200MB. Acceptable for this hardware (2x RTX 5090, 32 cores). Not blocking.

### Convergence Status
- Final Status: `converged`
- Rounds: 2 (first-pass analysis + one challenge/revise round)

## Pending User Decisions

- DEC-1: Extraction cost vs. model consumption
  - Claude Position: Honor user's choice of `packets_num=999999`. Extraction captures all packets but `seq_length=320` effectively limits the model to ~5 packets. The extra extraction is cached and only runs once.
  - Codex Position: This is methodologically weak — full extraction cost with prefix-only consumption. Recommends pre-truncating to the token budget.
  - Tradeoff Summary: Full extraction preserves optionality (can increase seq_length later without re-extracting). But wastes ~10x cache space and extraction time vs. capping at 5 packets.
  - Decision Status: `User chose full extraction (option C) during brainstorming`

- DEC-2: Row-level vs. grouped cross-validation
  - Claude Position: Use row-level StratifiedKFold as user chose during brainstorming.
  - Codex Position: BehavIoT may contain near-repeated sessions within the same device/activity directories. Grouped splitting (StratifiedGroupKFold by device or capture session) would reduce leakage and give more realistic generalization estimates.
  - Tradeoff Summary: Grouped CV is more conservative but harder to implement and usually produces lower scores. Row-level is standard and simpler.
  - Decision Status: `User chose row-level stratification during brainstorming`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Key Implementation Details
- Confusion matrix in reporting_utils uses sklearn convention (rows=true, cols=predicted). The existing `evaluate()` stdout uses reversed convention — this is known and acceptable.
- `_extract_single_pcap` uses lazy import of `get_feature_flow` inside the function body for multiprocessing compatibility (pickling requirement).
- Smoke config: `n_folds=2`, `epochs_num=2`, `earlystop=2`, `batch_size=64`, `max_samples_per_class=null` (no cap).
- Full config: `n_folds=5`, `epochs_num=4`, `earlystop=4`, `batch_size=128`, `max_samples_per_class=null`.

--- Original Design Draft Start ---

(See `/home/shinanliu/TrafficFormer/docs/plans/2026-03-28-behaviot-multi-label-kfold.md` for the complete implementation draft with full code for all 9 tasks.)

--- Original Design Draft End ---
