# BehavIoT Activity Classification Improvement — Experiment Design

**Date:** 2026-03-28
**Status:** Approved

## Goal

Improve TrafficFormer's activity classification performance on BehavIoT from the current baseline (activity_type_label macro F1 = 0.55, activity_label macro F1 = 0.30) through controlled ablations targeting temporal context, class imbalance, and data quality.

## Baseline Results (to beat)

| Target | Accuracy | Macro F1 |
|--------|----------|----------|
| activity_type_label (11 classes) | 0.5365 | 0.5493 |
| activity_label (52 classes) | 0.3793 | 0.3007 |

## Pre-Step: Device-MAC Single-Flow Extraction

### Change
Replace "mix all flows" extraction with device-MAC-filtered single-flow extraction:
- Parse `device_behaviot.txt` for per-device MAC addresses (50 devices mapped)
- For each pcap, filter to only packets where Ether.src or Ether.dst matches the device MAC
- No minimum packet count — short samples are padded to seq_length by the tokenizer
- Re-extract all 7,514 pcaps, invalidate and regenerate cache
- This replaces the current `_extract_single_pcap` which mixes all flows

### Rationale
The current extraction mixes all flows (device traffic + phone control traffic + DNS/NTP). Background traffic adds noise without activity-discriminative signal. Filtering to the device's own flows gives a cleaner input for activity classification.

### Expected Impact
- Recovers ~565 previously skipped pcaps (those rejected by per-flow 3-packet check)
- Cleaner features: only device-relevant traffic, no background noise
- Total dataset: ~7,000-7,500 samples (up from 5,799)

## Experiment Design

### Hypothesis
Activity classification is limited by three factors: (1) insufficient temporal context (5 packets), (2) class imbalance (17:1 ratio), and (3) noisy multi-flow input. Addressing these independently and jointly should improve macro F1.

### Independent Variables
1. `seq_length`: {320, 512, 1024}
2. `class_weights`: {none, inverse_sqrt_frequency}

### Dependent Variables
- Macro F1 (primary), accuracy, weighted F1 — mean ± std across 5 folds
- Per-class F1 for confusion analysis

### Control Variables
- Pretrained checkpoint: nomoe_bertflow_pre-trained_model.bin-120000
- Learning rate: 6e-5 (AdamW, linear schedule, 10% warmup)
- Batch size: 32
- Epochs: 4, earlystop: 4
- Seed: 42
- Feature extraction: device-MAC filtered (same for all experiments)
- 5-fold StratifiedKFold (same splits across experiments)

### Experiments

| ID | seq_length | max_seq_length | Class Weights | Purpose |
|----|------------|----------------|---------------|---------|
| B0-baseline | 320 | 512 | None | New baseline on MAC-filtered data |
| A1-seq512 | 512 | 512 | None | Test more temporal context (~8 packets) |
| A2-seq1024 | 1024 | 1024 | None | Test extended context (~16 packets), positions 513-1024 randomly initialized |
| A3-classweight | 320 | 512 | Inverse sqrt freq | Test class balancing at baseline context |
| A4-combined | Best of {512, 1024} | Matching | Inverse sqrt freq | Combined best context + class balancing |

### Class Weight Formula (Inverse Square Root Frequency)
```
weight_c = sqrt(N_total / (N_classes * N_c))
```
Where N_c is the sample count for class c. This gives `routine` (90 samples) ~4x the weight of `off` (1,530 samples), vs 17x with full inverse.

### Evaluation
- Both `activity_type_label` and `activity_label` per experiment
- 5-fold stratified CV, same seed across all experiments
- TensorBoard logging: train/loss, dev/macro_f1, test/* per fold
- File artifacts: metrics.json, confusion_matrix.csv, predictions.tsv per fold
- Aggregated: mean ± std across folds

### Padding
Samples with fewer tokens than seq_length are zero-padded by `read_dataset()` in run_classifier.py (existing behavior, lines 166-168). No code change needed for padding.

### Position Embeddings (A2-seq1024)
The BERT config has `max_seq_length=512`. For seq_length=1024, increase `max_seq_length` to 1024. Positions 0-511 are initialized from the pretrained checkpoint; positions 512-1023 are randomly initialized and learned during fine-tuning via `load_state_dict(strict=False)`.

## Validation Pyramid

| Level | Enabled | Config |
|-------|---------|--------|
| L0: Static Checks | Yes | Same as baseline (device consistency, optimizer, TensorBoard) |
| L1: Runtime Validation | Yes | Real data, smoke config, 5-minute budget |
| L2: E2E Pipeline | Yes | 3-5 steps per stage |

## Implementation Scope

### Code Changes Required
1. **New extraction function:** `_extract_single_flow_by_mac()` — reads device MAC from metadata, filters pcap packets, converts to bigrams
2. **Device MAC loader:** Parse `device_behaviot.txt` into a dict
3. **Class weight computation:** `compute_class_weights()` in behaviot_data_gen.py
4. **run_classifier.py:** Accept `--class_weights_path` argument, load weights into NLLLoss
5. **Runner:** New configs for each ablation, pass class weights and seq_length

### Configs Needed
- `configs/behaviot/ablation_b0.json` (seq=320, no weights)
- `configs/behaviot/ablation_a1.json` (seq=512, no weights)
- `configs/behaviot/ablation_a2.json` (seq=1024, max_seq=1024, no weights)
- `configs/behaviot/ablation_a3.json` (seq=320, class weights)
- `configs/behaviot/ablation_a4.json` (seq=best, class weights)

## Expected Outcomes

### Optimistic
- seq_length increase recovers 5-10% macro F1 (activity patterns visible in 8-16 packets)
- Class weights recover 3-5% macro F1 (rare classes like routine, setpoint improve)
- Combined: activity_type_label macro F1 reaches ~0.65-0.70

### Realistic
- seq_length helps moderately (2-5% macro F1) — some confusion pairs (on/off) may remain hard regardless of context
- Class weights help rare classes but hurt majority class accuracy slightly
- Combined: activity_type_label macro F1 reaches ~0.60-0.65

### Pessimistic
- Encrypted payload content is the fundamental bottleneck
- More context doesn't help because activity signals are in encrypted commands, not packet headers
- Combined improvement < 3% macro F1

## Hardware
- 1x NVIDIA RTX 5090 (32GB)
- 32 CPU cores
- Estimated runtime: ~1 hour per experiment × 5 experiments = ~5 hours total
