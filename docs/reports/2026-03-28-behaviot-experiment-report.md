# BehavIoT Traffic Classification with TrafficFormer: Experiment Report

**Date:** March 28, 2026
**Author:** Shinan Liu (with Claude Code assistance)
**Repository:** [shinan6/TrafficFormer](https://github.com/shinan6/TrafficFormer)

---

## 1. Executive Summary

We adapted TrafficFormer — a BERT-based encrypted traffic classifier that operates on bigram-tokenized packet headers — to the BehavIoT IoT device dataset. We evaluated the model on four classification targets using 5-fold stratified cross-validation with pretrained initialization. The key finding is a stark performance divide: **device identification achieves 98% accuracy (macro F1 > 0.90), while activity recognition reaches only 38–54% accuracy (macro F1 0.30–0.55)**. This suggests that encrypted packet headers carry strong device fingerprints but weak activity signatures.

---

## 2. Background and Motivation

TrafficFormer pre-trains a BERT-base transformer (768-dim, 12 heads, 12 layers) on encrypted network traffic represented as bigram tokens — consecutive pairs of hexadecimal characters from packet headers. The model learns protocol-level patterns from byte sequences while anonymizing sensitive fields (IPs, ports, timestamps). Previous work demonstrated strong performance on encrypted application classification. This study extends TrafficFormer to IoT behavioral analysis using the BehavIoT dataset.

**Research question:** Can packet header bigrams, pre-trained on general encrypted traffic, discriminate IoT device identity and behavioral activity from short capture sessions?

---

## 3. Dataset

**Source:** BehavIoT dataset (`/mnt/data/behavoiot/pcap_vs_label.csv`)
**Total samples:** 7,514 pcap files (short capture sessions, ~38 seconds each)

Each pcap file is labeled across four hierarchical classification targets:

| Target | Granularity | Classes | Sample Range | Description |
|--------|-------------|---------|--------------|-------------|
| `device_type_label` | Coarse device | 10 | 39 – 1,966 | Device category (bulb, camera, plug, ...) |
| `device_label` | Fine device | 36 | 6 – 603 | Specific device model (amazon-plug, tplink-bulb, ...) |
| `activity_type_label` | Coarse activity | 11 | 90 – 1,530 | Activity category (on, off, audio, brightness, ...) |
| `activity_label` | Fine activity | 52 | 6 – 548 | Specific activity (android_lan_on, alexa_audio, ...) |

**Class imbalance:** Significant across all targets. For `device_type_label`, the `appliance` class has only 39 samples vs. 1,966 for `bulb` (50:1 ratio). For `activity_type_label`, `routine` has 90 samples vs. 1,530 for `off` (17:1 ratio). No class balancing techniques were applied in this baseline study.

### 3.1 Data Attrition

Of the 7,514 manifest entries, **5,799 (77.2%)** were successfully extracted. The remaining 1,715 (22.8%) were skipped during feature extraction:

| Skip Reason | Count | % of Skipped |
|-------------|-------|--------------|
| Scapy/flowcontainer field parsing errors | 1,032 | 60.2% |
| get_feature_flow returned -1 (IPv6, <3 packets, no TCP/UDP) | 565 | 32.9% |
| Arithmetic errors in flow metadata extraction | 110 | 6.4% |
| Fewer than 3 IP packets after EAPOL filtering | 6 | 0.4% |
| Other | 2 | 0.1% |

**Note:** The 22.8% skip rate is a concern. The scapy field parsing errors (60% of skips) appear related to malformed TCP options in certain pcap files and may warrant investigation. The skipped samples could introduce systematic bias if certain device types or activities are disproportionately affected.

---

## 4. Method

### 4.1 Feature Extraction

Each pcap file is processed as follows:

1. **EAPOL filtering:** BehavIoT pcaps begin with 802.1x authentication frames. Non-IP packets are stripped before extraction.
2. **Flow extraction:** `flowcontainer.extract()` identifies TCP/UDP flows via tshark.
3. **Anonymization:** IP addresses, ports, TCP timestamps, and TLS random times are randomized to remove identifying metadata while preserving protocol structure.
4. **Bigram tokenization:** Packet bytes (starting at offset 76, covering transport/application headers) are converted to 2-character hexadecimal bigrams. All packets from all flows in the pcap are concatenated with `[SEP]` tokens.
5. **No packet limit:** All packets are extracted (`packets_num=999999`). Truncation to 320 tokens occurs at tokenization time within the classifier.

**Effective model input:** With `seq_length=320` and ~63 bigram tokens per 64-byte packet slice, the model sees approximately **5 packets** per sample regardless of pcap size. The remaining packets in longer captures are discarded by the tokenizer.

Feature extraction runs with 32-worker multiprocessing and results are cached to a 2.6GB JSON file, keyed by manifest content and extraction parameters.

### 4.2 Cross-Validation Protocol

- **Outer split:** 5-fold `StratifiedKFold` (shuffle=True, seed=42)
- **Inner split:** 90% train / 10% dev within each fold's training portion
  - Stratified when feasible; falls back to random split for targets with many rare classes (device_label, activity_label)
- **Early stopping:** Training halts after 4 consecutive epochs without dev macro F1 improvement
- **Feasibility check:** Classes with fewer than 5 extractable samples are dropped before splitting

### 4.3 Model and Training

| Parameter | Value |
|-----------|-------|
| Architecture | BERT-base (768-dim, 12 heads, 12 layers, word_pos_seg embedding) |
| Pretrained checkpoint | `nomoe_bertflow_pre-trained_model.bin-120000` (715MB, bertflow pretraining) |
| Sequence length | 320 tokens |
| Batch size | 32 |
| Learning rate | 6e-5 (AdamW, linear schedule with 10% warmup) |
| Epochs | 4 (with earlystop=4) |
| Pooling | First token ([CLS]) |
| Loss | NLLLoss on LogSoftmax logits |
| GPU | NVIDIA RTX 5090 (32GB), single-GPU training |

All four classification targets use identical hyperparameters for fair comparison. The only difference is the output layer size (matching the number of classes per target).

---

## 5. Results

### 5.1 Summary

| Target | Classes | Accuracy | Macro F1 | Weighted F1 |
|--------|---------|----------|----------|-------------|
| `device_type_label` | 10 | **0.9812 ± 0.0029** | **0.9131 ± 0.0361** | **0.9806 ± 0.0035** |
| `device_label` | 36 | **0.9800 ± 0.0031** | **0.9049 ± 0.0245** | **0.9789 ± 0.0031** |
| `activity_type_label` | 11 | 0.5365 ± 0.0026 | 0.5493 ± 0.0088 | 0.5327 ± 0.0048 |
| `activity_label` | 52 | 0.3793 ± 0.0238 | 0.3007 ± 0.0261 | 0.3566 ± 0.0227 |

All metrics are reported as mean ± standard deviation across 5 folds.

### 5.2 Per-Fold Stability

**Device targets** show remarkable stability across folds (accuracy std < 0.003), indicating the learned device fingerprints generalize consistently.

| Fold | device_type F1 | device F1 | activity_type F1 | activity F1 |
|------|----------------|-----------|-------------------|-------------|
| 0 | 0.9447 | 0.9100 | 0.5421 | 0.3256 |
| 1 | 0.9511 | 0.8761 | 0.5493 | 0.2621 |
| 2 | 0.8659 | 0.8986 | 0.5530 | 0.2953 |
| 3 | 0.9304 | 0.9487 | 0.5387 | 0.2868 |
| 4 | 0.8735 | 0.8914 | 0.5636 | 0.3335 |

**Activity targets** are also stable (F1 std < 0.03) but at a much lower performance level, suggesting the model consistently fails to learn activity-discriminative features rather than suffering from fold-dependent variance.

### 5.3 Confusion Analysis (activity_type_label)

The confusion matrices across 5 folds reveal systematic misclassification patterns:

**Major confusions (averaged across folds):**

| True → Predicted | Avg Misclassifications | Interpretation |
|-----------------|----------------------|----------------|
| brightness → color | ~67/fold | Both are lighting parameter changes |
| color → brightness | ~55/fold | Same lighting control interface |
| off → on | ~72/fold | Power state toggles use similar protocols |
| on → off | ~91/fold | Most confused pair overall |
| audio → capture | ~20/fold | Both involve media streaming |
| capture → audio | ~15/fold | Audio/video use similar codecs |

**Well-separated classes:**
- **watch** (video streaming): ~79/101 correct — distinctive long-duration traffic pattern
- **setpoint** (thermostat): ~21/27 correct — unique device/protocol signature
- **idle**: ~22/37 correct — low traffic volume is distinctive

**Poorly-separated classes:**
- **routine** (automation sequences): ~2/12 correct — too few samples and heterogeneous traffic patterns
- **on/off**: Highly confused with each other and with color/brightness — the network traffic for toggling a smart bulb on vs. off is nearly identical at the packet header level

### 5.4 Device vs. Activity Performance Gap

The ~45 percentage point gap between device accuracy (98%) and activity accuracy (54%) reveals a fundamental characteristic of encrypted IoT traffic:

1. **Device fingerprints are in headers:** Different IoT devices use distinct TLS implementations, packet sizing patterns, and communication protocols that are visible in the first 5 packets of encrypted headers. These fingerprints persist across activities.

2. **Activity signals are in payload timing:** The difference between "turning a light on" vs. "changing its color" lies in the application-layer commands, which are encrypted. The packet header bigrams capture protocol structure (e.g., TLS record types, packet sizes) but not semantic command content.

3. **seq_length=320 limitation:** The model sees only ~5 packets. Activity patterns may require longer temporal context (10-50 packets) to distinguish power toggles from brightness changes.

---

## 6. Limitations and Threats to Validity

### 6.1 Feature Extraction Limitations

- **22.8% sample attrition:** 1,715 of 7,514 pcaps failed extraction. If failures correlate with device type or activity, the effective dataset is biased. The largest failure mode (scapy field parsing errors, 1,032 samples) deserves investigation.

- **5-packet effective window:** Despite extracting all packets (`packets_num=999999`), `seq_length=320` truncates input to ~5 packets. Longer captures contribute no additional information. This design choice was intentional (preserving full extraction for future seq_length experiments) but means current results reflect a narrow temporal window.

- **All-flows mixing:** Each pcap's packets are concatenated across all TCP/UDP flows without flow boundary markers. The model cannot distinguish control-plane traffic (e.g., DNS, NTP) from device-specific application flows. A flow-aware representation might improve activity classification.

### 6.2 Experimental Design Limitations

- **No class balancing:** Severe class imbalance (up to 50:1) is unaddressed. Macro F1 partially compensates by weighting all classes equally, but rare classes like `routine` (90 samples) and `appliance` (39 samples) have limited representation in each fold.

- **Row-level splitting:** Pcap files from the same device and time period may share network conditions. Row-level `StratifiedKFold` does not account for this temporal/device clustering, potentially overstating generalization performance. `StratifiedGroupKFold` by device would give more conservative estimates.

- **Fixed hyperparameters:** All four targets use identical learning rate, batch size, and training epochs. The 52-class `activity_label` task may benefit from different hyperparameters (e.g., lower learning rate, more epochs) compared to the 10-class `device_type_label` task.

- **Non-stratified inner dev split:** For targets with many rare classes (device_label, activity_label), the inner 90/10 train/dev split falls back to random (non-stratified) splitting. This means some classes may be absent from the dev set, making early stopping less reliable for those classes.

### 6.3 Reproducibility Notes

- **flowcontainer dependency:** The `flowcontainer` package enforces a tshark version ceiling (4.0.0) that required patching for tshark 4.2.2. This patch must be reapplied after reinstalling flowcontainer.
- **Scapy version:** Upgraded from 2.5.0 to 2.6.1 for cryptography compatibility. Original `requirements.txt` specifies 2.5.0.
- **Feature cache:** The 2.6GB feature cache is not committed to git. Regeneration requires ~10 minutes with 32 workers.

---

## 7. Recommendations and Next Steps

### 7.1 High Priority

1. **Investigate the 22.8% extraction failure rate.** The 1,032 scapy field parsing errors may be fixable with error-tolerant packet parsing. Recovering even half of these would add ~500 samples and could improve rare-class performance.

2. **Increase seq_length for activity classification.** The current 5-packet window is likely insufficient for activity discrimination. Experiment with `seq_length=640` (10 packets) and `seq_length=1280` (20 packets). The feature cache already stores full extractions, so only retraining is needed.

3. **Add class-weighted loss.** For `activity_type_label` and `activity_label`, pass class-weight inversely proportional to class frequency to `NLLLoss()`. This is a one-line change in `run_classifier.py` and could significantly improve macro F1 for rare classes.

### 7.2 Medium Priority

4. **Flow-aware feature representation.** Instead of concatenating all flows, identify the primary device flow (largest by packet count or bytes) and use it exclusively. Background DNS/NTP traffic adds noise without discriminative signal.

5. **Grouped cross-validation.** Replace `StratifiedKFold` with `StratifiedGroupKFold` grouped by device ID or capture session to get more realistic generalization estimates.

6. **Data augmentation for rare classes.** The existing `enhance_based_tsv()` function randomizes IPs/ports/sequences while preserving protocol structure. Apply it to minority classes (`routine`, `appliance`, `sensor`) to balance the training set.

### 7.3 Lower Priority

7. **Multi-task learning.** Train a single model with four output heads (one per target) sharing the BERT encoder. This could improve activity classification by leveraging device identity as an implicit feature.

8. **Hyperparameter search per target.** The 52-class activity_label task likely needs different training dynamics than the 10-class device_type_label task. A simple grid search over learning rate and epochs per target could help.

9. **Pretrain on BehavIoT traffic.** The current checkpoint was pretrained on general encrypted traffic. Domain-adaptive pretraining on BehavIoT pcaps (without labels, using the bertflow objective) could improve all downstream tasks.

---

## 8. Technical Implementation Summary

### 8.1 Code Changes

| Component | Files | Purpose |
|-----------|-------|---------|
| Data generation | `data_generation/behaviot_data_gen.py` | Config, manifest, extraction, k-fold, TSV |
| Reporting | `uer/reporting_utils.py` | Metrics, confusion matrix, predictions, aggregation |
| Runner | `scripts/run_behaviot.py` | End-to-end orchestration |
| Classifier mods | `fine-tuning/run_classifier.py` | Result saving, TensorBoard, labels_num_override |
| Tests | `tests/test_*.py` (3 files) | 39 unit tests |
| Configs | `configs/behaviot/{smoke,full}.json` | Experiment configurations |

### 8.2 Pipeline Architecture

```
manifest CSV → feature extraction (32 workers, cached) → label map + feasibility check
    → StratifiedKFold (5 folds) → per-fold TSV generation (truncated to seq_length)
    → run_classifier.py (subprocess, TensorBoard logging)
    → per-fold metrics.json + confusion_matrix.csv + predictions.tsv
    → aggregated_metrics.json (mean ± std)
```

### 8.3 Quality Assurance

- **13 commits** with incremental fixes discovered during development and Codex code review
- **39 unit tests** covering config loading, manifest parsing, feature extraction caching, k-fold splitting, TSV generation, reporting utilities, checkpoint resolution, and backward compatibility
- **6 rounds of automated code review** identified and resolved: stale cache detection, partial fold aggregation, missing-class reporting crashes, temp file leaks, label count mismatches, and eval-only mode regressions

---

## Appendix A: Full Per-Class Metrics (activity_type_label, Fold 4)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| audio | 0.547 | 0.678 | 0.606 | 90 |
| brightness | 0.573 | 0.357 | 0.440 | 154 |
| capture | 0.775 | 0.679 | 0.724 | 81 |
| color | 0.352 | 0.579 | 0.438 | 145 |
| idle | 0.639 | 0.605 | 0.622 | 38 |
| off | 0.491 | 0.531 | 0.510 | 258 |
| on | 0.528 | 0.420 | 0.468 | 219 |
| routine | 0.500 | 0.250 | 0.333 | 12 |
| setpoint | 0.889 | 0.593 | 0.711 | 27 |
| volume | 0.621 | 0.529 | 0.571 | 34 |
| watch | 0.847 | 0.762 | 0.803 | 101 |

**Best-performing classes:** watch (F1=0.80), capture (F1=0.72), setpoint (F1=0.71) — these have distinct traffic patterns.
**Worst-performing classes:** routine (F1=0.33), color (F1=0.44), brightness (F1=0.44) — confused with similar activities.
