# BehavIoT Activity Classification Improvement Plan

## Goal Description

Improve TrafficFormer's activity classification on BehavIoT from macro F1 0.55 (activity_type_label) and 0.30 (activity_label) through three interventions: (1) device-MAC filtered extraction replacing all-flows mixing, (2) longer sequence lengths (512, 1024), and (3) inverse-sqrt class-weighted NLLLoss. Run 5 experiments (B0 baseline + A1-A4 ablations) on both activity targets with 5-fold CV, then combine the best seq_length with class weights.

## Acceptance Criteria

- AC-1: Device-MAC filtered extraction
  - Positive Tests:
    - All pcaps with a matching device MAC and at least 1 IP packet produce a non-empty datagram
    - Pcaps previously skipped by flowcontainer (565 samples) are now extractable
    - Extraction cache key includes extraction mode; changing strategy invalidates cache
  - Negative Tests:
    - Pcaps with no matching MAC packets produce a skip entry (not a crash)
    - Non-IP packets (ARP, EAPOL) are excluded from the datagram

- AC-2: B0 baseline on MAC-filtered data
  - Positive Tests:
    - B0 runs 5-fold CV on both activity_type_label and activity_label
    - B0 results include aggregated_metrics.json with macro F1 mean±std
    - Post-extraction class counts are logged per target
  - Negative Tests:
    - B0 does NOT reuse old all-flows cache (cache key mismatch forces re-extraction)

- AC-3: Sequence length ablations (A1, A2)
  - Positive Tests:
    - A1 trains with seq_length=512 and model input is 512 tokens
    - A2 trains with seq_length=1024, max_seq_length=1024, and model input is 1024 tokens
    - A2 position embeddings 0-511 are copied from pretrained checkpoint; 512-1023 are randomly initialized
    - TSV write truncation matches the configured seq_length
  - Negative Tests:
    - A2 does NOT crash on load_state_dict due to position embedding shape mismatch
    - A2 batch_size is reduced to fit in GPU memory (16 if 32 causes OOM)

- AC-4: Class-weighted NLLLoss (A3)
  - Positive Tests:
    - Class weights are computed from train fold class counts only (not dev/test)
    - Weights use inverse-sqrt formula: weight_c = sqrt(N_total / (N_classes * N_c))
    - Class weights JSON is saved per fold for auditability
    - run_classifier.py accepts --class_weights_path and applies weights to NLLLoss
  - Negative Tests:
    - Omitting --class_weights_path produces unweighted NLLLoss (backward compatible)

- AC-5: Combined experiment (A4)
  - Positive Tests:
    - A4 uses the best seq_length from {A1, A2} per target + class weights
    - A4 results are compared against B0 (not legacy baseline)
  - Negative Tests:
    - A4 is NOT skipped even if only one of seq_length/class_weights helps individually

- AC-6: Experiment reporting
  - Positive Tests:
    - All 5 experiments produce aggregated_metrics.json for both targets
    - Final comparison table shows all experiments vs B0
    - TensorBoard logs exist for all folds
  - Negative Tests:
    - Partial fold failures cause exit code 1 (not silent success)

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation includes: MAC-filtered extraction with full re-extraction and caching, position embedding expansion for 1024 tokens, class-weighted loss with per-fold weight computation, 5 experiments × 2 targets × 5 folds = 50 training runs, comparison table with delta vs B0, TensorBoard logging.

### Lower Bound (Minimum Acceptable Scope)
The implementation includes: MAC-filtered extraction, B0 baseline, A1 (seq=512), A3 (class weights), and A4 (combined). A2 (seq=1024) may be skipped if position embedding expansion proves too complex.

### Allowed Choices
- Can use: existing run_behaviot.py pipeline, scapy for packet filtering, bigram_generation from utils.py, subprocess training, JSON configs
- Cannot use: external libraries beyond existing requirements, modifications to UER core framework, grouped cross-validation (row-level CV only)

## Feasibility Hints and Suggestions

### Conceptual Approach
```
1. Implement _extract_by_device_mac (filter pcap by device MAC, IP only, anonymize, bigrams)
2. Update extract_all_features to accept device_macs dict
3. Add compute_class_weights() for inverse-sqrt weighting
4. Modify Classifier to accept class weights in NLLLoss
5. Handle A2 position embedding expansion in model loading
6. Create ablation configs (B0, A1, A2, A3, A4)
7. Run B0 → A1 → A2 → A3 → determine best seq → A4
8. Compile comparison table
```

### Relevant References
- `data_generation/behaviot_data_gen.py` — extraction, splitting, TSV generation
- `fine-tuning/run_classifier.py` — Classifier class, NLLLoss, load_or_initialize_parameters
- `uer/layers/embeddings.py` — WordPosSegEmbedding, position embedding table
- `scripts/run_behaviot.py` — runner, build_classifier_command
- `/mnt/data/behavoiot/device_behaviot.txt` — 50 device MAC addresses
- `models/bert/base_config.json` — max_seq_length=512

## Dependencies and Sequence

### Milestones

1. **Data Infrastructure**: MAC extraction + cache invalidation
   - Phase A: load_device_macs, _extract_by_device_mac
   - Phase B: Update extract_all_features, cache key

2. **Training Modifications**: Class weights + position embedding expansion
   - Phase A: compute_class_weights
   - Phase B: Modify run_classifier.py for --class_weights_path
   - Phase C: Position embedding handling for seq=1024

3. **Configs and Runner**: Ablation configs, runner updates
   - Phase A: Create B0, A1, A2, A3 configs + bert config for 1024
   - Phase B: Update runner for device_macs_path, class_weight_method

4. **Experiments**: Run B0, A1-A4, compile results
   - Phase A: Run B0 baseline
   - Phase B: Run A1, A2, A3
   - Phase C: Determine best seq, run A4
   - Phase D: Comparison table

Milestone 2 can proceed in parallel with Milestone 1. Milestone 3 depends on 1+2. Milestone 4 depends on 3.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Implement load_device_macs and _extract_by_device_mac with tests | AC-1 | coding | - |
| task2 | Update extract_all_features for device_macs, update cache key | AC-1, AC-2 | coding | task1 |
| task3 | Implement compute_class_weights with tests | AC-4 | coding | - |
| task4 | Modify run_classifier.py: --class_weights_path, weighted NLLLoss | AC-4 | coding | task3 |
| task5 | Handle position embedding expansion for seq=1024 | AC-3 | coding | - |
| task6 | Create ablation configs (B0, A1, A2, A3) and update runner | AC-2, AC-3, AC-4, AC-5 | coding | task2, task4, task5 |
| task7 | Review extraction, class weights, and position expansion for correctness | AC-1, AC-3, AC-4 | analyze | task6 |
| task8 | Run B0 baseline on both activity targets | AC-2, AC-6 | coding | task6 |
| task9 | Run A1, A2, A3 ablations on both activity targets | AC-3, AC-4, AC-6 | coding | task8 |
| task10 | Determine best seq, create A4 config, run A4 | AC-5, AC-6 | coding | task9 |
| task11 | Compile comparison table, record conclusions | AC-6 | coding | task10 |

## Claude-Codex Deliberation

### Agreements
- MAC filtering targets the right bottleneck (noisy multi-flow input)
- B0 as new baseline is correct since sample universe changes
- Recovering skipped pcaps is valuable (~565 additional samples)
- Class weights from train fold only is the defensible choice
- Per-fold weight saving is good for auditability

### Resolved Disagreements
- **MAC filter ≠ single flow**: Codex noted MAC filtering yields multiple L4 flows, not a single flow. Resolution: keep all device-MAC packets (multiple flows fine — the user means "single device" not "single TCP connection").
- **AC-4 definition**: Codex required AC-4 at model-input level, not TSV level. Resolution: AC-3 now specifies "model input is N tokens" rather than "TSV has N tokens."
- **A2 batch size**: Codex identified potential OOM at seq=1024 with batch=32. Resolution: A2 config uses batch_size=16.
- **A4 per-target ambiguity**: Codex noted A1 vs A2 winner could differ per target. Resolution: A4 uses best seq_length per target independently.

### Convergence Status
- Final Status: `converged`
- Rounds: 2 (first-pass analysis + one challenge/revise round)

## Pending User Decisions

No pending decisions — all questions resolved during brainstorming session.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase

### Key Implementation Details
- MAC addresses in device_behaviot.txt must be normalized to lowercase before comparison
- Position embedding expansion: copy pretrained weights[:512], random init [512:1024], log the operation
- A2 config: batch_size=16 (reduced from 32) to fit seq_length=1024 in 32GB VRAM
- Class weight formula: weight_c = sqrt(N_total / (N_classes * N_c)) where counts are from train fold only
- TSV truncation: write_fold_tsvs max_tokens parameter should match seq_length for each experiment
- Feature cache: separate cache directory for MAC-filtered extraction (generated/behaviot_ablation/cache)

--- Original Design Draft Start ---

(See docs/plans/2026-03-28-activity-improvement-plan.md for the full implementation draft with code for all subtasks.)

--- Original Design Draft End ---
