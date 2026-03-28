# Hierarchical + Device-Conditioned Activity Classification — Experiment Design

**Date:** 2026-03-29
**Status:** Approved

## Goal

Improve activity classification by (1) merging confusable class pairs into a 9-class hierarchical label set, and (2) conditioning on predicted device type via feature concatenation. Two experiments isolate each contribution.

## Analysis Motivating This Design

### Error Budget (A4-v2, 11-class, averaged across 5 folds)

| Error Type | Misclassifications/fold | % of Errors | Root Cause |
|---|---|---|---|
| off ↔ on | ~158 | 33% | Encrypted power commands indistinguishable |
| brightness ↔ color | ~93 | 19% | Both are lighting parameter changes |
| Other | ~229 | 48% | Various |

52% of all errors come from 2 confusion pairs that are irreducible at the packet header level.

### Baseline Results

| Config | 11-class Accuracy | 11-class Macro F1 |
|---|---|---|
| Original baseline | 53.7% | 0.549 |
| A4-v2 (seq1024 + weights) | 60.3% | 0.601 |
| A4-v2 remapped to 9 classes | 86.1% | ~0.73 |

## Experiment Design

### Exp1: 9-Class Hierarchical (merge only)

**Hypothesis:** Merging confusable pairs eliminates 52% of errors, raising macro F1 from ~0.60 to ~0.73.

**Label remapping:**
- {on, off} → power_toggle
- {brightness, color} → lighting_adjust
- audio, capture, idle, routine, setpoint, volume, watch → unchanged

**Config:** seq_length=1024, inverse-sqrt class weights, MAC-filtered extraction, batch_size=16, dual GPU.

**Independent variable:** Label set (11 → 9 classes)
**Control:** Same splits, seed, checkpoint, extraction as A4-v2

### Exp2: 9-Class + Device Feature Concatenation

**Hypothesis:** Device conditioning improves classes with device-exclusive activities (capture→camera, setpoint→thermostat) by ~5-10% macro F1 over Exp1.

**Pipeline:**
1. Pre-step: Run existing device_type_label classifier (98% accurate) on all extractable samples. Save per-sample predictions to JSON.
2. During training/eval: Look up predicted device type for each sample, feed to `nn.Embedding(10, 64)`.
3. Concatenate device embedding (64-dim) with [CLS] token (768-dim) → 832-dim input to classification head.

**Architecture change in Classifier:**
```
[CLS] output (768) ──┐
                      ├── concat (832) → Linear(832, 832) → tanh → Linear(832, 9) → logits
device_type pred ──→ Embedding(10, 64) ──┘
```

**Independent variable:** Device conditioning (off vs on)
**Control:** Same 9-class label set, same everything else as Exp1

## Expected Outcomes

| Experiment | Accuracy | Macro F1 | Confidence |
|---|---|---|---|
| Exp1 (merge only) | ~86% | ~0.73 | High — directly computed from confusion matrix |
| Exp2 (merge + device) | ~85-88% | ~0.78-0.82 | Medium — depends on device-activity correlation |

### Per-class expected improvements (Exp2 vs Exp1)

| Class | Exp1 recall | Exp2 est. | Why |
|---|---|---|---|
| capture | 85% | ~94% | Camera-only activity |
| watch | 91% | ~96% | Display-only activity |
| setpoint | 76% | ~92% | Thermostat-only activity |
| volume | 78% | ~88% | Speaker activity |
| audio | 68% | ~85% | Speaker activity |
| lighting_adjust | 96% | ~90% | Bulb-only, but merged class already high |
| power_toggle | 89% | ~88% | All devices — device type doesn't help much |
| idle | 66% | ~72% | All devices — moderate help |
| routine | 23% | ~30% | Still hard — heterogeneous, few samples |

## Validation Pyramid

| Level | Enabled | Config |
|---|---|---|
| L0: Static Checks | Yes | Same as ablations |
| L1: Runtime Validation | Yes | Real data, 5 min |
| L2: E2E Pipeline | Yes | 3-5 steps/stage |

## Implementation Scope

### Code Changes

1. **Label remapping** — function in `behaviot_data_gen.py` to map 11-class labels to 9-class
2. **Device prediction generator** — script that runs existing device_type model on all samples, saves predictions JSON
3. **Modified Classifier** — `DeviceConditionedClassifier` in `run_classifier.py` with `nn.Embedding(10, 64)` + concatenation
4. **New args** — `--device_predictions_path`, `--use_device_embedding` in `run_classifier.py`
5. **Configs** — `configs/behaviot/hier_exp1.json`, `configs/behaviot/hier_exp2.json`
6. **Runner updates** — support hierarchical label remapping and device predictions

### Files Affected

| File | Change |
|---|---|
| `data_generation/behaviot_data_gen.py` | `remap_to_hierarchical()` function |
| `fine-tuning/run_classifier.py` | `DeviceConditionedClassifier`, `--device_predictions_path` |
| `scripts/run_behaviot.py` | Hierarchical label support, device prediction passing |
| `scripts/generate_device_predictions.py` | New: run device classifier, save predictions |
| `configs/behaviot/hier_exp1.json` | 9-class, seq=1024, weights, no device |
| `configs/behaviot/hier_exp2.json` | 9-class, seq=1024, weights, with device |

## Hardware

- 2x NVIDIA RTX 5090 (32GB each), dual-GPU DataParallel
- Estimated runtime: ~2 hours per experiment × 2 = ~4 hours total
