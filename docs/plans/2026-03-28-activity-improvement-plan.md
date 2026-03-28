# BehavIoT Activity Classification Improvement — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** Improve activity classification macro F1 from 0.55 (activity_type_label) and 0.30 (activity_label) through device-MAC single-flow extraction, seq_length ablation, and class-weighted loss.

**Hypothesis:** Activity classification is limited by (1) noisy multi-flow input, (2) insufficient temporal context, and (3) class imbalance. Addressing these will improve macro F1.

**Validation scope:** L0 static checks (same as baseline), L1 runtime validation (real data, 5 min), L2 E2E pipeline (3-5 steps/stage).

**Evaluation design:** Per-epoch dev macro F1 for early stopping, full test evaluation after training. Existing infra handles both modes via run_classifier.py. TensorBoard logging enabled.

**Architecture:** Replace `_extract_single_pcap` (all-flows mixing) with `_extract_by_device_mac` (device-MAC filtered). Add `--class_weights_path` to run_classifier.py for inverse-sqrt weighted NLLLoss. Run 5 experiments (B0 baseline + A1-A4 ablations) on both activity targets with 5-fold CV.

---

## Shared Scaffold

### Existing infra (don't touch, advise if problems found)
- Training loop: `fine-tuning/run_classifier.py` (Classifier class, evaluate, train_model, main)
- Data splitting: `data_generation/behaviot_data_gen.py` (build_kfold_splits, write_fold_tsvs)
- Reporting: `uer/reporting_utils.py` (compute_metrics, aggregate_fold_metrics)
- Runner: `scripts/run_behaviot.py` (build_classifier_command, main)
- UER framework: `uer/` (encoders, embeddings, layers — unchanged)

### Needs setup
- Device MAC metadata loader: `data_generation/behaviot_data_gen.py` (new function)
- MAC-filtered extraction: `data_generation/behaviot_data_gen.py` (new function replacing `_extract_single_pcap`)
- Class weight computation: `data_generation/behaviot_data_gen.py` (new function)
- NLLLoss weight support: `fine-tuning/run_classifier.py` (modify Classifier.forward)
- BERT config for 1024: `models/bert/base_config_1024.json` (new file)
- Ablation configs: `configs/behaviot/ablation_*.json` (5 new files)

---

## Subtask 1: Device-MAC Single-Flow Extraction

**Hypothesis:** Filtering to device-specific flows removes background noise (DNS, NTP, phone traffic) and recovers ~565 previously skipped pcaps.

**Implementation:** Replace `_extract_single_pcap` with `_extract_by_device_mac` that:
1. Loads device MAC from `device_behaviot.txt`
2. Filters pcap to packets matching device MAC
3. Anonymizes and converts to bigrams directly (bypasses flowcontainer entirely)
4. No minimum packet count — padding handled by tokenizer

**Unit Tests:** MAC loading, packet filtering, bigram output format, empty pcap handling

**Validation Pyramid:** L0 at code review, L1 verify extraction produces valid datagrams on real pcaps

**Expected Conclusion:** ~7,000+ extractable samples (up from 5,799), cleaner single-device features

### Step 1: Write unit tests

Create `tests/test_mac_extraction.py`:

```python
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_generation.behaviot_data_gen import (
    load_device_macs,
    _extract_by_device_mac,
    extract_all_features,
)


class TestLoadDeviceMacs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mac_file = Path(self.tmp.name) / "device_macs.txt"
        self.mac_file.write_text(
            "amazon-plug ec:8a:c4:63:d8:e3 Amazon\n"
            "aqara-hub 54:ef:44:29:5e:eb Aqara\n"
            "tplink-bulb 50:c7:bf:a0:f3:76 TPLink\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_mac_mapping(self):
        macs = load_device_macs(str(self.mac_file))
        self.assertEqual(macs["amazon-plug"], "ec:8a:c4:63:d8:e3")
        self.assertEqual(macs["tplink-bulb"], "50:c7:bf:a0:f3:76")
        self.assertEqual(len(macs), 3)

    def test_mac_values_are_lowercase(self):
        macs = load_device_macs(str(self.mac_file))
        for mac in macs.values():
            self.assertEqual(mac, mac.lower())


class TestExtractByDeviceMac(unittest.TestCase):
    def test_returns_datagram_string(self):
        # Use a real pcap known to work
        result = _extract_by_device_mac((
            "/mnt/data/behavoiot/tagged-2021/amazon-plug/android_lan_off/2021-08-01_21:23:08.38s.pcap",
            64, 76, "ec:8a:c4:63:d8:e3"
        ))
        pcap_path, datagram, error = result
        self.assertIsNotNone(datagram, f"Expected datagram but got error: {error}")
        self.assertGreater(len(datagram), 0)
        self.assertIn(" ", datagram)  # space-separated bigrams

    def test_handles_no_matching_mac(self):
        result = _extract_by_device_mac((
            "/mnt/data/behavoiot/tagged-2021/amazon-plug/android_lan_off/2021-08-01_21:23:08.38s.pcap",
            64, 76, "ff:ff:ff:ff:ff:ff"  # non-existent MAC
        ))
        _, datagram, error = result
        self.assertIsNone(datagram)
        self.assertIn("no_matching_packets", error)

    def test_pads_short_pcaps(self):
        # Even pcaps with 1-2 device packets should return a datagram (not skip)
        result = _extract_by_device_mac((
            "/mnt/data/behavoiot/tagged-2021/aqara-hub/android_lan_on/2021-08-03_12:14:47.38s.pcap",
            64, 76, "54:ef:44:29:5e:eb"
        ))
        _, datagram, error = result
        # Should return whatever bigrams are available, even if short
        if error:
            self.assertNotIn("fewer_than", error)  # should NOT reject for being short


if __name__ == "__main__":
    unittest.main()
```

### Step 2: Run unit tests to verify they fail

Run:
```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_mac_extraction -v
```
Expected: FAIL with `ImportError` for `load_device_macs` and `_extract_by_device_mac`.

### Step 3: Implement core code

Add to `data_generation/behaviot_data_gen.py`:

```python
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

    Bypasses flowcontainer entirely — reads pcap directly with scapy,
    filters to packets involving the device MAC, anonymizes, and converts
    to bigrams. No minimum packet count; empty results produce empty strings.
    """
    pcap_path, payload_length, start_index, device_mac = args
    try:
        import binascii
        import scapy.all as scapy_mod

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

        import os, sys
        data_gen_dir = os.path.dirname(os.path.abspath(__file__))
        if data_gen_dir not in sys.path:
            sys.path.insert(0, data_gen_dir)
        from utils import bigram_generation, random_ipv4, random_field
        from data_generation.finetuning_data_gen import (
            random_ip_port, random_tls_randomtime,
        )

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
```

Update `extract_all_features()` to accept a `device_macs` parameter and the manifest rows with `device_label`:

```python
def extract_all_features(manifest_rows, config, cache_dir=None, n_workers=32,
                         device_macs=None):
    """Extract bigram features from all pcaps with multiprocessing and caching.

    If device_macs is provided, uses MAC-filtered single-flow extraction.
    Otherwise uses the legacy all-flows extraction.
    """
    # ... (cache key should include whether device_macs is used) ...

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

    # ... rest of multiprocessing logic uses extract_fn ...
```

### Step 4: Run unit tests to verify they pass

Run:
```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_mac_extraction -v
```
Expected: PASS

### Step 5: Validate on full dataset

Run:
```bash
cd /home/shinanliu/TrafficFormer
python3 -c "
from data_generation.behaviot_data_gen import load_device_macs, load_manifest, extract_all_features
macs = load_device_macs('/mnt/data/behavoiot/device_behaviot.txt')
rows = load_manifest('/mnt/data/behavoiot/pcap_vs_label.csv')
# Clear old cache first
import shutil; shutil.rmtree('generated/behaviot/cache', ignore_errors=True)
features, skipped = extract_all_features(rows, {'payload_length': 64, 'start_index': 76, 'manifest_csv': '/mnt/data/behavoiot/pcap_vs_label.csv'}, cache_dir='generated/behaviot/cache', device_macs=macs)
print(f'Extracted: {len(features)}, Skipped: {len(skipped)}')
"
```
Expected: ~7,000+ extracted (up from 5,799), fewer skips.

### Step 6: Commit

```bash
git add data_generation/behaviot_data_gen.py tests/test_mac_extraction.py
git commit -m "feat: device-MAC single-flow extraction bypassing flowcontainer"
```

---

## Subtask 2: Class-Weighted NLLLoss

**Hypothesis:** Inverse-sqrt class weights improve macro F1 for rare classes (routine, setpoint, volume) without degrading majority class performance.

**Implementation:**
1. `compute_class_weights()` in behaviot_data_gen.py
2. `--class_weights_path` argument in run_classifier.py
3. Pass weights to `NLLLoss(weight=...)`

**Unit Tests:** Weight computation correctness, loss with weights vs without

**Expected Conclusion:** Rare class F1 improves, macro F1 improves by 2-5%

### Step 1: Write unit tests

Add to `tests/test_behaviot_data_gen.py`:

```python
from data_generation.behaviot_data_gen import compute_class_weights

class TestClassWeights(unittest.TestCase):
    def test_inverse_sqrt_weights(self):
        class_counts = {"off": 1530, "on": 1282, "routine": 90}
        weights = compute_class_weights(class_counts, method="inverse_sqrt")
        # routine should have higher weight than off
        self.assertGreater(weights[2], weights[0])  # routine > off
        # All weights should be positive
        for w in weights:
            self.assertGreater(w, 0)

    def test_weights_length_matches_classes(self):
        class_counts = {"a": 100, "b": 50, "c": 10}
        weights = compute_class_weights(class_counts, method="inverse_sqrt")
        self.assertEqual(len(weights), 3)

    def test_no_weights_returns_none(self):
        class_counts = {"a": 100, "b": 50}
        weights = compute_class_weights(class_counts, method="none")
        self.assertIsNone(weights)
```

### Step 2: Run tests to verify they fail

Run:
```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen.TestClassWeights -v
```
Expected: FAIL with `ImportError`.

### Step 3: Implement core code

Add to `data_generation/behaviot_data_gen.py`:

```python
import math

def compute_class_weights(class_counts, method="inverse_sqrt"):
    """Compute class weights for weighted NLLLoss.

    Args:
        class_counts: dict mapping class_name to sample count (sorted by label ID)
        method: "inverse_sqrt" or "none"

    Returns:
        List of float weights (one per class, ordered by label ID) or None.
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
```

Modify `fine-tuning/run_classifier.py` — add `--class_weights_path` argument and modify loss computation:

In argument parsing (after `--labels_num_override`):
```python
parser.add_argument("--class_weights_path", type=str, default=None,
                    help="Path to JSON file with class weight list for NLLLoss.")
```

In Classifier.__init__, accept optional weights:
```python
# After self.soft_alpha = args.soft_alpha
self.class_weights = None
if hasattr(args, 'class_weights') and args.class_weights is not None:
    self.class_weights = args.class_weights
```

In Classifier.forward, use weighted NLLLoss:
```python
if self.class_weights is not None:
    loss = nn.NLLLoss(weight=self.class_weights.to(logits.device))(
        nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
else:
    loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
```

In main(), load class weights before model construction:
```python
args.class_weights = None
if args.class_weights_path:
    import json as _json
    weights_list = _json.loads(Path(args.class_weights_path).read_text())
    args.class_weights = torch.FloatTensor(weights_list)
```

### Step 4: Run tests to verify they pass

Run:
```bash
cd /home/shinanliu/TrafficFormer
python3 -m unittest tests.test_behaviot_data_gen.TestClassWeights -v
python3 -c "import ast; ast.parse(open('fine-tuning/run_classifier.py').read()); print('Syntax OK')"
```
Expected: PASS

### Step 5: Commit

```bash
git add data_generation/behaviot_data_gen.py fine-tuning/run_classifier.py tests/test_behaviot_data_gen.py
git commit -m "feat: class-weighted NLLLoss with inverse-sqrt frequency weighting"
```

---

## Subtask 3: Ablation Configs and Runner Updates

**Implementation:** Create configs for all 5 experiments, BERT config for 1024, update runner to pass class weights.

### Step 1: Create config files

`models/bert/base_config_1024.json`:
```json
{
  "emb_size": 768,
  "feedforward_size": 3072,
  "hidden_size": 768,
  "hidden_act": "gelu",
  "heads_num": 12,
  "layers_num": 12,
  "max_seq_length": 1024,
  "dropout": 0.1
}
```

`configs/behaviot/ablation_b0.json` (baseline, seq=320, no weights):
```json
{
    "manifest_csv": "/mnt/data/behavoiot/pcap_vs_label.csv",
    "device_macs_path": "/mnt/data/behavoiot/device_behaviot.txt",
    "n_folds": 5,
    "results_root": "results/behaviot_ablation",
    "generated_dataset_root": "generated/behaviot_ablation",
    "pretrained_model_zip": "/home/shinanliu/pre-trained_model.bin.zip",
    "pretrained_model_path": "models/pretrained/pre-trained_model.bin",
    "vocab_path": "models/encryptd_vocab.txt",
    "config_path": "models/bert/base_config.json",
    "payload_length": 64,
    "start_index": 76,
    "seq_length": 320,
    "learning_rate": 6e-5,
    "batch_size": 32,
    "epochs_num": 4,
    "earlystop": 4,
    "seed": 42,
    "max_samples_per_class": null,
    "min_samples_per_class": 5,
    "class_weight_method": "none"
}
```

`configs/behaviot/ablation_a1.json` — same but `"seq_length": 512`

`configs/behaviot/ablation_a2.json` — `"seq_length": 1024`, `"config_path": "models/bert/base_config_1024.json"`

`configs/behaviot/ablation_a3.json` — `"seq_length": 320`, `"class_weight_method": "inverse_sqrt"`

`configs/behaviot/ablation_a4.json` — best seq_length + `"class_weight_method": "inverse_sqrt"` (fill after A1/A2 results)

### Step 2: Update runner

In `scripts/run_behaviot.py`, update `main()`:

1. Load device MACs if `device_macs_path` in config:
```python
device_macs = None
if config.get("device_macs_path"):
    from data_generation.behaviot_data_gen import load_device_macs
    device_macs = load_device_macs(config["device_macs_path"])
```

2. Pass `device_macs` to `extract_all_features()`.

3. Compute and save class weights per fold if `class_weight_method != "none"`:
```python
if config.get("class_weight_method", "none") != "none":
    from data_generation.behaviot_data_gen import compute_class_weights
    from collections import Counter
    train_counts = Counter(r[target] for r in fold_splits[0] if r["pcap_path"] in features)
    weights = compute_class_weights(dict(sorted(train_counts.items())), config["class_weight_method"])
    weights_path = fold_results_dir / "class_weights.json"
    weights_path.write_text(json.dumps(weights))
```

4. Pass `--class_weights_path` in `build_classifier_command()` when weights exist.

### Step 3: Commit

```bash
git add models/bert/base_config_1024.json configs/behaviot/ablation_*.json scripts/run_behaviot.py
git commit -m "feat: ablation configs and runner support for class weights + device MAC extraction"
```

---

## Subtask 4: Run New Baseline (B0)

**Hypothesis:** Device-MAC filtered extraction alone improves activity classification by providing cleaner features.

### Step 1: Clear caches and run B0

```bash
cd /home/shinanliu/TrafficFormer
rm -rf generated/behaviot_ablation/cache
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_b0.json --target activity_type_label
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_b0.json --target activity_label
```

### Step 2: Record B0 results

Compare against original baseline:
- Original: activity_type_label macro F1 = 0.5493, activity_label macro F1 = 0.3007
- B0: [record actual results]

### Step 3: Commit results

```bash
git add results/behaviot_ablation/
git commit -m "results: B0 baseline with device-MAC filtered extraction"
```

---

## Subtask 5: Run Ablations A1-A4

### Step 1: Run A1 (seq_length=512)

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_a1.json --target activity_type_label
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_a1.json --target activity_label
```

### Step 2: Run A2 (seq_length=1024)

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_a2.json --target activity_type_label
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_a2.json --target activity_label
```

### Step 3: Run A3 (class weights, seq=320)

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_a3.json --target activity_type_label
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_a3.json --target activity_label
```

### Step 4: Determine best seq_length and run A4

Compare A1 vs A2 macro F1. Set A4 config to the better seq_length + class weights.

```bash
# Update ablation_a4.json with best seq_length
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_a4.json --target activity_type_label
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_behaviot.py --config configs/behaviot/ablation_a4.json --target activity_label
```

### Step 5: Record all results and conclusions

Create comparison table:

| Experiment | activity_type F1 | activity F1 | Delta vs B0 |
|-----------|-----------------|------------|-------------|
| B0 (baseline, seq=320) | ? | ? | — |
| A1 (seq=512) | ? | ? | ? |
| A2 (seq=1024) | ? | ? | ? |
| A3 (class weights) | ? | ? | ? |
| A4 (combined) | ? | ? | ? |

### Step 6: Commit results

```bash
git add results/behaviot_ablation/ configs/behaviot/ablation_a4.json
git commit -m "results: A1-A4 ablation results for activity classification improvement"
```
