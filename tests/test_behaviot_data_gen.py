import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from data_generation.behaviot_data_gen import (
    load_config,
    load_manifest,
    build_label_map,
    extract_all_features,
    build_kfold_splits,
    write_fold_tsvs,
    cap_samples_per_class,
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
        self.assertIn("off", lmap)
        self.assertIn("on", lmap)
        self.assertNotIn("audio", lmap)
        self.assertEqual(len(lmap), 2)

    def test_build_label_map_works_for_any_column(self):
        rows = load_manifest(str(self.csv_path))
        for col in ["device_type_label", "device_label",
                     "activity_type_label", "activity_label"]:
            lmap = build_label_map(rows, col)
            self.assertGreater(len(lmap), 0)


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
        self.assertEqual(mock_extract.call_count, call_count_after_first)
        self.assertEqual(f1, f2)


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
        # Need enough samples for stratified split to work (>=n_folds per class, dev feasible)
        rows = [{"pcap_path": f"/tmp/{i}.pcap", "label": "A" if i % 2 == 0 else "B"} for i in range(40)]
        rows.append({"pcap_path": "/tmp/unknown.pcap", "label": "UNKNOWN"})
        folds = build_kfold_splits(rows, "label", {"A": 0, "B": 1}, n_folds=2, seed=42)
        all_paths = set()
        for train, dev, test in folds:
            for r in train + dev + test:
                all_paths.add(r["pcap_path"])
        self.assertNotIn("/tmp/unknown.pcap", all_paths)

    def test_infeasible_split_raises_error(self):
        rows = [{"pcap_path": "/tmp/0.pcap", "label": "A"},
                {"pcap_path": "/tmp/1.pcap", "label": "B"}]
        label_to_id = {"A": 0, "B": 1}
        with self.assertRaises(ValueError):
            build_kfold_splits(rows, "label", label_to_id, n_folds=5, seed=42)


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

    def test_tsv_format_matches_run_classifier(self):
        train = [{"pcap_path": "/tmp/a.pcap", "lbl": "A"}]
        features = {"/tmp/a.pcap": "45 00"}
        label_to_id = {"A": 0, "B": 1}
        write_fold_tsvs((train, [], []), features, label_to_id, "lbl", str(self.output_dir))
        with open(self.output_dir / "train_dataset.tsv") as f:
            lines = f.readlines()
        self.assertEqual(lines[0].strip(), "label\ttext_a")
        parts = lines[1].strip().split("\t")
        self.assertEqual(parts[0], "0")
        self.assertEqual(parts[1], "45 00")

    def test_skips_pcaps_without_features(self):
        train = [{"pcap_path": "/tmp/a.pcap", "lbl": "A"},
                 {"pcap_path": "/tmp/missing.pcap", "lbl": "B"}]
        features = {"/tmp/a.pcap": "45 00"}
        label_to_id = {"A": 0, "B": 1}
        write_fold_tsvs((train, [], []), features, label_to_id, "lbl", str(self.output_dir))
        with open(self.output_dir / "train_dataset.tsv") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)


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


class TestExtractionNoTruncation(unittest.TestCase):
    """AC-1.1: verify packets_num=999999 is passed (no extraction-time truncation)."""

    @patch("data_generation.behaviot_data_gen._extract_single_pcap")
    def test_extract_passes_large_packets_num(self, mock_extract):
        """The worker function should call get_feature_flow with packets_num=999999."""
        mock_extract.side_effect = lambda args: (args[0], "45 00", None)
        rows = [{"pcap_path": "/tmp/a.pcap"}]
        config = {"payload_length": 64, "start_index": 76}
        extract_all_features(rows, config, n_workers=1)
        # Verify _extract_single_pcap was called with the right args
        call_args = mock_extract.call_args[0][0]
        self.assertEqual(call_args, ("/tmp/a.pcap", 64, 76))

    def test_worker_function_uses_999999(self):
        """Verify the worker function source code passes packets_num=999999."""
        import inspect
        from data_generation.behaviot_data_gen import _extract_single_pcap
        source = inspect.getsource(_extract_single_pcap)
        self.assertIn("packets_num=999999", source)


class TestKFoldStratifiedFallback(unittest.TestCase):
    """AC-2: verify inner dev split falls back gracefully for high-class targets."""

    def test_fallback_split_still_produces_dev(self):
        """With many classes and few samples, non-stratified fallback should still work."""
        # 20 classes, 2 samples each = 40 total; 2-fold outer = 20 train
        # 10% dev = 2 samples, but 20 classes can't stratify into 2 → fallback
        rows = [{"pcap_path": f"/tmp/{c}_{i}.pcap", "label": f"class_{c}"}
                for c in range(20) for i in range(2)]
        label_to_id = {f"class_{c}": c for c in range(20)}
        folds = build_kfold_splits(rows, "label", label_to_id, n_folds=2, seed=42)
        for train, dev, test in folds:
            self.assertGreater(len(dev), 0, "Dev split should not be empty after fallback")
            self.assertGreater(len(train), 0)
            self.assertGreater(len(test), 0)


if __name__ == "__main__":
    unittest.main()
