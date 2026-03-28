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
            "min_samples_per_class": 5, "max_samples_per_class": None,
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
