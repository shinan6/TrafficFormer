import json
import math
import tempfile
import unittest
from pathlib import Path

from data_generation.behaviot_data_gen import (
    load_device_macs,
    _extract_by_device_mac,
    compute_class_weights,
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
    def test_returns_datagram_for_known_device(self):
        result = _extract_by_device_mac((
            "/mnt/data/behavoiot/tagged-2021/amazon-plug/android_lan_off/2021-08-01_21:23:08.38s.pcap",
            64, 76, "ec:8a:c4:63:d8:e3"
        ))
        _, datagram, error = result
        self.assertIsNotNone(datagram, f"Expected datagram but got error: {error}")
        self.assertGreater(len(datagram), 0)
        self.assertIn(" ", datagram)

    def test_no_matching_mac_returns_skip(self):
        result = _extract_by_device_mac((
            "/mnt/data/behavoiot/tagged-2021/amazon-plug/android_lan_off/2021-08-01_21:23:08.38s.pcap",
            64, 76, "ff:ff:ff:ff:ff:ff"
        ))
        _, datagram, error = result
        self.assertIsNone(datagram)
        self.assertIn("no_matching_packets", error)

    def test_short_pcap_not_rejected(self):
        """Even pcaps with 1-2 device packets should return a datagram."""
        result = _extract_by_device_mac((
            "/mnt/data/behavoiot/tagged-2021/aqara-hub/android_lan_on/2021-08-03_12:14:47.38s.pcap",
            64, 76, "54:ef:44:29:5e:eb"
        ))
        _, datagram, error = result
        # Should produce something (even if short) or skip gracefully, but NOT crash
        if error:
            self.assertNotIn("fewer_than", error)


class TestComputeClassWeights(unittest.TestCase):
    def test_inverse_sqrt_weights(self):
        counts = {"off": 1530, "on": 1282, "routine": 90}
        weights = compute_class_weights(counts, method="inverse_sqrt")
        # routine (smallest) should have highest weight
        self.assertGreater(weights[2], weights[0])  # routine > off (sorted order)
        for w in weights:
            self.assertGreater(w, 0)

    def test_weights_length_matches_classes(self):
        counts = {"a": 100, "b": 50, "c": 10}
        weights = compute_class_weights(counts, method="inverse_sqrt")
        self.assertEqual(len(weights), 3)

    def test_no_weights_returns_none(self):
        counts = {"a": 100, "b": 50}
        weights = compute_class_weights(counts, method="none")
        self.assertIsNone(weights)

    def test_inverse_sqrt_formula(self):
        counts = {"a": 100, "b": 400}
        weights = compute_class_weights(counts, method="inverse_sqrt")
        # weight_a = sqrt(500 / (2 * 100)) = sqrt(2.5)
        # weight_b = sqrt(500 / (2 * 400)) = sqrt(0.625)
        self.assertAlmostEqual(weights[0], math.sqrt(2.5), places=5)
        self.assertAlmostEqual(weights[1], math.sqrt(0.625), places=5)


if __name__ == "__main__":
    unittest.main()
