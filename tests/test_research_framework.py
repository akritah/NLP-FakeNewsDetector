import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.data.audit_splits import audit
from src.utils.experiment_tracking import count_parameters


class TinyModel:
    class Parameter:
        def __init__(self, numel, requires_grad=True):
            self.numel = lambda: numel
            self.requires_grad = requires_grad

    def parameters(self):
        return [self.Parameter(10), self.Parameter(3, False)]


class TestResearchFramework(unittest.TestCase):
    def test_fixed_split_integrity(self):
        report = audit("data/splits")
        self.assertEqual(report["id_overlap"]["train_val"], 0)
        self.assertEqual(report["id_overlap"]["train_test"], 0)
        self.assertEqual(report["id_overlap"]["val_test"], 0)
        self.assertEqual(report["record_overlap"]["train_val"], 0)
        self.assertEqual(report["record_overlap"]["train_test"], 0)
        self.assertEqual(report["record_overlap"]["val_test"], 0)

    def test_experiment_config_is_separate_and_fixed(self):
        with open("configs/experiments/cross_attention_vit.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.assertEqual(config["dataset"]["split_dir"], "data/splits")
        self.assertEqual(config["model"]["fusion_method"], "cross_attention")
        self.assertNotEqual(config["experiment"]["name"], "baseline")

    def test_parameter_count(self):
        counts = count_parameters(TinyModel())
        self.assertEqual(counts["total"], 13)
        self.assertEqual(counts["trainable"], 10)


if __name__ == "__main__":
    unittest.main()
