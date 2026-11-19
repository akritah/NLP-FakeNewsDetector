import unittest

from PIL import Image

from src.models.proposed.consistency import ConsistencyLabel, assess_consistency
from src.models.proposed.counterfactual import (
    evaluate_counterfactuals,
    summarize_counterfactuals,
)
from src.models.proposed.integration import integrate_prediction
from src.models.proposed.vlm_analyzer import VLMAnalyzer


class TestProposedComponents(unittest.TestCase):
    def test_consistency_is_separate_from_classification(self):
        result = assess_consistency(
            "A claim",
            "A caption",
            label=ConsistencyLabel.INCONSISTENT,
            confidence=0.8,
        )
        prediction = integrate_prediction(
            "REAL",
            0.9,
            consistency=result,
        )
        self.assertEqual(prediction.predicted_label, "REAL")
        self.assertEqual(prediction.consistency.label, ConsistencyLabel.INCONSISTENT)

    def test_vlm_analyzer_accepts_injected_backend(self):
        class Processor:
            def __call__(self, images, return_tensors):
                return {"pixel_values": "pixels"}

            def decode(self, output, skip_special_tokens):
                return "a person outdoors"

        class Model:
            def eval(self):
                return self

            def generate(self, **inputs):
                return ["caption"]

        analysis = VLMAnalyzer(model=Model(), processor=Processor()).analyze(
            "A person is outdoors",
            Image.new("RGB", (4, 4)),
        )
        self.assertEqual(analysis.image_description, "a person outdoors")

    def test_counterfactuals_keep_original_label(self):
        samples = [{
            "id": "1",
            "text": "claim",
            "label": 0,
            "images": {"original": "a", "mismatched": "b"},
        }]

        def predict(text, image):
            return (0 if image == "a" else 1), 0.75

        records = evaluate_counterfactuals(samples, predict)
        summary = summarize_counterfactuals(records)
        self.assertEqual([record.true_label for record in records], [0, 0])
        self.assertEqual(summary["mismatched"]["prediction_change_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
