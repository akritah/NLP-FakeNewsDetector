"""Counterfactual image-swap evaluation without changing ground-truth labels."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence


@dataclass
class CounterfactualRecord:
    sample_id: Any
    condition: str
    true_label: Any
    predicted_label: Any
    confidence: float


def evaluate_counterfactuals(
    samples: Iterable[Dict[str, Any]],
    predict: Callable[[str, Any], tuple],
) -> List[CounterfactualRecord]:
    """Evaluate supplied image conditions using an existing classifier.

    Each sample must contain ``id``, ``text``, ``label`` and an ``images``
    mapping with conditions such as ``original``, ``mismatched``,
    ``similar``, or ``contradictory``. The original ground-truth label is
    retained for every condition; swapping an image does not relabel a sample.
    """
    records = []
    for sample in samples:
        images = sample.get("images", {})
        for condition, image in images.items():
            predicted, confidence = predict(sample["text"], image)
            records.append(
                CounterfactualRecord(
                    sample_id=sample["id"],
                    condition=condition,
                    true_label=sample["label"],
                    predicted_label=predicted,
                    confidence=float(confidence),
                )
            )
    return records


def summarize_counterfactuals(records: Sequence[CounterfactualRecord]) -> Dict[str, Any]:
    """Summarize accuracy and prediction changes by condition."""
    by_condition: Dict[str, List[CounterfactualRecord]] = {}
    for record in records:
        by_condition.setdefault(record.condition, []).append(record)
    original = {
        record.sample_id: record
        for record in by_condition.get("original", [])
    }
    summary = {}
    for condition, condition_records in by_condition.items():
        correct = sum(record.predicted_label == record.true_label for record in condition_records)
        changes = sum(
            record.sample_id in original
            and record.predicted_label != original[record.sample_id].predicted_label
            for record in condition_records
        )
        summary[condition] = {
            "count": len(condition_records),
            "accuracy": correct / len(condition_records) if condition_records else None,
            "prediction_change_rate": changes / len(condition_records) if condition_records else None,
            "mean_confidence": (
                sum(record.confidence for record in condition_records) / len(condition_records)
                if condition_records else None
            ),
        }
    return summary
