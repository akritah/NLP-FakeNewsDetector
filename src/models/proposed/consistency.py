"""Explicit claim-image consistency labels independent of REAL/FAKE labels."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConsistencyLabel(str, Enum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNRELATED = "unrelated"
    UNCERTAIN = "uncertain"


@dataclass
class ConsistencyResult:
    label: ConsistencyLabel
    confidence: Optional[float]
    rationale: str

    def to_dict(self):
        return {
            "label": self.label.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def assess_consistency(
    claim: str,
    image_description: str,
    *,
    label: ConsistencyLabel = ConsistencyLabel.UNCERTAIN,
    confidence: Optional[float] = None,
    rationale: Optional[str] = None,
) -> ConsistencyResult:
    """Normalize an externally produced consistency assessment.

    This function intentionally does not infer factual truth from string
    overlap. A trained or prompted consistency model can provide the label;
    callers that lack such a model must retain ``UNCERTAIN``.
    """
    if not claim.strip() or not image_description.strip():
        raise ValueError("claim and image_description must be non-empty")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return ConsistencyResult(
        label=label,
        confidence=confidence,
        rationale=rationale or "No independently calibrated consistency rationale was supplied.",
    )
