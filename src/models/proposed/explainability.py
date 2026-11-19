"""Structured explanations that distinguish observations from evidence."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .consistency import ConsistencyResult
from .evidence import EvidenceItem
from .vlm_analyzer import VLMAnalysis


@dataclass
class Explanation:
    predicted_label: str
    classification_confidence: Optional[float]
    textual_signals: List[str] = field(default_factory=list)
    visual_observations: List[str] = field(default_factory=list)
    consistency: Optional[ConsistencyResult] = None
    uncertainty_reasons: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "classification_confidence": self.classification_confidence,
            "textual_signals": self.textual_signals,
            "visual_observations": self.visual_observations,
            "consistency": self.consistency.to_dict() if self.consistency else None,
            "uncertainty_reasons": self.uncertainty_reasons,
            "evidence": [item.to_dict() for item in self.evidence],
        }


def build_explanation(
    predicted_label: str,
    classification_confidence: Optional[float],
    *,
    vlm_analysis: Optional[VLMAnalysis] = None,
    consistency: Optional[ConsistencyResult] = None,
    evidence: Optional[List[EvidenceItem]] = None,
    textual_signals: Optional[List[str]] = None,
) -> Explanation:
    observations = []
    uncertainty = []
    if vlm_analysis:
        observations.append(vlm_analysis.image_description)
    if consistency and consistency.label.value == "uncertain":
        uncertainty.append("The claim-image consistency assessment is uncertain.")
    if not evidence:
        uncertainty.append("No external evidence source was configured or returned.")
    return Explanation(
        predicted_label=predicted_label,
        classification_confidence=classification_confidence,
        textual_signals=textual_signals or [],
        visual_observations=observations,
        consistency=consistency,
        uncertainty_reasons=uncertainty,
        evidence=evidence or [],
    )
