"""Integration contract for classifier, VLM, consistency, and evidence outputs."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .consistency import ConsistencyResult
from .evidence import EvidenceItem
from .explainability import Explanation, build_explanation
from .vlm_analyzer import VLMAnalysis


@dataclass
class IntegratedPrediction:
    predicted_label: str
    classification_confidence: Optional[float]
    vlm_analysis: Optional[VLMAnalysis]
    consistency: Optional[ConsistencyResult]
    explanation: Explanation
    evidence: List[EvidenceItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "classification_confidence": self.classification_confidence,
            "vlm_analysis": self.vlm_analysis.to_dict() if self.vlm_analysis else None,
            "consistency": self.consistency.to_dict() if self.consistency else None,
            "explanation": self.explanation.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
        }


def integrate_prediction(
    predicted_label: str,
    classification_confidence: Optional[float],
    *,
    vlm_analysis: Optional[VLMAnalysis] = None,
    consistency: Optional[ConsistencyResult] = None,
    evidence: Optional[List[EvidenceItem]] = None,
    textual_signals: Optional[List[str]] = None,
) -> IntegratedPrediction:
    evidence_items = evidence or []
    return IntegratedPrediction(
        predicted_label=predicted_label,
        classification_confidence=classification_confidence,
        vlm_analysis=vlm_analysis,
        consistency=consistency,
        explanation=build_explanation(
            predicted_label,
            classification_confidence,
            vlm_analysis=vlm_analysis,
            consistency=consistency,
            evidence=evidence_items,
            textual_signals=textual_signals,
        ),
        evidence=evidence_items,
    )
