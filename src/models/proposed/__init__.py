"""Optional components for the proposed vision-language research track."""

from .consistency import ConsistencyResult, ConsistencyLabel, assess_consistency
from .counterfactual import (
    CounterfactualRecord,
    evaluate_counterfactuals,
    summarize_counterfactuals,
)
from .evidence import EvidenceItem, EvidenceRetriever
from .explainability import Explanation, build_explanation
from .integration import IntegratedPrediction, integrate_prediction
from .vlm_analyzer import VLMAnalysis, VLMAnalyzer

__all__ = [
    "ConsistencyLabel",
    "ConsistencyResult",
    "CounterfactualRecord",
    "EvidenceItem",
    "EvidenceRetriever",
    "Explanation",
    "IntegratedPrediction",
    "VLMAnalysis",
    "VLMAnalyzer",
    "assess_consistency",
    "build_explanation",
    "evaluate_counterfactuals",
    "integrate_prediction",
    "summarize_counterfactuals",
]
