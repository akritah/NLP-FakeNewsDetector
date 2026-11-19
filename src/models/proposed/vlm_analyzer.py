"""Lazy, optional vision-language analysis for a claim and an image."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class VLMAnalysis:
    claim: str
    image_description: str
    objects: List[str] = field(default_factory=list)
    visual_context: str = ""
    relevance: str = "uncertain"
    support: str = "uncertain"
    confidence: Optional[float] = None
    explanation: str = ""
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "image_description": self.image_description,
            "objects": self.objects,
            "visual_context": self.visual_context,
            "relevance": self.relevance,
            "support": self.support,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "model_name": self.model_name,
        }


class VLMAnalyzer:
    """Generate an image description using an optional BLIP checkpoint.

    Captioning is deliberately kept separate from factual verification. The
    analyzer reports visual observations and leaves claim truth to the
    classifier and any independently retrieved evidence.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
        model: Any = None,
        processor: Any = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = model
        self._processor = processor

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                "VLM analysis requires transformers with BLIP support."
            ) from exc
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        if self.device:
            self._model.to(self.device)
        self._model.eval()

    def analyze(self, claim: str, image: Image.Image) -> VLMAnalysis:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        self._load()
        inputs = self._processor(images=image, return_tensors="pt")
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self._model.generate(**inputs, max_new_tokens=40)
        description = self._processor.decode(outputs[0], skip_special_tokens=True)
        return VLMAnalysis(
            claim=claim,
            image_description=description,
            visual_context=description,
            relevance="uncertain",
            support="uncertain",
            explanation=(
                "The caption describes visible content only. Claim support "
                "requires a separate consistency analysis and evidence."
            ),
            model_name=self.model_name,
        )
