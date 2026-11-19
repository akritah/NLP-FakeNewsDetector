"""Interfaces for optional external evidence retrieval."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol


@dataclass
class EvidenceItem:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    credibility: Optional[float] = None
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "credibility": self.credibility,
            "verified": self.verified,
        }


class EvidenceRetriever(Protocol):
    def search(self, claim: str, limit: int = 5) -> List[EvidenceItem]:
        ...


class DisabledEvidenceRetriever:
    """Explicit no-op implementation used when no retrieval API is configured."""

    def search(self, claim: str, limit: int = 5) -> List[EvidenceItem]:
        if not claim.strip():
            raise ValueError("claim must be a non-empty string")
        return []
