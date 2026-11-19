"""Executable counterfactual image/text interventions for held-out samples."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
from PIL import Image

from src.models.proposed.counterfactual import (
    CounterfactualRecord,
    evaluate_counterfactuals,
    summarize_counterfactuals,
)


@dataclass
class InterventionSummary:
    condition: str
    count: int
    prediction_change_rate: float
    mean_confidence_delta: float
    mean_probability_delta: float


def build_image_interventions(
    frame: pd.DataFrame,
    image_dir: str,
    *,
    limit: Optional[int] = None,
    seed: int = 42,
) -> List[dict]:
    """Create original, random, same-label, and opposite-label image pairs.

    The label is never changed after replacement. Same/opposite-label image
    pools are metadata-controlled interventions, not ground-truth relabeling.
    """
    source = frame.head(limit).copy() if limit else frame.copy()
    rng = source.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    samples = []
    for _, row in source.iterrows():
        same = source[source["label"] == row["label"]]
        opposite = source[source["label"] != row["label"]]
        random_row = rng[rng["id"] != row["id"]].iloc[0]
        same_row = same[same["id"] != row["id"]].iloc[0] if len(same) > 1 else row
        opposite_row = opposite.iloc[0] if not opposite.empty else row

        def image_path(candidate):
            return str(Path(image_dir) / str(candidate["image_url"]))

        samples.append({
            "id": row["id"],
            "text": str(row["title"]),
            "label": int(row["label"]),
            "images": {
                "original": image_path(row),
                "random": image_path(random_row),
                "same_label": image_path(same_row),
                "opposite_label": image_path(opposite_row),
                "ablated": None,
            },
        })
    return samples


def perturb_text(text: str, mode: str = "remove_last_clause") -> str:
    """Apply a transparent text intervention without changing its label."""
    if not text.strip():
        return text
    if mode == "remove_last_clause":
        parts = text.split(",")
        return ",".join(parts[:-1]).strip() if len(parts) > 1 else text
    if mode == "mask_claim":
        words = text.split()
        midpoint = max(1, len(words) // 2)
        return " ".join(words[:midpoint] + ["[MASK]"] + words[midpoint + 1:])
    raise ValueError(f"Unsupported text perturbation mode: {mode}")


def build_text_interventions(samples: Iterable[dict], mode: str = "mask_claim") -> List[dict]:
    """Create text interventions while preserving each sample's image/label."""
    return [
        {
            "id": sample["id"],
            "text": perturb_text(sample["text"], mode),
            "label": sample["label"],
            "image": sample["images"]["original"],
        }
        for sample in samples
    ]


def load_rgb(path: Optional[str]):
    if path is None:
        return None
    return Image.open(path).convert("RGB")


def summarize_records(records: Iterable[CounterfactualRecord]) -> pd.DataFrame:
    summary = summarize_counterfactuals(list(records))
    return pd.DataFrame(
        [{"condition": condition, **values} for condition, values in summary.items()]
    )
