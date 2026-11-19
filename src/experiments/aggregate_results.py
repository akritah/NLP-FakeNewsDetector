import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate(root: str) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted(Path(root).glob("*/run_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = metadata.get("test_metrics", {})
        rows.append({
            "experiment": metadata.get("experiment"),
            "seed": metadata.get("seed"),
            "accuracy": metrics.get("accuracy"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "f1_macro": metrics.get("f1_macro"),
            "roc_auc": metrics.get("roc_auc"),
            "training_seconds": metadata.get("training_seconds"),
            "inference_seconds_per_sample": metadata.get(
                "test_inference_seconds_per_sample"
            ),
            "parameters_total": metadata.get("parameter_counts", {}).get("total"),
            "parameters_trainable": metadata.get("parameter_counts", {}).get(
                "trainable"
            ),
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    numeric = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
        "training_seconds",
        "inference_seconds_per_sample",
    ]
    summary = results.groupby("experiment")[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/research")
    parser.add_argument("--output", default="outputs/research/comparison.csv")
    args = parser.parse_args()
    results = aggregate(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
