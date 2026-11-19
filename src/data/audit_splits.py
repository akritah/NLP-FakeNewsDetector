import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def text_key(value):
    return " ".join(str(value).strip().lower().split())


def audit(split_dir: str) -> dict:
    split_path = Path(split_dir)
    frames = {}
    for name in ("train", "val", "test"):
        path = split_path / f"{name}.csv"
        frames[name] = pd.read_csv(path)

    def keys(frame):
        return set(
            zip(
                frame["id"].astype(str),
                frame["title"].map(text_key),
                frame["image_url"].astype(str),
            )
        )

    train_keys = keys(frames["train"])
    val_keys = keys(frames["val"])
    test_keys = keys(frames["test"])

    def overlap(left_name, left, right_name, right):
        return len(left & right)

    report = {
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "id_overlap": {
            "train_val": len(set(frames["train"]["id"]) & set(frames["val"]["id"])),
            "train_test": len(set(frames["train"]["id"]) & set(frames["test"]["id"])),
            "val_test": len(set(frames["val"]["id"]) & set(frames["test"]["id"])),
        },
        "record_overlap": {
            "train_val": overlap("train", train_keys, "val", val_keys),
            "train_test": overlap("train", train_keys, "test", test_keys),
            "val_test": overlap("val", val_keys, "test", test_keys),
        },
        "duplicate_ids_within_splits": {
            name: int(frame["id"].duplicated().sum())
            for name, frame in frames.items()
        },
        "missing_text": {
            name: int(frame["title"].isna().sum())
            for name, frame in frames.items()
        },
        "missing_or_invalid_labels": {
            name: int((~frame["label"].astype(str).isin({"0", "1"})).sum())
            for name, frame in frames.items()
        },
        "class_distribution": {
            name: frame["label"].astype(str).value_counts().to_dict()
            for name, frame in frames.items()
        },
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output", default="data/splits/integrity_report.json")
    args = parser.parse_args()
    report = audit(args.split_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
