"""Audit verified multimodal data and create reproducible stratified splits."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def _normalise_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_and_split(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    text_column: str = "title",
    image_column: str = "image_url",
    label_column: str = "label",
    id_column: str = "id",
    seed: int = 42,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> dict:
    frame = pd.read_csv(csv_path)
    required = {id_column, text_column, image_column, label_column}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    frame = frame.copy()
    frame["_image_path"] = frame[image_column].map(
        lambda value: str(Path(image_dir) / str(value))
    )
    frame["_text_key"] = frame[text_column].map(_normalise_text)
    frame["_image_hash"] = None
    valid_rows = []
    invalid_reasons = {}

    for index, row in frame.iterrows():
        image_path = Path(row["_image_path"])
        reason = None
        if not row[id_column] or not row["_text_key"]:
            reason = "missing_id_or_text"
        elif str(row[label_column]) not in {"0", "1"}:
            reason = "invalid_label"
        elif not image_path.is_file() or image_path.stat().st_size == 0:
            reason = "missing_or_empty_image"
        else:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                frame.at[index, "_image_hash"] = _file_sha256(image_path)
            except Exception:
                reason = "corrupt_image"
        if reason:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        else:
            valid_rows.append(index)

    clean = frame.loc[valid_rows].copy()
    before_duplicates = len(clean)
    clean = clean.drop_duplicates(subset=[id_column], keep="first")
    clean = clean.drop_duplicates(subset=["_text_key", "_image_hash"], keep="first")
    duplicate_count = before_duplicates - len(clean)

    if clean[label_column].nunique() < 2:
        raise ValueError("Both binary classes are required for stratified splits.")

    train_frame, remainder = train_test_split(
        clean,
        train_size=train_size,
        stratify=clean[label_column],
        random_state=seed,
    )
    val_fraction_of_remainder = val_size / (1.0 - train_size)
    val_frame, test_frame = train_test_split(
        remainder,
        train_size=val_fraction_of_remainder,
        stratify=remainder[label_column],
        random_state=seed,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    public_columns = [column for column in frame.columns if not column.startswith("_")]
    for name, split in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
        split[public_columns].to_csv(output / f"{name}.csv", index=False)

    manifest = {
        "source_csv": str(csv_path),
        "image_dir": str(image_dir),
        "seed": seed,
        "splits": {"train": len(train_frame), "val": len(val_frame), "test": len(test_frame)},
        "class_distribution": {
            name: split[label_column].astype(str).value_counts().to_dict()
            for name, split in (("train", train_frame), ("val", val_frame), ("test", test_frame))
        },
        "invalid_rows": invalid_reasons,
        "duplicates_removed": duplicate_count,
        "verified_rows": len(clean),
        "id_overlap": {
            "train_val": len(set(train_frame[id_column]) & set(val_frame[id_column])),
            "train_test": len(set(train_frame[id_column]) & set(test_frame[id_column])),
            "val_test": len(set(val_frame[id_column]) & set(test_frame[id_column])),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/fakeddit_verified.csv")
    parser.add_argument("--images", default="data/raw/images")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(audit_and_split(args.csv, args.images, args.output, seed=args.seed), indent=2))
