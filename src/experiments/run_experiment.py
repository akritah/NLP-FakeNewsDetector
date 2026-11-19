import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import (
    CLIPProcessor,
    RobertaTokenizer,
    ViTImageProcessor,
)
from tqdm import tqdm

from src.data.dataset import FakedditDataset
from src.models.research_model import ResearchMultimodalModel
from src.utils.checkpointing import save_checkpoint
from src.utils.experiment_tracking import count_parameters, write_run_metadata
from src.utils.metrics import compute_metrics, get_confusion_matrix
from src.utils.reproducibility import set_seed


def create_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    predictions = []
    labels = []
    loss_total = 0.0
    for batch, batch_labels in tqdm(loader, desc="train", leave=False):
        text_inputs = {
            key: value.to(device) for key, value in batch["text"].items()
        }
        image_inputs = {
            key: value.to(device) for key, value in batch["image"].items()
        }
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(text_inputs, image_inputs, batch.get("raw_text"))
            loss = criterion(logits, batch_labels)

        if scaler:
            scaler.scale(loss).backward()
            if hasattr(torch.nn.utils, "clip_grad_norm_"):
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_total += float(loss.detach().cpu())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels.extend(batch_labels.detach().cpu().tolist())
    metrics = compute_metrics(labels, predictions)
    return loss_total / max(len(loader), 1), metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    predictions = []
    labels = []
    probabilities = []
    loss_total = 0.0
    started = time.perf_counter()
    for batch, batch_labels in tqdm(loader, desc="eval", leave=False):
        text_inputs = {
            key: value.to(device) for key, value in batch["text"].items()
        }
        image_inputs = {
            key: value.to(device) for key, value in batch["image"].items()
        }
        batch_labels = batch_labels.to(device)
        logits = model(text_inputs, image_inputs, batch.get("raw_text"))
        loss = criterion(logits, batch_labels)
        loss_total += float(loss.detach().cpu())
        probabilities.extend(
            torch.softmax(logits, dim=1).detach().cpu().numpy()
        )
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels.extend(batch_labels.detach().cpu().tolist())
    elapsed = time.perf_counter() - started
    metrics = compute_metrics(labels, predictions, np.asarray(probabilities))
    batch_count = len(loader)
    sample_count = len(labels)
    return (
        loss_total / max(batch_count, 1),
        metrics,
        labels,
        predictions,
        probabilities,
        elapsed / max(sample_count, 1),
    )


def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visual_config = config["model"].get("visual_encoder", {})
    if visual_config.get("type", "vit") == "clip":
        processor = CLIPProcessor.from_pretrained(
            visual_config.get("checkpoint", "openai/clip-vit-base-patch32")
        )
    else:
        processor = ViTImageProcessor.from_pretrained(
            visual_config.get("checkpoint", "google/vit-base-patch16-224")
        )
    tokenizer = RobertaTokenizer.from_pretrained(
        config["model"]["text_encoder"]["checkpoint"]
    )

    dataset_kwargs = {
        "image_dir": config["dataset"]["image_dir"],
        "text_col": config["dataset"]["text_column"],
        "image_col": config["dataset"]["image_column"],
        "label_col": config["dataset"]["label_column"],
        "label_mapping": config["dataset"]["label_mapping"],
        "tokenizer": tokenizer,
        "image_processor": processor,
        "max_length": config["dataset"]["max_seq_length"],
    }
    split_dir = config["dataset"]["split_dir"]
    smoke_size = (
        int(config["experiment"]["smoke_size"])
        if config["experiment"].get("smoke_test", False)
        else None
    )
    dataset_kwargs["subset_size"] = smoke_size
    train_ds = FakedditDataset(
        csv_path=os.path.join(split_dir, "train.csv"), **dataset_kwargs
    )
    val_ds = FakedditDataset(
        csv_path=os.path.join(split_dir, "val.csv"), **dataset_kwargs
    )
    test_ds = FakedditDataset(
        csv_path=os.path.join(split_dir, "test.csv"), **dataset_kwargs
    )

    train_loader = create_loader(
        train_ds, int(config["training"]["batch_size"]), shuffle=True
    )
    val_loader = create_loader(val_ds, int(config["training"]["batch_size"]))
    test_loader = create_loader(test_ds, int(config["training"]["batch_size"]))

    model = ResearchMultimodalModel(config).to(device)
    parameter_counts = count_parameters(model)
    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = (
        torch.amp.GradScaler("cuda")
        if bool(config["training"].get("mixed_precision")) and device.type == "cuda"
        else None
    )

    experiment_name = config["experiment"]["name"]
    output_dir = Path(config["experiment"]["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    best_f1 = float("-inf")
    patience = 0
    history = []
    started = time.perf_counter()

    for epoch in range(int(config["training"]["epochs"])):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_metrics, *_ = evaluate(
            model, val_loader, criterion, device
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_macro_f1": val_metrics["f1_macro"],
            "validation_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        print(
            "epoch=%d val_f1=%.6f val_accuracy=%.6f"
            % (
                epoch + 1,
                val_metrics["f1_macro"],
                val_metrics["accuracy"],
            )
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience = 0
            save_checkpoint(
                model,
                optimizer,
                None,
                epoch,
                val_loss,
                str(checkpoint_path),
            )
        else:
            patience += 1
            if patience >= int(config["training"]["early_stopping_patience"]):
                break

    training_seconds = time.perf_counter() - started

    model_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    test_loss, test_metrics, labels, predictions, probabilities, latency = evaluate(
        model, test_loader, criterion, device
    )

    np_labels = np.asarray(labels)
    np_predictions = np.asarray(predictions)
    confusion = get_confusion_matrix(np_labels, np_predictions)

    metadata = {
        "experiment": experiment_name,
        "seed": seed,
        "device": str(device),
        "split_dir": str(split_dir),
        "config": str(config_path),
        "parameter_counts": parameter_counts,
        "training_seconds": training_seconds,
        "test_inference_seconds_per_sample": latency,
        "best_validation_macro_f1": best_f1,
        "test_metrics": test_metrics,
        "confusion_matrix": confusion.tolist(),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    write_run_metadata(output_dir / "test_metrics.json", test_metrics)
    np.savetxt(
        output_dir / "confusion_matrix.csv",
        confusion,
        delimiter=",",
        fmt="%d",
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metadata = run(args.config)
    print(metadata)
