import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer, ViTImageProcessor
import yaml
import logging
import os
import numpy as np
from tqdm import tqdm

from src.data.dataset import FakedditDataset
from src.models.multimodal_model import MultimodalModel
from src.utils.reproducibility import set_seed
from src.utils.metrics import compute_metrics
from src.utils.checkpointing import save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training")
    for batch, labels in pbar:
        # Move to device
        text_inputs = {k: v.to(device) for k, v in batch['text'].items()}
        image_inputs = {k: v.to(device) for k, v in batch['image'].items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(text_inputs, image_inputs)
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader), all_preds, all_labels

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch, labels in loader:
            text_inputs = {k: v.to(device) for k, v in batch['text'].items()}
            image_inputs = {k: v.to(device) for k, v in batch['image'].items()}
            labels = labels.to(device)

            logits = model(text_inputs, image_inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    return total_loss / len(loader), all_preds, all_labels, all_probs

def run_training(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup Tokenizer and Processor
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['text_encoder']['checkpoint'])
    processor = ViTImageProcessor.from_pretrained(config['model']['image_encoder']['checkpoint'])

    dataset_kwargs = {
        "image_dir": config['dataset']['image_dir'],
        "text_col": config['dataset']['text_column'],
        "image_col": config['dataset']['image_column'],
        "label_col": config['dataset']['label_column'],
        "label_mapping": config['dataset']['label_mapping'],
        "tokenizer": tokenizer,
        "image_processor": processor,
        "max_length": config['dataset']['max_seq_length'],
    }
    split_dir = config['dataset'].get("split_dir")
    split_paths = {
        name: os.path.join(split_dir, f"{name}.csv")
        for name in ("train", "val", "test")
    } if split_dir else {}

    if split_paths and all(os.path.isfile(path) for path in split_paths.values()):
        train_ds = FakedditDataset(csv_path=split_paths["train"], **dataset_kwargs)
        val_ds = FakedditDataset(csv_path=split_paths["val"], **dataset_kwargs)
        test_ds = FakedditDataset(csv_path=split_paths["test"], **dataset_kwargs)
    else:
        full_dataset = FakedditDataset(
            csv_path=config['dataset']['csv_path'],
            subset_size=config['dataset'].get('smoke_test_size'),
            **dataset_kwargs
        )
        train_size = int(config['dataset']['train_split'] * len(full_dataset))
        val_size = int(config['dataset']['val_split'] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_ds, val_ds, test_ds = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config['training']['seed'])
        )

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'])

    # Model, Optimizer, Criterion
    model = MultimodalModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] and device.type == 'cuda' else None

    best_val_loss = float('inf')
    patience_counter = 0

    output_dir = "outputs/" + config['model']['fusion_method']
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")

        train_loss, train_preds, train_labels = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        val_loss, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )

        # Evaluation
        metrics = compute_metrics(val_labels, val_preds, np.array(val_probs))
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {metrics['f1_macro']:.4f}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, None, epoch, val_loss, f"{output_dir}/best_model.pt")
            patience_counter = 0
            logger.info("Best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info("Early stopping triggered.")
            break

    return model, test_loader, device

if __name__ == "__main__":
    run_training("configs/baseline.yaml")
