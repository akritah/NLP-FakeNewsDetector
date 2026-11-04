import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import os
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, ViTImageProcessor
from src.data.dataset import FakedditDataset
from src.utils.metrics import compute_metrics, get_confusion_matrix
from src.utils.checkpointing import load_checkpoint
from src.models.multimodal_model import MultimodalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch, labels in loader:
            text_inputs = {k: v.to(device) for k, v in batch['text'].items()}
            image_inputs = {k: v.to(device) for k, v in batch['image'].items()}
            labels = labels.to(device)

            logits = model(text_inputs, image_inputs)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, path, classes):
    cm = get_confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(path)
    plt.close()

def run_final_evaluation(config_path: str, checkpoint_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = MultimodalModel(config).to(device)
    load_checkpoint(checkpoint_path, model)

    split_dir = config["dataset"].get("split_dir")
    test_csv = os.path.join(split_dir, "test.csv") if split_dir else config["dataset"]["csv_path"]
    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["text_encoder"]["checkpoint"])
    processor = ViTImageProcessor.from_pretrained(config["model"]["image_encoder"]["checkpoint"])
    dataset = FakedditDataset(
        csv_path=test_csv,
        image_dir=config["dataset"]["image_dir"],
        text_col=config["dataset"]["text_column"],
        image_col=config["dataset"]["image_column"],
        label_col=config["dataset"]["label_column"],
        label_mapping=config["dataset"]["label_mapping"],
        tokenizer=tokenizer,
        image_processor=processor,
        max_length=config["dataset"]["max_seq_length"],
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"])
    logger.info("Running final evaluation on test set...")
    labels, predictions, probabilities = evaluate_model(model, loader, device)
    metrics = compute_metrics(labels, predictions, probabilities)
    output_dir = os.path.dirname(checkpoint_path) or "."
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)
    plot_confusion_matrix(
        labels, predictions, os.path.join(output_dir, "confusion_matrix.png"), ["REAL", "FAKE"]
    )
    results = {
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat()
    }

    return results

if __name__ == "__main__":
    # Example usage:
    # run_final_evaluation("configs/baseline.yaml", "outputs/concat/best_model.pt")
    pass
