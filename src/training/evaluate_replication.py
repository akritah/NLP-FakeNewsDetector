import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, ViTImageProcessor, DistilBertTokenizer
import yaml
import numpy as np
import pandas as pd
from src.data.dataset import FakedditDataset
from src.models.replication_model import ReplicationModel
from src.utils.metrics import compute_metrics
from src.utils.checkpointing import load_checkpoint

def run_final_evaluation(config_path, checkpoint_path, csv_path, image_dir):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(config['model']['text_encoder']['checkpoint'])
    processor = ViTImageProcessor.from_pretrained(config['model']['image_encoder']['checkpoint'])

    class ReplicationDataset(FakedditDataset):
        def __getitem__(self, idx):
            data, label = super().__getitem__(idx)
            row = self.df.iloc[idx]
            data['raw_text'] = str(row[self.text_col])
            return data, label

    dataset = ReplicationDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        text_col=config['dataset']['text_column'],
        image_col=config['dataset']['image_column'],
        label_col=config['dataset']['label_column'],
        label_mapping=config['dataset']['label_mapping'],
        tokenizer=tokenizer,
        image_processor=processor,
        max_length=config['dataset']['max_seq_length']
    )

    loader = DataLoader(dataset, batch_size=32)
    model = ReplicationModel(config).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch, labels in loader:
            text_inputs = {k: v.to(device) for k, v in batch['text'].items()}
            image_inputs = {k: v.to(device) for k, v in batch['image'].items()}
            raw_text = batch['raw_text']

            logits, _, _, _ = model(text_inputs, image_inputs, raw_text)

            all_labels.extend(labels.numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return metrics

if __name__ == "__main__":
    # Example usage
    res = run_final_evaluation(
        "configs/replication.yaml",
        "outputs/replication/best_replication_model.pt",
        "data/raw/fakeddit_verified.csv",
        "data/raw/images/"
    )
    print(f"Final Replication Results: {res}")
