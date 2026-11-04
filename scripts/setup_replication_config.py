import yaml
import os

# Load current config
with open("configs/baseline.yaml", "r") as f:
    config = yaml.safe_load(f)

# Update for Replication
config['model']['emotion_encoder'] = {
    'checkpoint': 'distilbert-base-uncased',
    'freeze': True
}
config['model']['vae_latent_dim'] = 128

# Save as replication config
with open("configs/replication.yaml", "w") as f:
    yaml.dump(config, f)
