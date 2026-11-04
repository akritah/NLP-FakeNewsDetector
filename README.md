# Multimodal Fake News Detection

Multimodal fake-news classification using paired Fakeddit titles and images. The project compares text-only, image-only, multimodal concatenation, and cross-attention models, and includes a paper-replication path with emotion features and a variational autoencoder.

## Project highlights

- Verified paired text-image dataset with preserved sample IDs.
- Reproducible, stratified train/validation/test splits.
- Duplicate and corrupt-image checks.
- RoBERTa contextual text encoder.
- Vision Transformer (`google/vit-base-patch16-224`) image encoder.
- Text-only and image-only baselines.
- Multimodal feature concatenation.
- Bidirectional text-image cross-attention.
- DistilBERT emotion embeddings.
- NRC emotion-lexicon features.
- 128-dimensional VAE feature compression for the replication architecture.
- MLP/FCNN binary classifier for REAL/FAKE prediction.
- Checkpointing, early stopping, seeded experiments, metrics, and confusion matrices.
- Colab/Kaggle notebook with cached frozen features for fast four-variant experiments.

## Repository layout

```text
FakeNewsDetector/
├── configs/
│   ├── baseline.yaml
│   └── replication.yaml
├── data/
│   ├── raw/
│   │   ├── fakeddit_verified.csv
│   │   └── images/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── manifest.json
│   └── README.md
├── notebooks/
│   └── colab_kaggle_four_variants.ipynb
├── results/
│   └── four_variants/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── image_downloader.py
├── requirements.txt
└── README.md
```

Generated checkpoints, cached features, confusion matrices, histories, and comparison tables are in [results/four_variants/](results/four_variants/).

## Dataset preparation

The project uses the verified Fakeddit subset. Each retained row contains:

```text
id, title, image_url, label
```

The `image_url` column in the verified file contains the local image filename, for example `awxhir.jpg`. Labels are mapped as:

```text
0 = REAL
1 = FAKE
```

The dataset audit and split generator is [prepare_dataset.py](src/data/prepare_dataset.py). Run it from the project root:

```powershell
$env:PYTHONPATH="C:\FakeNewsDetector"
python src\data\prepare_dataset.py
```

It validates titles, labels, image files, image integrity, duplicate IDs, duplicate text-image pairs, class balance, and split ID overlap. It writes the fixed splits to `data/splits/` and the audit manifest to `data/splits/manifest.json`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Architecture

### Baseline multimodal model

```text
Title → RoBERTa-base → 768-d text feature
Image → ViT-base/patch16-224 → 768-d image feature
                 ↓
Projection to 256 dimensions
                 ↓
Concat or cross-attention fusion
                 ↓
MLP classifier
                 ↓
REAL / FAKE
```

The baseline implementation is in [multimodal_model.py](src/models/multimodal_model.py). The fusion module is in [fusion.py](src/models/fusion.py).

### Paper-replication model

```text
RoBERTa semantic features
DistilBERT emotion features
NRC emotion features
ViT visual features
          ↓
Feature concatenation
          ↓
VAE compression to 128 dimensions
          ↓
MLP/FCNN classifier
          ↓
REAL / FAKE
```

The replication implementation is in [replication_model.py](src/models/replication_model.py), with the VAE in [vae.py](src/models/vae.py) and training in [train_replication.py](src/training/train_replication.py).

## Training

The baseline configuration is in [baseline.yaml](configs/baseline.yaml). It supports:

```text
fusion_method: concat
fusion_method: cross_attention
fusion_method: text_only
fusion_method: image_only
```

Run the baseline trainer:

```powershell
$env:PYTHONPATH="C:\FakeNewsDetector"
python src\training\train.py
```

Run the paper-replication trainer:

```powershell
$env:PYTHONPATH="C:\FakeNewsDetector"
python src\training\train_replication.py
```

The training pipeline includes seeded data handling, AdamW optimization, validation monitoring, early stopping, and best-checkpoint saving.

## Four-variant Colab/Kaggle experiment

Open [colab_kaggle_four_variants.ipynb](notebooks/colab_kaggle_four_variants.ipynb) in Google Colab or Kaggle.

The notebook:

1. Installs the required packages.
2. Loads the verified CSV and local images.
3. Checks paired samples and labels.
4. Creates fixed stratified splits.
5. Extracts and caches frozen RoBERTa and ViT features.
6. Trains text-only, image-only, concat, and cross-attention classifiers.
7. Applies early stopping.
8. Calculates Accuracy, Macro-Precision, Macro-Recall, Macro-F1, and ROC-AUC.
9. Saves checkpoints, training histories, confusion matrices, CSV results, and Markdown results.

For a quick demonstration, keep:

```python
FAST_MODE = True
```

For the full verified subset, set:

```python
FAST_MODE = False
```

## Results

The completed four-variant comparison is available in [comparison.csv](results/four_variants/comparison.csv) and [comparison.md](results/four_variants/comparison.md).

| Variant | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Text-only | 0.8389 | 0.8313 | 0.8326 | 0.8320 | 0.9150 |
| Image-only | 0.6040 | 0.3020 | 0.5000 | 0.3766 | 0.5000 |
| Multimodal-Concat | 0.8501 | 0.8430 | 0.8545 | 0.8463 | 0.9275 |
| Multimodal-Cross-Attention | **0.8568** | **0.8508** | 0.8494 | **0.8501** | 0.9244 |

The cross-attention model achieved the strongest overall result, while concatenation also improved over the text-only baseline.

## Evaluation outputs

The evaluation pipeline in [evaluate.py](src/training/evaluate.py) supports:

- Accuracy
- Macro-precision
- Macro-recall
- Macro-F1
- ROC-AUC
- Confusion-matrix plots
- Test metric CSV output

The result directory contains per-variant:

- `.pt` checkpoints
- Training-history CSV files
- Confusion-matrix PNG files
- Cached encoder features
- Final comparison CSV and Markdown table

## Validation

Run the architecture and VAE tests:

```powershell
$env:PYTHONPATH="C:\FakeNewsDetector"
python tests\test_baseline.py
python tests\test_vae.py
```

The test suite verifies encoder dimensions, ViT patch extraction, fusion dimensions, classifier output dimensions, VAE shapes, VAE loss, and stochastic reparameterization.

## Main implementation files

- [Text encoder](src/models/text_encoder.py)
- [Image encoder](src/models/image_encoder.py)
- [Fusion module](src/models/fusion.py)
- [Baseline model](src/models/multimodal_model.py)
- [Replication model](src/models/replication_model.py)
- [VAE](src/models/vae.py)
- [Emotion features](src/models/emotion_features.py)
- [Dataset audit and split generation](src/data/prepare_dataset.py)
- [Verified dataset generation](src/data/verify_dataset.py)
- [Training](src/training/train.py)
- [Replication training](src/training/train_replication.py)
- [Evaluation](src/training/evaluate.py)
- [Metrics](src/utils/metrics.py)
