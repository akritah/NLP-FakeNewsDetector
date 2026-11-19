# Experiment results

## Existing completed Fakeddit four-variant experiment

| Experiment | Accuracy | Macro precision | Macro recall | Macro F1 | ROC-AUC | Status |
|---|---:|---:|---:|---:|---:|---|
| Text-only | 83.8926% | 83.1349% | 83.2611% | 83.1962% | 91.5024% | COMPLETED |
| Image-only | 60.4027% | 30.2013% | 50.0000% | 37.6569% | 50.0000% | COMPLETED |
| ViT + RoBERTa concat | 85.0112% | 84.2966% | 85.4519% | 84.6344% | 92.7516% | COMPLETED |
| ViT + RoBERTa cross-attention | 85.6823% | 85.0767% | 84.9372% | 85.0050% | 92.4357% | COMPLETED |

Source: `results/four_variants/comparison.csv`.

These metrics come from an existing Fakeddit run and are not directly
comparable to the published PolitiFact, GossipCop, or IFND results.

## Published paper benchmark

| Dataset | Accuracy | F1 |
|---|---:|---:|
| PolitiFact | 82.8% | 83.1% |
| GossipCop | 72.7% | 65.3% |
| IFND | 76.1% | 84.2% |

These values are benchmark targets supplied by the project specification,
not values reproduced or newly measured in this repository.

## Controlled upgrade status

| Experiment | Status |
|---|---|
| Text-only RoBERTa | NOT RUN in the new reproducible runner |
| ViT image-only | NOT RUN in the new reproducible runner |
| RoBERTa + ViT concat | NOT RUN in the new reproducible runner |
| RoBERTa + ViT cross-attention | NOT RUN in the new reproducible runner |
| Cross-attention + emotion | NOT RUN / not yet implemented |
| CLIP visual representation | NOT RUN / implementation present for frozen experiments |
| Learned cross-modal consistency | NOT RUN / feature implementation present |
| Hyperparameter optimization | NOT RUN |
| Multi-seed aggregation | NOT RUN |

No improvement over the published benchmark is claimed.
