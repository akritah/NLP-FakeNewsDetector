# Experiment protocol

## Fixed data

Run `src/data/prepare_dataset.py` first. Use the resulting
`data/splits/train.csv`, `val.csv`, and `test.csv` for comparisons. The split
manifest records invalid rows, duplicates, class distributions, and ID
overlap.

Do not mix results from the random smoke-test fallback with fixed-split
results. A run that uses the fallback must be labeled as a smoke test.

## Required experiment tracks

| Track | Status/command |
|---|---|
| Text-only baseline | `python src/training/train.py` with `fusion_method: text_only` |
| Image-only baseline | `python src/training/train.py` with `fusion_method: image_only` |
| Concatenation baseline | `python src/training/train.py` with `fusion_method: concat` |
| Cross-attention baseline | `python src/training/train.py` with `fusion_method: cross_attention` |
| Original replication | `python src/training/train_replication.py` |
| Proposed VLM analysis | Use `VLMAnalyzer` with a configured checkpoint |
| Integrated system | Call the replication classifier and `integrate_prediction` |
| Counterfactual evaluation | Supply controlled image conditions to `evaluate_counterfactuals` |

## Metrics

Report accuracy, macro precision, macro recall, macro F1, ROC-AUC, confusion
matrix, and class-wise metrics. Also record model checkpoint, data split,
seed, parameter count, hardware, and inference/training time where available.

Experiments that have not actually run must be reported as `NOT RUN`. VLM
captioning, consistency, and image-text similarity must not be described as
factual verification without independently retrieved and verified evidence.

## Counterfactual rules

For each fixed sample, evaluate the original image and any available
mismatched, semantically similar, or contradictory image. Keep the original
label for every condition. Report accuracy, confidence change, and prediction
change rate by condition. A swapped image is an intervention, not a relabeling
operation.
