# Research Upgrade Roadmap

## Current evidence

- The fixed splits contain 2,080 training, 445 validation, and 447 test rows.
- The train/validation/test sample IDs have zero overlap.
- The original audited data has no missing text, missing images, or duplicate IDs.
- Four existing frozen-feature variants have been reported in
  `results/four_variants/comparison.csv`.
- The cross-attention variant reported 85.68% accuracy and 85.01% macro-F1;
  concat reported 85.01% accuracy and 84.63% macro-F1.
- This machine is CPU-only. It has no CUDA device.

These results are an existing Fakeddit experiment, not evidence of outperforming
the PolitiFact, GossipCop, or IFND benchmarks. The datasets and protocols are
not directly interchangeable.

## Priority 1: Remove training-design weaknesses

1. Keep all comparable experiments on the audited fixed splits.
2. Select checkpoints using validation macro-F1 rather than validation loss.
3. Make encoder freezing configurable per experiment.
4. Seed every random source and record the seed.
5. Report parameter count, training time, inference latency, and validation
   macro-F1 for each run.
6. Run multiple seeds only where runtime is practical.

## Priority 2: Controlled ablation

Run and compare separate tracks:

| Track | Purpose |
|---|---|
| RoBERTa text-only | Establish text baseline |
| ViT image-only | Establish visual baseline |
| RoBERTa + ViT concat | Measure straightforward multimodal gain |
| RoBERTa + ViT cross-attention | Measure interaction gain |
| Cross-attention + emotion | Test whether emotion features add held-out value |
| CLIP image/visual representation | Compare aligned vision-language features to ViT |
| Learned cross-modal consistency | Test whether learned consistency improves classification |
| Hyperparameter-optimized best model | Controlled validation-only selection |

Emotion and cross-modal consistency features must pass held-out ablation
tests. They must not be retained merely because the source paper includes
them.

## Priority 3: Representation improvements

Likely improvements, in descending implementation priority:

1. End-to-end frozen-feature baseline with stronger regularization and class
   weights.
2. CLIP image/text embeddings with cosine similarity and projection features.
3. Bidirectional cross-attention with residual connections and normalization.
4. Partial encoder fine-tuning, starting with the last text/vision block.
5. Mixed negative sampling or semantically controlled mismatched images.
6. Hyperparameter optimization restricted to train/validation data.

## Priority 4: VLM consistency research

Keep VLM reasoning separate from the classifier. Initially add:

- image description/observation,
- claim-image consistency label,
- support/contradiction/unrelated/uncertain estimate,
- confidence and uncertainty reasons.

Only learned and held-out-evaluated consistency features may be added to the
classification model. Captions and CLIP similarity are not factual evidence.

## Recommended first implementation

The first implementation should create a new `src/experiments/` module and
configuration set that:

- consumes fixed split CSVs,
- loads existing text/image encoders,
- supports text-only, image-only, concat, cross-attention, and CLIP modes,
- selects using validation macro-F1,
- writes machine-readable metrics and run metadata,
- does not modify existing baseline or replication scripts.

## Metrics and reporting

Each run should report:

- accuracy,
- precision macro,
- recall macro,
- macro-F1,
- ROC-AUC,
- confusion matrix,
- per-class precision/recall/F1,
- trainable and total parameter counts,
- training time,
- inference latency,
- seed,
- split manifest,
- checkpoint path.

No experiment should be described as improving on the published benchmark
unless it uses a comparable dataset and protocol. A direct Fakeddit-to-
PolitiFact/GossipCop/IFND accuracy claim is not valid.
