# Architecture

## Existing classification track

The original track has two related models:

1. Baseline variants: RoBERTa text features and ViT image features, evaluated
   as text-only, image-only, concatenation, or cross-attention models.
2. Replication model: RoBERTa semantic features, the documented DistilBERT/NRC
   emotion branch, ViT visual features, concatenation, a VAE with a
   128-dimensional latent space, and an MLP classifier.

The replication classifier remains independently callable and its prediction
must not be replaced by a VLM similarity score.

## Proposed VLM track

```text
Text + image
    |
    +--> existing replication classifier --> REAL/FAKE + probability
    |
    +--> optional VLM caption/visual analysis --> observations
    |
    +--> separate consistency model/result --> consistent/
          inconsistent/unrelated/uncertain
    |
    +--> optional evidence retriever --> source references
    |
    +--> structured explanation and uncertainty report
```

`src/models/proposed/vlm_analyzer.py` provides a lazy BLIP captioning adapter.
It is optional and raises an explicit error if its dependency or checkpoint is
unavailable. Captioning describes visible content; it is not factual
verification.

`src/models/proposed/consistency.py` keeps semantic consistency separate from
the REAL/FAKE label. The default result is `uncertain` unless a consistency
component supplies a label.

`src/models/proposed/evidence.py` defines an evidence interface and an explicit
disabled implementation. No URLs or credibility scores are fabricated.

`src/models/proposed/integration.py` combines outputs into a structured object
without changing the original classifier probability.

`src/models/proposed/counterfactual.py` evaluates original, mismatched, similar,
and contradictory image conditions while retaining the original ground-truth
label. It measures prediction changes rather than automatically changing
labels.
