# Repository Audit

## Scope

This audit covers the source code, configurations, tests, notebook, documented
results, and available artifacts in the repository. It distinguishes the
original paper-replication track from the proposed vision-language-model (VLM)
extension.

## Executive summary

The repository already contains a working-looking baseline architecture built
from RoBERTa, ViT, feature fusion, and an MLP classifier. It also contains a
VAE-based replication model and an emotion-feature branch. However, the
replication path needs correction before it can be treated as reproducible:

- the replication-specific dataset subclass is required to provide raw text to
  the emotion branch and this requirement was implicit rather than part of the
  shared dataset contract;
- the baseline and replication trainers currently create random splits from the
  full CSV instead of consuming the audited fixed split files when those files
  exist;
- the two dataset adapters have different missing-image and emotion-vector
  behavior;
- the standalone feature-rich adapter is not used by the main training path;
- no VLM, counterfactual, evidence-retrieval, or explanation module existed;
- existing result tables are documented as completed experiments, but the
  execution provenance and seed aggregation are not represented in code.

No saved checkpoints, raw datasets, or result artifacts are deleted by this
work.

## Component inventory

| Component | File(s) | Status | Track | Findings |
|---|---|---|---|---|
| Image download | `image_downloader.py` | IMPLEMENTED | Data preparation | Downloads images with retries and skips existing files. It is an acquisition utility, not a verification step. |
| Verified CSV creation | `src/data/verify_dataset.py` | IMPLEMENTED BUT NEEDS CORRECTION | Data preparation | Filters existing non-empty images, but does not verify image decodability, text, labels, or duplicates. The later audit script covers those checks. |
| Dataset audit and fixed splits | `src/data/prepare_dataset.py` | IMPLEMENTED | Data preparation | Checks required columns, image integrity, duplicates, class presence, stratified splits, and split ID overlap. |
| Main dataset adapter | `src/data/dataset.py` | IMPLEMENTED BUT NEEDS CORRECTION | Shared | Correctly tokenizes and processes pairs, but validates only columns/values and raises at image load time. It does not expose a fixed-split factory. |
| Alternative dataset adapter | `src/data/fakeddit_dataset.py` | PARTIALLY IMPLEMENTED | Replication/support | Adds raw text, DistilBERT and NRC vectors, but silently substitutes zero images and uses a separate 10-dimensional NRC convention. It is not used by the main trainers. |
| RoBERTa encoder | `src/models/text_encoder.py` | IMPLEMENTED | Both | Provides CLS and masked mean pooling and optional freezing. |
| ViT encoder | `src/models/image_encoder.py` | IMPLEMENTED | Both | Provides pooled features and patch sequences. |
| Fusion | `src/models/fusion.py` | IMPLEMENTED BUT NEEDS CORRECTION | Baseline/replication | Supports concatenation and bidirectional attention. The replication model uses concatenation. Attention mask handling and API should be tested against real batches. |
| Baseline wrapper | `src/models/multimodal_model.py` | IMPLEMENTED BUT NEEDS CORRECTION | Original baseline | Supports text-only, image-only, concat, and cross-attention. It contains an unreachable duplicate branch for cross-attention and assumes encoder dimensions from checkpoints. |
| MLP classifier | `src/models/classifier.py` | IMPLEMENTED | Both | Generic MLP/FCNN classifier. |
| Emotion branch | `src/models/emotion_features.py` | PARTIALLY IMPLEMENTED | Original replication | DistilBERT is used as an emotion embedding extractor and concatenated with an eight-emotion lexical vector. The use of DistilBERT must be kept separate from fake-news classification. |
| Standalone DistilBERT helper | `src/features/distilbert_emotion.py` | IMPLEMENTED BUT NEEDS CORRECTION | Support | Produces a frozen embedding, but lazy global state and per-example calls are expensive. It is not a standalone fake-news classifier. |
| NRC helper | `src/features/emotion_lexicon.py` | PARTIALLY IMPLEMENTED | Support | Defines a ten-dimensional polarity-inclusive vector and silently returns zeros when the lexicon file is absent. This differs from the eight-dimensional vector in `emotion_features.py`. |
| VAE | `src/models/vae.py` | IMPLEMENTED | Original replication | Implements encoder, decoder, reparameterization, reconstruction MSE, and KL loss. |
| Replication wrapper | `src/models/replication_model.py` | IMPLEMENTED BUT NEEDS CORRECTION | Original replication | Correctly connects semantic, emotion, visual, concatenation, VAE, and classifier branches, but depends on replication raw-text handling and hard-codes emotion dimensions. |
| Baseline training | `src/training/train.py` | IMPLEMENTED BUT NEEDS CORRECTION | Original baseline | Includes AdamW, early stopping, checkpointing, metrics, and AMP. It currently uses `random_split` rather than the audited fixed splits. |
| Replication training | `src/training/train_replication.py` | IMPLEMENTED BUT NEEDS CORRECTION | Original replication | Adds classification plus VAE losses. It recomputes features for the VAE loss and relies on a local dataset subclass to add raw text. |
| Baseline evaluation | `src/training/evaluate.py` | IMPLEMENTED BUT NEEDS CORRECTION | Evaluation | Computes metrics and confusion matrices. It can use the fixed test CSV, but training is not guaranteed to have used the corresponding fixed train/validation CSVs. |
| Replication evaluation | `src/training/evaluate_replication.py` | IMPLEMENTED BUT NEEDS CORRECTION | Evaluation | Runs the replication model and metrics, but checkpoint loading assumptions differ from the shared checkpoint helper. |
| Metrics | `src/utils/metrics.py` | IMPLEMENTED | Evaluation | Accuracy, macro metrics, per-class F1, ROC-AUC, and confusion matrix are available. |
| Checkpointing | `src/utils/checkpointing.py` | IMPLEMENTED BUT NEEDS CORRECTION | Shared | Saves a wrapper checkpoint, while replication evaluation directly calls `load_state_dict(torch.load(...))`; the formats are inconsistent. |
| Reproducibility | `src/utils/reproducibility.py` | IMPLEMENTED | Shared | Seeds Python, NumPy, PyTorch, CUDA, and cuDNN settings. |
| Four-variant notebook | `notebooks/colab_kaggle_four_variants.ipynb` | IMPLEMENTED | Experiment | Provides cached-feature text-only, image-only, concat, and cross-attention experiments and recorded outputs. |
| Baseline tests | `tests/test_baseline.py` | PARTIALLY IMPLEMENTED | Testing | Tests architectural shapes but instantiate online pretrained encoders, making them dependent on model availability and downloads. |
| VAE tests | `tests/test_vae.py` | IMPLEMENTED | Testing | Covers shapes, finite loss, and stochastic reparameterization. |
| Emotion test | `src/tests/test_emotion_features.py` | PARTIALLY IMPLEMENTED | Testing | Is an executable smoke script rather than a unittest/pytest test and requires downloaded models/data paths. |
| VLM analysis | Missing before extension | MISSING | Proposed extension | No image captioning, structured visual analysis, or claim-image assessment existed. |
| Consistency analysis | Missing before extension | MISSING | Proposed extension | No separate consistent/inconsistent/unrelated/uncertain output existed. |
| Counterfactual evaluation | Missing before extension | MISSING | Proposed extension | No image-mismatch protocol existed. |
| Explainability | Missing before extension | MISSING | Proposed extension | No structured explanation output existed. |
| Evidence retrieval | Missing before extension | MISSING | Proposed extension | No configured retrieval API or source-verification mechanism existed. |

## Original replication assessment

The logical replication architecture is present:

```text
RoBERTa semantic features
       + DistilBERT/NRC emotion features
       + ViT visual features
       -> concatenation
       -> VAE latent mean (128 dimensions)
       -> MLP classifier
```

The available code uses DistilBERT as a frozen feature extractor inside the
emotion branch, not as a standalone fake-news classifier. That use is kept
separate from the classifier. The repository does not contain enough evidence
to verify every detail of the cited paper's exact emotion implementation, so
the implementation should be described as a documented reproduction choice,
not as a proven exact paper match.

## Proposed VLM assessment

The proposed research problem was not implemented in the original repository.
In particular, text-image similarity alone must not be presented as factual
verification. The extension therefore keeps:

1. classifier prediction,
2. semantic consistency,
3. optional external evidence,
4. uncertainty,

as separate fields. An optional integration policy may expose these fields
together, but it does not arbitrarily overwrite the classifier probability.

## Data and result limitations

- The raw data and downloaded images are local artifacts; their provenance is
  not encoded in every result file.
- Existing result tables should not be reinterpreted as results from the new
  VLM extension.
- The repository does not provide multiple-seed mean/std reports.
- External evidence retrieval is not configured, so no factual verification
  claim can be made from the new VLM interfaces alone.
- Hardware and model-download availability determine whether heavyweight
  pretrained encoders can be executed.

## Planned corrections and additions

1. Use fixed audited CSV splits when available, while retaining a documented
   fallback for smoke tests.
2. Make replication raw-text flow explicit and preserve the emotion branch.
3. Add a consistent checkpoint-loading path.
4. Add optional, lazy VLM analysis with explicit dependency/model errors.
5. Add separate consistency, explanation, evidence, integration, and
   counterfactual modules.
6. Add lightweight unit tests that do not require downloading a VLM.
7. Document architecture, experiment protocol, and changes.
