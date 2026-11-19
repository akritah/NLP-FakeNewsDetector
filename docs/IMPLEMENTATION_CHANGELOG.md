# Implementation changelog

## 2026-11-18

- Added a separate controlled research framework under
  `src/experiments/` and `configs/experiments/`.
- Added fixed-split audit, macro-F1 checkpoint selection, parameter/time
  tracking, CLIP visual support, cross-modal consistency features, and
  multi-seed result aggregation.
- Added explicit result reporting in `docs/RESULTS.md`.

## 2026-11-18

- Added the initial repository audit.
- Preserved the DistilBERT component as an emotion feature extractor rather
  than treating it as a standalone fake-news classifier.
- Configured baseline and replication training to prefer the audited fixed
  splits when all split files are present.
- Made replication evaluation use the shared checkpoint format.
- Added an optional lazy VLM analyzer with explicit model/dependency errors.
- Added separate consistency, evidence, explanation, integration, and
  counterfactual components.
- Added dependency-free tests for the proposed component contracts.
