# Replication Specification: Multimodal Fake News Detection

**Paper:** *“Improving Fake News Detection Concatenating Multimodal Features with Transformers-Based Deep Learning Models”*
**Authors:** Mohini Chakarverti, Anurag Goswami, Rohit Kumar Kaliyar and Ashima Yadav.

## 1. Methodology Overview
The paper proposes a multimodal pipeline that extracts semantic, emotional, and visual features, compresses them using a VAE, and classifies them using a Fully Connected Neural Network (FCNN).

### Component Mapping
| Component | Paper Specification | Implementation Strategy | Status |
| :--- | :--- | :--- | :--- |
| **Semantic Text** | RoBERTa-base | `src/models/text_encoder.py` (Reuse) | $\checkmark$ |
| **Emotion Text** | DistilBERT + NRC Lexicon | `src/models/emotion_features.py` (New) | $\square$ |
| **Visuals** | Vision Transformer (ViT) | `src/models/image_encoder.py` (Reuse) | $\checkmark$ |
| **Compression** | Variational Autoencoder (VAE) | `src/models/vae.py` (New) | $\square$ |
| **Fusion** | Concatenation | `src/models/fusion.py` (Modify to remove Cross-Attn) | $\square$ |
| **Classifier** | FCNN / MLP | `src/models/classifier.py` (Reuse) | $\checkmark$ |

## 2. Ambiguities & Implementation Decisions

### A. The Emotion Branch
- **Ambiguity**: The paper mentions both DistilBERT and the NRC Emotion Lexicon. It is unclear if DistilBERT is used as a feature extractor for a separate emotion classifier or if the Lexicon is used for ground-truth labels.
- **Decision**: I will implement a dual-approach: a DistilBERT-based emotion embedding and a lexical-based emotion vector from the NRC Lexicon, concatenated into a single "emotion feature" vector.

### B. VAE Placement
- **Ambiguity**: The methodology text suggests VAE compresses visual features, but Algorithm 1 suggests VAE is applied *after* multimodal concatenation.
- **Decision**: I will implement the VAE as a modular component. We will test two variants:
    1. **Image-VAE**: Compresses only the ViT features.
    2. **Fused-VAE**: Compresses the final concatenated vector.
- These will be labeled as `VAE_Visual` and `VAE_Fused` respectively.

### C. Hyperparameters
- **Learning Rate**: 3e-5 (as per paper).
- **Weight Decay**: 1e-5.
- **Max Seq Length**: 128.
- **VAE Latent Dim**: 128.
- **Epochs**: The paper reports 500-5,000. This is extremely high for Transformers. I will implement an **Early Stopping** mechanism to find the optimal convergence point.

## 3. Dataset Requirements
The paper evaluates on:
- **GossipCop**
- **PolitiFact**
- **IFND**

**Current State**: We currently have **Fakeddit**.
**Strategy**: I will implement the architecture on Fakeddit first to prove the pipeline works, then create adapters for the three target datasets if they are provided.

## 4. Evaluation Metrics
The target metrics for replication are:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
