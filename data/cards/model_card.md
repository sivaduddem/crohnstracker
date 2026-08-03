# Model Card — EfficientNet-B0 Lesion Classifier

## Intended use
Research and educational decision support for capsule/colonoscopy stills. **Not** a medical device. Similarity retrieval does **not** imply diagnosis.

## Architecture
- Backbone: EfficientNet-B0 (`timm`)
- Head: linear classifier over 4 target labels
- Embeddings: penultimate features (backbone output before classifier)
- Retrieval: FAISS `IndexFlatIP` on L2-normalized embeddings (cosine similarity)
- Explainability: Grad-CAM on a late EfficientNet block

## Training data
Kvasir-Capsule frames remapped to `normal | erythema | erosion | ulcer`, split by video.

## Outputs
- Class probabilities + confidence
- Top-K visually similar historical cases
- Similarity analysis (NN distance, avg Top-K distance, novelty flag)
- Optional Grad-CAM overlay

## Limitations
- Domain shift across endoscope types and prep quality
- Class imbalance and rare findings
- Grad-CAM is explanatory, not a lesion segmentation
- Demo mode uses mock predictions — never clinical results

## Ethical considerations
Always display research disclaimers. Support manual review for novel/low-similarity cases. Do not replace physician expertise.
