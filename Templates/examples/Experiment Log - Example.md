---
id: 202606061000
title: "Fine-tune DistilBERT for churn classification"
created: 2026-06-06
updated: 2026-06-07
tags:
  - experiment
  - ml
type: experiment
status: active
project: Customer Churn Prediction
dataset: internal_tickets_v3
framework: transformers
---

# Fine-tune DistilBERT for churn classification

## Hypothesis

Using DistilBERT on customer support ticket text will outperform the TF-IDF + XGBoost baseline (AUC 0.82) because pre-trained embeddings capture semantic meaning that bag-of-words misses — especially for short, informal text with typos.

## Setup

### Data

- Dataset: `internal_tickets_v3` — 45K customer support tickets labeled with 30-day churn outcome
- Size: 36K train / 4.5K val / 4.5K test
- Split: stratified by churn label (12% positive class)
- Preprocessing: lowercase, truncate at 128 tokens, remove PII with regex

### Model

- Architecture: DistilBERT-base-uncased + linear classification head
- Pretrained: `distilbert-base-uncased` from HuggingFace
- Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| learning_rate | 2e-5 |
| batch_size | 32 |
| epochs | 5 |
| max_seq_length | 128 |
| warmup_steps | 500 |
| weight_decay | 0.01 |
| class_weight | {0: 1.0, 1: 7.5} |

### Environment

- Hardware: ml.g4dn.xlarge (1x T4 GPU, 16GB)
- Framework: transformers 4.40, PyTorch 2.3
- Random seed: 42

## Results

| Metric | TF-IDF + XGBoost | DistilBERT (epoch 3) | Delta |
|--------|------------------|---------------------|-------|
| AUC | 0.820 | 0.867 | +0.047 |
| F1 (churn) | 0.41 | 0.52 | +0.11 |
| Precision | 0.55 | 0.58 | +0.03 |
| Recall | 0.33 | 0.47 | +0.14 |

## Observations

Hypothesis confirmed — DistilBERT significantly outperforms the baseline, especially on recall (+14pp). The model correctly identifies churners who use indirect language ("thinking about alternatives", "not sure if this is worth it") that TF-IDF misses. Best performance at epoch 3; epochs 4-5 show signs of overfitting on val loss.

## Failure modes

- Still struggles with very short tickets (< 10 words) — not enough signal
- False positives on tickets mentioning competitor names in positive context ("better than X")
- Misclassifies angry-but-loyal customers who complain but never leave

## Next steps

- [ ] Try freezing first 4 layers and only fine-tuning top 2 (faster, less overfitting)
- [ ] Add structured features (tenure, ticket count) as a second input branch
- [ ] Calibrate probabilities with Platt scaling — raw logits are overconfident
- [ ] Test on Q1 2026 data to check temporal stability

## Artifacts

- Notebook: `notebooks/05_distilbert_churn.ipynb`
- Weights: `s3://ml-models/churn/distilbert_v1/`
- Logs: `wandb.ai/team/churn-bert-exp1`

## Connections

- Part of: [[Customer Churn Prediction Project]]
- Builds on: [[TF-IDF + XGBoost Churn Baseline]]
- Led to: [[Experiment - Hybrid DistilBERT + Tabular]]
