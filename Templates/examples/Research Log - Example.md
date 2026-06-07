---
created: 2026-06-07
tags:
  - daily
  - ml
type: research-log
---

# 2026-06-07, Saturday

## Focus for today

- [x] Finish reading "Attention Is All You Need" and write paper note
- [x] Run DistilBERT churn experiment epoch 3-5
- [ ] Start literature review on calibration methods

## Reading / Papers

- Finished [[Attention Is All You Need]] — the key insight is architectural simplicity, not attention novelty. Multi-head attention = running parallel learned projections. The positional encoding is the weakest part; modern alternatives (RoPE, ALiBi) are better.
- Skimmed Guo et al. 2017 "On Calibration of Modern Neural Networks" — modern NNs are poorly calibrated. Temperature scaling is the simplest fix. Need to read properly → created fleeting note.

## Experiments

- DistilBERT churn model: epoch 3 gave best val AUC (0.867). Epochs 4-5 overfit. See [[Fine-tune DistilBERT for churn classification]].
- Quick test: freezing embeddings saved 40% training time but only lost 0.005 AUC. Worth it for iteration speed.

## Ideas & Observations

- Could combine the churn model's uncertainty (via MC Dropout) with the probability calibration — flag predictions that are both uncertain AND uncalibrated for human review.
- Wonder if [[Bias-Variance Tradeoff]] applies differently when fine-tuning pre-trained models vs. training from scratch. The pre-training acts as a strong prior (reduces variance?).

## Debugging notes

- Hit OOM on batch_size=64 with max_seq_length=256. Reduced to batch=32, seq=128. Didn't hurt AUC much (0.867 vs 0.871 with longer sequences). The tradeoff isn't worth the memory.
- `transformers` tokenizer was silently truncating without warning. Added `truncation=True, max_length=128` explicitly.

## Learnings

- Early stopping on val loss ≠ early stopping on val AUC. They diverge around epoch 3. Should always monitor the actual metric of interest.
- DistilBERT is 60% faster than BERT-base with only 3% quality drop for this task. Good default for resource-constrained fine-tuning.

## Tomorrow

- Write proper paper note for calibration paper
- Implement Platt scaling on the DistilBERT outputs
- Start hybrid model experiment (text embeddings + tabular features)
