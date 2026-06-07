---
id: 202606051030
title: "Attention Is All You Need"
created: 2026-06-05
updated: 2026-06-07
tags:
  - paper
  - ml
type: paper
status: evergreen
authors: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
year: 2017
venue: NeurIPS
url: https://arxiv.org/abs/1706.03762
---

# Attention Is All You Need

## TL;DR

Introduces the Transformer architecture — a sequence-to-sequence model based entirely on self-attention, eliminating recurrence and convolutions while achieving state-of-the-art translation quality.

## Problem

RNNs process sequences sequentially (can't parallelize), and attention in existing models was used only as an add-on to recurrent layers. Training time scaled poorly with sequence length.

## Method

- Replace RNN encoder-decoder with stacked self-attention + feed-forward layers
- Multi-head attention: run attention in parallel across different learned subspaces
- Positional encoding (sinusoidal) to inject sequence order without recurrence
- Scaled dot-product attention: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- Residual connections + layer norm around every sub-layer

## Key results

| Benchmark | Metric | Result | vs. SOTA |
|-----------|--------|--------|----------|
| WMT 2014 EN-DE | BLEU | 28.4 | +2.0 over previous best |
| WMT 2014 EN-FR | BLEU | 41.0 | new SOTA, 1/4 training cost |

## Strengths

- Fully parallelizable → much faster training than RNNs
- Captures long-range dependencies in O(1) layers (vs. O(n) for RNNs)
- Multi-head attention learns diverse relationship types simultaneously
- Simple architecture, surprisingly few components

## Weaknesses / Limitations

- O(n²) memory and compute in sequence length (quadratic attention)
- No inherent notion of position — relies on positional encodings which may not generalize beyond training lengths
- Requires large data to train effectively (doesn't shine on small datasets)

## Key equations

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## My takeaways

- The key insight isn't "attention" — it's that you can build a powerful sequence model from a simple repeated block (attention + FFN + residuals). Modularity wins.
- The positional encoding feels like a hack. Later work (RoPE, ALiBi) addresses this better.
- This paper is more about architecture simplicity than about attention being novel.

## Connections

- Builds on: [[Bahdanau Attention]], [[Sequence to Sequence Learning]]
- Compared to: [[LSTM]], [[ConvS2S]]
- Useful for: [[BERT]], [[GPT]], [[Vision Transformer]]
- Concepts used: [[Self-Attention]], [[Positional Encoding]], [[Layer Normalization]]

## Citation

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  booktitle={NeurIPS},
  year={2017}
}
```
