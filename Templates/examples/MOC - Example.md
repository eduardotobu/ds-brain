---
id: 202606011200
title: "MOC: Natural Language Processing"
created: 2026-06-01
updated: 2026-06-07
tags:
  - moc
  - ml
type: moc
status:
---

# MOC: Natural Language Processing

## Overview

Everything related to processing and understanding text with ML. From classical methods (TF-IDF, n-grams) through modern transformer-based approaches. Focus on practical application to business text (support tickets, reviews, documents).

## Foundational concepts

- [[Tokenization]]
- [[Word Embeddings]]
- [[Self-Attention]]
- [[Positional Encoding]]
- [[Transfer Learning]]
- [[Language Modeling (Autoregressive vs. Masked)]]

## Models & Techniques

- [[BERT]] — bidirectional masked LM, great for classification/NER
- [[DistilBERT]] — 60% faster BERT with 97% performance
- [[GPT]] — autoregressive LM, generation tasks
- [[TF-IDF]] — classical baseline, still solid for small data
- [[Sentence-BERT]] — embeddings for semantic similarity

## Key papers

- [[Attention Is All You Need]] — Transformer architecture (Vaswani 2017)
- [[BERT Pre-training of Deep Bidirectional Transformers]] — (Devlin 2018)
- [[DistilBERT a Distilled Version of BERT]] — (Sanh 2019)

## Experiments & Projects

- [[Fine-tune DistilBERT for churn classification]]
- [[TF-IDF + XGBoost Churn Baseline]]
- [[Experiment - Hybrid DistilBERT + Tabular]]

## Open questions

- [[Does pre-training eliminate the bias-variance tradeoff in fine-tuning?]]
- When to use sentence embeddings + traditional ML vs. end-to-end fine-tuning?
- How to handle multilingual text (Spanish + English mix) in our customer data?

## Learning path

1. [[Tokenization]] → [[Word Embeddings]] → understand representation
2. [[Self-Attention]] → [[Positional Encoding]] → understand mechanism
3. [[Attention Is All You Need]] → understand architecture
4. [[BERT Pre-training of Deep Bidirectional Transformers]] → understand pre-training
5. [[Fine-tune DistilBERT for churn classification]] → hands-on application

## Narrative

NLP has converged on a single paradigm: pre-train a large transformer on massive text, then fine-tune on your specific task. The key insight is that language understanding is transferable — a model that learns to predict masked words also learns syntax, semantics, and world knowledge. For most practical applications, the choice isn't "which architecture" but "which pre-trained checkpoint and how to fine-tune it". Classical methods (TF-IDF) remain relevant as baselines and for resource-constrained settings.

## To-do

- [ ] Add notes on calibration for NLP classifiers
- [ ] Create concept note for [[Byte-Pair Encoding]]
- [ ] Explore retrieval-augmented generation (RAG) for our document QA use case
