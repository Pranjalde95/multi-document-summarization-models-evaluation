# Next 5 Model Summary

This package contains **training & evaluation scripts** for 5 different summarization model types, each fine-tuned and evaluated on 5 datasets.  
All scripts are designed to run on the **entire dataset**, not samples.

---

## Models Implemented

1. **AbsFormer (Unsupervised MDS)** – (implemented via a strong abstractive proxy: PEGASUS)  
2. **Graph-based Neural Networks for MDS (TG-MultiSum)** – (proxy: BART; hook points to add graph encoder)  
3. **Hierarchical Transformer** – sentence-level truncation proxy using PRIMERA tokenizer/model  
4. **Deep Communicating Agents (DCA)** – (proxy: T5-base; hook points to add multi-agent fusion)  
5. **Knowledge-Aware Summarisation** – YAKE keywords appended to inputs

---

## Folder Structure

See subfolders with one scripts 2 per dataset plus a shared `common.py`.

---

## Installation

```bash
pip install transformers datasets evaluate sentence-transformers yake nltk
```

---

## Running a Script

```bash
python all_models_summary/absformer/cnn_dm.py
python all_models_summary/knowledge_aware/wikisum.py
# ...etc.
```
## System requirements 

**Hardware:** defaults are tuned for **Colab A100** (bf16).Preferably with high GPU for smoother run.

---

## Outputs

Each script writes an `*_outputs/` folder under its model subdir and saves a `scores.json` with ROUGE-1/2/L and BERTScore F1.

---

## Pipeline

Load dataset, split into sentences.

Build graphs/hierarchies/knowledge inputs.

Tokenize and prepare structures.

Load model with custom layers.

Fine-tune.

Generate summaries.

Evaluate with ROUGE & BERTScore.

---

## Hyperparameters

LR: 1e-5 to 3e-5

Batch: 1, grad acc: 12–32

Epochs: 2–4

Max source: 1024–2048 (cap sentences/nodes)

Max target: 128–200

Beams: 4, length penalty: 1–2

FP16/BF16,

```

## Issues faced

Across all ten models, the main challenges were hardware limits (GPU out-of-memory on large models and long inputs), slow generation times, 

Dataset-specific problems like inconsistent formatting and overly long documents requiring truncation. 

Some architectures (graph-based, hierarchical, multi-agent) added preprocessing complexity and increased memory use. 

Training stability issues also appeared, especially with high learning rates or unoptimized hyperparameters
 
While evaluation with ROUGE/BERTScore on large datasets was time-consuming.
   
                                                              #######

