# Summarization Benchmarks (5 Models × 5 Datasets)

This repo contains **25 standalone scripts** — one per **model–dataset** pair — to fine-tune and evaluate sequence-to-sequence summarization models on common datasets. Each script:

- Trains on the **entire training split**
- Evaluates on the **entire test split**
- Logs ROUGE during eval steps
- Saves final **ROUGE-1/2/L** and **BERTScore F1** to `scores.json`

## Models
- **BART** (`facebook/bart-large-cnn`)
- **PEGASUS** (`google/pegasus-large`)
- **T5** (`t5-large`, with `summarize:` prefix)
- **PRIMERA** (`allenai/PRIMERA` — LED-based long-context)
- **BigBird-PEGASUS** (`google/bigbird-pegasus-large-arxiv` — long-context)

## Datasets
- **CNN/DailyMail** (`abisee/cnn_dailymail` — SDS)
- **WikiHow** (`sentence-transformers/wikihow` — SDS, headline)
- **XSum** (`EdinburghNLP/xsum` — SDS, one-sentence summaries)
- **WikiSum** (`d0rj/wikisum` — MDS; we split 98/1/1 from the provided `train` and concat top-k sources)
- **Newsroom** (`lil-lab/newsroom` — Mixed; default SDS; optional MDS-style variant not included here for simplicity)

> **Hardware:** defaults are tuned for **Colab A100** (bf16).Preferably with high GPU for smoother run.

```
## installation
pip install -r requirements.txt
# Example: train/eval BART on CNN/DailyMail
python scripts/bart/cnn_dm.py
# Scores at: ./outputs/bart_cnn_dm/scores.json
```

All scripts follow the same pattern. Change `EPOCHS` to 3–5 for better results; 1 epoch is for smoke tests.

## Outputs
Each script writes to `./outputs/<model>_<dataset>/`:
- `scores.json` – final ROUGE-1/2/L and BERTScore F1

```

## Pipeline

Load dataset and tokenize.

Long-input handling for PRIMERA/BigBird-Pegasus.

Load pretrained model & tokenizer.

Fine-tune on training data.