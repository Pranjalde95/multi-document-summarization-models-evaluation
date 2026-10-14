# Exploratory Advanced Summarization Approaches

This folder contains **25 model–dataset pipeline scripts** (five exploratory approaches × five datasets), plus shared preprocessing, training and evaluation code in [`common.py`](scripts_2_advanced_models/common.py). Each script configures a pretrained sequence-to-sequence model and calls the shared workflow.

These are **proxies or simplified experimental adaptations**, not complete independent implementations of the named research architectures.

## What the code implements

| Directory | Actual backbone | Implemented approach | Important distinction |
| --- | --- | --- | --- |
| `Absformer/` | `google/pegasus-large` | Standard PEGASUS summarization pipeline | Does **not** reproduce the full AbsFormer architecture or unsupervised method |
| `GNN (tg_multisum)/` | `facebook/bart-large-cnn` | BART baseline | No graph construction, graph encoder or TG-MultiSum implementation in the shared pipeline |
| `hierarchical_transformer/` | `allenai/PRIMERA` | Sentence-truncation preprocessing via `hierarchical=True` | Not a separate hierarchical transformer with learned hierarchical attention |
| `DCA/` | `t5-base` | T5 baseline | No multi-agent communication or fusion mechanism implemented |
| `External Knowledge (keyword_aware)/` | `google/pegasus-large` | YAKE-extracted keywords appended to input when available | Keyword augmentation, not an external knowledge graph or retrieval system |

## Datasets

Scripts are provided for CNN/DailyMail, WikiHow, XSum, WikiSum and Newsroom. Several are single-document datasets. The folder name and model labels do not establish that all configurations perform genuine multi-document summarization.

## Workflow represented in code

1. Load a dataset using Hugging Face Datasets.
2. Choose source and target text columns; tokenize inputs and target summaries.
3. Optionally prepend keyword information or truncate sentence lists.
4. Load a pretrained encoder–decoder model.
5. Configure a fine-tuning/evaluation pipeline.
6. Compute ROUGE-1/2/L and BERTScore F1 where generation and evaluation succeed.
7. Write `scores.json` to the model's output directory after a successful run.

## Installation and example invocation

From the repository root:

```bash
pip install -r requirements.txt
PYTHONPATH=scripts_2_advanced_models python scripts_2_advanced_models/Absformer/cnn_dm.py
PYTHONPATH=scripts_2_advanced_models python "scripts_2_advanced_models/External Knowledge (keyword_aware)/wikisum.py"
```

The `PYTHONPATH` setting lets scripts import the shared `common.py`. The commands illustrate how scripts are structured, not proof that they have been run successfully on the current dependency versions.

## GPU considerations

Script defaults use BF16 and assume compatible GPU hardware, with configurations suitable for an A100-class machine. Lower-memory GPUs may require reduced source lengths, smaller models or different precision settings. Model loading, dataset preparation and evaluation can be resource-intensive.

## Results and scope

These files define experimental methods. A script's existence or `scores.json` output path does **not** mean full-dataset fine-tuning and evaluation were completed. Report only performance results attributable to specific completed runs.

The proxy methods can serve as baselines or starting points for studying advanced architectures, but should not be presented as implementations of AbsFormer, TG-MultiSum, hierarchical-attention transformers or DCA.
