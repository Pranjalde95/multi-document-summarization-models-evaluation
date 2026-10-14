# Benchmark Summarization Pipelines (5 Models × 5 Datasets)

This directory provides **25 scripts**—one for each of five pretrained sequence-to-sequence model backbones and five datasets. The scripts define fine-tuning and evaluation procedures; they do **not** by themselves document completed full-dataset experiments.

## Pretrained model backbones

- BART (`facebook/bart-large-cnn`)
- PEGASUS (`google/pegasus-large`)
- T5 (`t5-large`)
- PRIMERA (`allenai/PRIMERA`)
- BigBird-PEGASUS (`google/bigbird-pegasus-large-arxiv`)

## Datasets and task type

- CNN/DailyMail — single-document summarization
- WikiHow — single-document summarization
- XSum — single-document summarization
- WikiSum — multi-document-oriented source handling
- Newsroom — predominantly single-document summarization in this repository

Dataset loading and split handling may require adjustment if upstream datasets change.

## Intended pipeline

1. Load and preprocess the dataset.
2. Tokenize source text and reference summaries.
3. Load a pretrained model and configure fine-tuning.
4. Generate summaries for evaluation.
5. Calculate ROUGE-1, ROUGE-2, ROUGE-L and BERTScore F1.
6. Save results to `./outputs/<model>_<dataset>/scores.json` following a successful run.

Scores and checkpoints are only valid evidence for an experiment if backed by that experiment's execution records.

## Installation

```bash
pip install -r requirements.txt
```

Example script path from the repository root (note the existing space in the folder name):

```bash
python "scripts_1_benchmark _models/Bart/cnn_dm.py"
```

## Hardware and configuration

The scripts contain BF16/gradient-accumulation settings intended for high-memory, BF16-compatible GPUs such as NVIDIA A100. These settings are *requirements or defaults*, not a report that training has been completed on an A100.

## Limitations

- Long-context models and full-dataset training can be expensive.
- Some older model and library interfaces may require updates.
- No complete set of run logs/checkpoints is provided here to substantiate full training across all 25 configurations.
- Code maintenance and syntax checks do not substitute for model training, evaluation or empirical verification.
