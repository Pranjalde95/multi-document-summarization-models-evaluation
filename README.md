# Abstractive Summarization: Model Pipelines and Evaluation

This repository contains Python training and evaluation pipelines for pretrained abstractive summarization models across five datasets. It includes conventional transformer benchmarks and exploratory, pretrained-model **proxies** for advanced summarization ideas.

The scripts describe experimental configurations; the presence of a script is not evidence that a full training/evaluation run finished. See the model-specific READMEs for methodology. Only report quantitative scores alongside the runs and logs that produced them.

## Benchmark models

Source: [`scripts_1_benchmark _models/`](scripts_1_benchmark%20_models/) (the directory name currently contains a space).

- BART
- PEGASUS
- T5
- PRIMERA
- BigBird-PEGASUS

[Benchmark methods and configuration](README_benchmark_models.md)

## Exploratory advanced approaches

Source: [`scripts_2_advanced_models/`](scripts_2_advanced_models/)

- AbsFormer-inspired: PEGASUS proxy; not a complete AbsFormer architecture
- Graph-based summarization-inspired: BART baseline; no graph neural network implemented
- Hierarchical summarization-inspired: sentence truncation/preprocessing with PRIMERA; not a new hierarchical transformer
- Deep Communicating Agents-inspired: T5-base baseline; no communicating agents or fusion module implemented
- Keyword-aware summarization: YAKE keyword augmentation of model inputs

[Advanced-model implementation details](README_advanced_models.md)

## Datasets

- CNN/DailyMail — single-document summarization
- WikiHow — single-document summarization
- XSum — single-document summarization
- WikiSum — multi-document-oriented dataset/configuration
- Newsroom — primarily single-document in these scripts

Not every model–dataset configuration constitutes a multi-document experiment.

## Evaluation design

The scripts include workflows for ROUGE-1, ROUGE-2, ROUGE-L and BERTScore (F1). Results must be tied to reproducible runs and the precise model, dataset split and checkpoint used.

## Hardware

Configurations assume substantial GPU capacity, including BF16 settings suitable for compatible hardware such as an NVIDIA A100. Hardware settings are **configuration requirements**, not a claim that every script was tested or trained on an A100.

## Code and limitations

- Some scripts are computationally expensive; running one may need significant GPU memory and time.
- Benchmark scripts are templates for fine-tuning and evaluation, not verified published results.
- Original research architectures should not be inferred from the proxy directory names.
- Full execution and dataset availability have not been revalidated as part of the documentation cleanup.
