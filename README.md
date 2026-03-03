# Multi-Document Summarization Benchmarks

This repository contains benchmarking and experimental evaluation of abstractive summarization models across five datasets.

## Benchmark Models
Located in:
scripts_1_benchmark_models/

Includes:
- BART
- PEGASUS
- T5
- PRIMERA
- BigBird-Pegasus

## Advanced Models
Located in:
scripts_2_advanced_models/

Includes:
- AbsFormer
- Graph-based models
- Hierarchical Transformer
- Deep Communicating Agents (DCA)
- Knowledge-aware summarization

## Datasets
- CNN/DailyMail
- WikiHow
- XSum
- WikiSum
- Newsroom

## Evaluation Metrics
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore (F1)

## Hardware
Tested on Colab A100. Reduce batch size for smaller GPUs.
