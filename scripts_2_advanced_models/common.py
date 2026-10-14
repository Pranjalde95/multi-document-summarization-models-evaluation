
import os, json
from typing import List

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Optional dependencies
try:
    import nltk
    nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize
except Exception:
    sent_tokenize = None

try:
    import yake
except Exception:
    yake = None

# Dataset registry
DATASETS = {
    "cnn_dm": ("abisee/cnn_dailymail", None),
    "xsum": ("EdinburghNLP/xsum", None),
    "wikihow": ("sentence-transformers/wikihow", None),
    "wikisum": ("d0rj/wikisum", None),
    "newsroom": ("lil-lab/newsroom", None),
}

SOURCE_CANDIDATES = [
    "article", "document", "documents", "text", "body", "story", "content", "input", "src", "source"
]
TARGET_CANDIDATES = [
    "highlights", "summary", "summaries", "target", "title", "titles", "output", "tgt"
]

def _coerce_to_str(x):
    if isinstance(x, list):
        return "\n".join([_coerce_to_str(xx) for xx in x])
    if isinstance(x, dict):
        for k in ("text", "content", "article", "document"):
            if k in x:
                return _coerce_to_str(x[k])
        try:
            return json.dumps(x, ensure_ascii=False)[:5000]
        except Exception:
            return str(x)
    return str(x)

def pick_columns(example):
    cols = set(example.keys())
    src_col = None
    tgt_col = None
    for c in SOURCE_CANDIDATES:
        if c in cols:
            src_col = c
            break
    for c in TARGET_CANDIDATES:
        if c in cols:
            tgt_col = c
            break
    if src_col is None and "text" in cols and "summary" in cols:
        src_col = "text"
        tgt_col = "summary"
    if src_col is None:
        src_col = list(cols)[0]
    if tgt_col is None:
        rest = [c for c in cols if c != src_col]
        tgt_col = rest[0] if rest else src_col
    return src_col, tgt_col

def load_split(ds_name: str):
    hf_id, config = DATASETS[ds_name]
    if config is None:
        ds = load_dataset(hf_id)
    else:
        ds = load_dataset(hf_id, config)
    if "train" in ds and "validation" in ds and "test" in ds:
        return ds["train"], ds["validation"], ds["test"]
    elif "train" in ds and "test" in ds:
        # Keep the provided test set held out from checkpoint selection.
        parts = ds["train"].train_test_split(test_size=0.01, seed=42)
        return parts["train"], parts["test"], ds["test"]
    elif "train" in ds and len(ds) == 1:
        # For train-only datasets, make independent, reproducible splits.
        parts = ds["train"].train_test_split(test_size=0.02, seed=42)
        held_out = parts["test"].train_test_split(test_size=0.5, seed=42)
        return parts["train"], held_out["train"], held_out["test"]
    else:
        raise ValueError(
            f"Dataset {ds_name} needs independent train/validation/test splits; got {list(ds.keys())}"
        )

def append_knowledge(texts: List[str], num_keywords: int = 10) -> List[str]:
    out = []
    if yake is None:
        return texts
    kw_extractor = yake.KeywordExtractor(top=num_keywords)
    for t in texts:
        try:
            kws = kw_extractor.extract_keywords(t)
            kws_sorted = [k for k, score in sorted(kws, key=lambda x: x[1])]
            augmented = t + "\n\n[KEYWORDS] " + ", ".join(kws_sorted[:num_keywords])
            out.append(augmented)
        except Exception:
            out.append(t)
    return out

def hierarchical_preprocess(text: str, max_sentences: int = 30) -> str:
    if sent_tokenize is None:
        return text
    sents = sent_tokenize(text)
    return " ".join(sents[:max_sentences])

def build_preprocess_fn(tokenizer, max_input_length=1024, max_target_length=128, knowledge=False, hierarchical=False):
    def preprocess(batch):
        src_col, tgt_col = pick_columns(batch)
        sources = [ _coerce_to_str(s) for s in batch[src_col] ]
        targets = [ _coerce_to_str(t) for t in batch[tgt_col] ]

        if knowledge:
            sources = append_knowledge(sources)
        if hierarchical:
            sources = [hierarchical_preprocess(s) for s in sources]

        model_inputs = tokenizer(sources, max_length=max_input_length, truncation=True)
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess

def compute_metrics_builder(tokenizer):
    """Score generated token IDs; mask ignored target tokens before decoding."""
    rouge = evaluate.load("rouge")
    try:
        bertscore = evaluate.load("bertscore")
    except Exception:
        bertscore = None

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        pad_id = tokenizer.pad_token_id
        decoded_preds = tokenizer.batch_decode(np.where(preds != -100, preds, pad_id), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(np.where(labels != -100, labels, pad_id), skip_special_tokens=True)
        results = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        out = {
            "rouge1": float(results["rouge1"]),
            "rouge2": float(results["rouge2"]),
            "rougeL": float(results["rougeL"]),
        }
        # BERTScore is expensive and may fail on restricted hardware.
        out["bertscore_f1"] = None
        if bertscore is not None:
            try:
                b = bertscore.compute(
                    predictions=decoded_preds, references=decoded_labels, lang="en"
                )
                out["bertscore_f1"] = float(np.mean(b["f1"]))
            except Exception:
                pass
        return out

    return compute_metrics

def train_eval(
    ds_key: str,
    model_name: str,
    output_dir: str,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    bf16: bool = True,
    knowledge: bool = False,
    hierarchical: bool = False,
    max_input_length: int = 1024,
    max_target_length: int = 128,
    gen_max_new_tokens: int = 128,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.03,
    logging_steps: int = 100
):
    train_ds, val_ds, test_ds = load_split(ds_key)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    preprocess_fn = build_preprocess_fn(
        tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        knowledge=knowledge,
        hierarchical=hierarchical
    )
    train_proc = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    val_proc = val_ds.map(preprocess_fn, batched=True, remove_columns=val_ds.column_names)
    test_proc = test_ds.map(preprocess_fn, batched=True, remove_columns=test_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=logging_steps,
        save_steps=logging_steps,
        save_total_limit=1,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        bf16=bf16,
        logging_steps=logging_steps,
        predict_with_generate=True,
        generation_max_length=gen_max_new_tokens,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_proc,
        eval_dataset=val_proc,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=test_proc)
    scores = {
        "rouge1": float(metrics.get("eval_rouge1", 0.0)),
        "rouge2": float(metrics.get("eval_rouge2", 0.0)),
        "rougeL": float(metrics.get("eval_rougeL", 0.0)),
        "bertscore_f1": metrics.get("eval_bertscore_f1"),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    return scores
