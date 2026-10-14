# This script is part of a simple, per-model-per-dataset training/evaluation suite.
# When run, this pipeline is configured to train/evaluate on dataset splits,
# and write `scores.json` with ROUGE-1/2/L and BERTScore F1 on success.
# Colab/A100-friendly defaults (bf16, small per-device batch + grad accumulation).

from transformers import (BartForConditionalGeneration, BartTokenizerFast,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
import os, json, numpy as np, torch, random
from datasets import load_dataset, DatasetDict
import evaluate


MODEL_NAME     = "facebook/bart-large-cnn"
MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 128
EPOCHS         = 1
TRAIN_BS       = 2
EVAL_BS        = 2
GRAD_ACCUM     = 8
LR             = 3e-5
BEAM           = 4
GEN_MAX_LEN    = 128
USE_BF16       = True

OUT_DIR = f"./outputs/bart_newsroom"
os.makedirs(OUT_DIR, exist_ok=True)

# Load dataset
raw = load_dataset('lil-lab/newsroom')

def to_map(ex): return {'source': ex['text'], 'target': ex['summary']}
data = raw.map(to_map, remove_columns=raw['train'].column_names)

tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize(batch):
    X = tokenizer(batch['source'], max_length=MAX_SOURCE_LEN, truncation=True)
    with tokenizer.as_target_tokenizer():
        Y = tokenizer(batch['target'], max_length=MAX_TARGET_LEN, truncation=True)
    X['labels'] = Y['input_ids']
    return X

tokenized = data.map(tokenize, batched=True, remove_columns=data['train'].column_names)
train_ds = tokenized['train']; val_ds = tokenized['validation']; test_ds = tokenized['test']

model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
rouge = evaluate.load("rouge")

def compute_metrics(pred):
    p, y = pred
    if isinstance(p, tuple): p = p[0]
    dec_p = tokenizer.batch_decode(p, skip_special_tokens=True)
    y = np.where(y != -100, y, tokenizer.pad_token_id)
    dec_y = tokenizer.batch_decode(y, skip_special_tokens=True)
    s = rouge.compute(predictions=dec_p, references=dec_y, use_stemmer=True)
    return {'rouge1': s['rouge1'], 'rouge2': s['rouge2'], 'rougeL': s['rougeL']}

args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR, evaluation_strategy="steps",
    per_device_train_batch_size=TRAIN_BS, per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM, num_train_epochs=EPOCHS,
    learning_rate=LR, predict_with_generate=True,
    generation_max_length=GEN_MAX_LEN, generation_num_beams=BEAM,
    logging_steps=50, eval_steps=2000, save_steps=2000, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_rougeL", greater_is_better=True,
    bf16=USE_BF16, report_to=["none"],
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collator, tokenizer=tokenizer, compute_metrics=compute_metrics
)

trainer.train()

# Final evaluation
bertscore = evaluate.load("bertscore")
pred = trainer.predict(test_ds, max_length=GEN_MAX_LEN, num_beams=BEAM)
dec_p = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
labels = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)
dec_y = tokenizer.batch_decode(labels, skip_special_tokens=True)
r = rouge.compute(predictions=dec_p, references=dec_y, use_stemmer=True)
bs = bertscore.compute(predictions=dec_p, references=dec_y, lang="en")
scores = dict(r); scores['bertscore_f1'] = float(np.mean(bs['f1']))
with open(os.path.join(OUT_DIR, "scores.json"), "w") as f: json.dump(scores, f, indent=2)
print(scores)
