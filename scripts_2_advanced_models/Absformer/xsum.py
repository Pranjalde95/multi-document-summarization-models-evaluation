# Auto-generated: absformer on xsum
import os
from common import train_eval

if __name__ == "__main__":
    model_name = "google/pegasus-large"
    out_dir = os.path.join("absformer", "xsum_outputs")
    scores = train_eval(
        ds_key="xsum",
        model_name=model_name,
        output_dir=out_dir,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        bf16=True,
        knowledge=False,
        hierarchical=False,
        max_input_length=1024,
        max_target_length=128,
        gen_max_new_tokens=128,
        learning_rate=3e-5,
        weight_decay=0.0,
        warmup_ratio=0.03,
        logging_steps=100,
    )
    print("Saved scores to:", os.path.join(out_dir, "scores.json"))
    print(scores)
