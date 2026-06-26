"""Fine-tune a small causal language model on prepared X-post JSONL data."""

from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def train(dataset_path: str, output_dir: str, model_name: str, epochs: float, block_size: int) -> None:
    """Train and save a text-generation model."""
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=block_size)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an X-style AI agent")
    parser.add_argument("--dataset", default="data/train.jsonl", help="Prepared training JSONL")
    parser.add_argument("--output-dir", default="models/x-agent", help="Where to save the model")
    parser.add_argument("--model-name", default="distilgpt2", help="Base Hugging Face causal LM")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs")
    parser.add_argument("--block-size", type=int, default=128, help="Maximum token length")
    args = parser.parse_args()

    train(args.dataset, args.output_dir, args.model_name, args.epochs, args.block_size)


if __name__ == "__main__":
    main()
