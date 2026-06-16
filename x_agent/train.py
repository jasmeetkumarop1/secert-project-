"""Fine-tune a small causal language model on prepared X-post JSONL data."""

from __future__ import annotations

import argparse

from x_agent.parameters import AdaptiveParameterScheduler


def _count_jsonl_rows(dataset_path: str) -> int:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def train(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    epochs: float,
    block_size: int,
    *,
    auto_parameters: bool = False,
    parameter_state: str = "data/parameter_state.json",
) -> None:
    """Train and save a text-generation model."""
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    scheduler = AdaptiveParameterScheduler(parameter_state)
    dataset_size = _count_jsonl_rows(dataset_path)
    adaptive = scheduler.training_parameters(
        dataset_size=dataset_size,
        base_epochs=epochs,
        base_block_size=block_size,
    )
    if auto_parameters:
        epochs = adaptive.epochs
        block_size = adaptive.block_size
    else:
        adaptive = scheduler.training_parameters(
            dataset_size=dataset_size,
            base_epochs=epochs,
            base_block_size=block_size,
            max_epochs=epochs,
            max_block_size=block_size,
            max_gradient_accumulation_steps=4,
        )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=block_size)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=adaptive.gradient_accumulation_steps,
        learning_rate=adaptive.learning_rate,
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
    scheduler.record_training_run(adaptive)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an X-style AI agent")
    parser.add_argument("--dataset", default="data/train.jsonl", help="Prepared training JSONL")
    parser.add_argument("--output-dir", default="models/x-agent", help="Where to save the model")
    parser.add_argument("--model-name", default="distilgpt2", help="Base Hugging Face causal LM")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs")
    parser.add_argument("--block-size", type=int, default=128, help="Maximum token length")
    parser.add_argument("--auto-parameters", action="store_true", help="Automatically grow safe training parameters over time")
    parser.add_argument("--parameter-state", default="data/parameter_state.json", help="Adaptive parameter state path")
    args = parser.parse_args()

    train(
        args.dataset,
        args.output_dir,
        args.model_name,
        args.epochs,
        args.block_size,
        auto_parameters=args.auto_parameters,
        parameter_state=args.parameter_state,
    )


if __name__ == "__main__":
    main()
