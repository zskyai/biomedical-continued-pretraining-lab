#!/usr/bin/env python3
"""Small reference CPT launcher; full experiments must be run with recorded hardware/config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_rows(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(str(json.loads(line).get("text", "")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_texts = read_rows(args.train_file)
    valid_texts = read_rows(args.validation_file)
    summary = {
        "model": args.model,
        "train_documents": len(train_texts),
        "validation_documents": len(valid_texts),
        "max_seq_len": args.max_seq_len,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "status": "dry_run" if args.dry_run else "launcher_validation_only",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "launch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install `.[train]` before running CPT") from exc
    # Keep the default path intentionally conservative: users should inspect the data and hardware
    # before launching. Tokenization is done in-memory only for small validation runs.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    encoded = tokenizer(
        train_texts,
        truncation=True,
        max_length=args.max_seq_len,
        padding="max_length",
    )
    valid_encoded = tokenizer(
        valid_texts,
        truncation=True,
        max_length=args.max_seq_len,
        padding="max_length",
    )

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, values: dict[str, list[list[int]]]):
            self.values = values

        def __len__(self) -> int:
            return len(self.values["input_ids"])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            item = {key: torch.tensor(value[index]) for key, value in self.values.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenDataset(encoded),
        eval_dataset=TokenDataset(valid_encoded),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()

