#!/usr/bin/env python3
"""Prepare PubMed/replay JSONL or Hugging Face streaming data for CPT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomedical_cpt.cleaning import (  # noqa: E402
    deduplicate_records,
    filter_quality,
    mix_domain_and_replay,
    split_records,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def from_hf(dataset_name: str, config: str | None, split: str, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets with `pip install -e \".[data]\"` for HF streaming") from exc
    stream = load_dataset(dataset_name, config, split=split, streaming=True)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(stream):
        if limit is not None and index >= limit:
            break
        if not isinstance(raw, dict):
            continue
        text = raw.get("text") or raw.get("abstract") or raw.get("abstractText")
        if isinstance(text, dict):
            text = text.get("text")
        if not text:
            title = raw.get("title") or ""
            abstract = raw.get("abstract") or raw.get("abstractText") or ""
            text = f"{title}\n{abstract}"
        record_id = raw.get("id") or raw.get("pmid") or raw.get("uid") or index
        rows.append({"id": str(record_id), "text": str(text)})
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-jsonl", type=Path)
    parser.add_argument("--replay-jsonl", type=Path)
    parser.add_argument("--pubmed-dataset")
    parser.add_argument("--pubmed-config")
    parser.add_argument("--replay-dataset")
    parser.add_argument("--replay-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-domain-records", type=int)
    parser.add_argument("--max-replay-records", type=int)
    parser.add_argument("--domain-ratio", type=float, default=0.8)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.domain_jsonl:
        domain_raw = read_jsonl(args.domain_jsonl)
    elif args.pubmed_dataset:
        domain_raw = from_hf(args.pubmed_dataset, args.pubmed_config, args.split, args.max_domain_records)
    else:
        parser.error("provide --domain-jsonl or --pubmed-dataset")
    if args.replay_jsonl:
        replay_raw = read_jsonl(args.replay_jsonl)
    elif args.replay_dataset:
        replay_raw = from_hf(args.replay_dataset, args.replay_config, args.split, args.max_replay_records)
    else:
        replay_raw = []

    domain, domain_reasons = filter_quality(domain_raw, min_chars=args.min_chars)
    replay, replay_reasons = filter_quality(replay_raw, min_chars=args.min_chars)
    domain, domain_dedup = deduplicate_records(domain)
    replay, replay_dedup = deduplicate_records(replay)
    mixed = mix_domain_and_replay(domain, replay, domain_ratio=args.domain_ratio, seed=1337)
    splits = split_records(mixed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in splits.items():
        hashes[name] = write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    metadata = {
        "data_contract_version": "1.0",
        "domain_input": str(args.domain_jsonl or args.pubmed_dataset),
        "replay_input": str(args.replay_jsonl or args.replay_dataset),
        "domain_ratio": args.domain_ratio,
        "raw_counts": {"domain": len(domain_raw), "replay": len(replay_raw)},
        "filtered_counts": {"domain": len(domain), "replay": len(replay)},
        "filter_reasons": {"domain": domain_reasons, "replay": replay_reasons},
        "dedup": {"domain": domain_dedup, "replay": replay_dedup},
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "file_sha256": hashes,
        "approx_characters": sum(len(str(row.get("text", ""))) for row in mixed),
        "source_counts": dict(Counter(str(row.get("source")) for row in mixed)),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

