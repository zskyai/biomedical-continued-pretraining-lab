#!/usr/bin/env python3
"""Run the deterministic offline data and metric smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomedical_cpt.cleaning import (  # noqa: E402
    deduplicate_records,
    filter_quality,
    mix_domain_and_replay,
    split_records,
)
from biomedical_cpt.contamination import contamination_report  # noqa: E402
from biomedical_cpt.metrics import pareto_point  # noqa: E402


def fixture_records() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    domain = [
        {"id": "pmid-1", "text": "A randomized biomedical cohort study measured inflammatory biomarkers and reported confidence intervals for treatment response."},
        {"id": "pmid-2", "text": "The pharmacokinetic analysis evaluated plasma concentration, half life, and adverse events across three dose groups."},
        {"id": "pmid-3", "text": "A systematic review compared sensitivity and specificity of a screening assay in older adult populations."},
        {"id": "pmid-duplicate", "text": "The pharmacokinetic analysis evaluated plasma concentration, half life, and adverse events across three dose groups."},
    ]
    replay = [
        {"id": "web-1", "text": "A clear technical explanation should state assumptions, compare alternatives, and report reproducible measurements."},
        {"id": "web-2", "text": "Distributed training systems monitor throughput, memory, checkpoint time, and recovery after worker failure."},
    ]
    return domain, replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/smoke")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    domain, replay = fixture_records()
    domain, domain_reasons = filter_quality(domain, min_chars=40)
    replay, replay_reasons = filter_quality(replay, min_chars=40)
    domain, domain_dedup = deduplicate_records(domain)
    replay, replay_dedup = deduplicate_records(replay)
    mixed = mix_domain_and_replay(domain, replay, domain_ratio=0.8, seed=1337, target_records=10)
    splits = split_records(mixed)
    contamination = contamination_report(
        ["inflammatory biomarkers treatment response"],
        [row["text"] for row in mixed],
        n=4,
    )
    metrics = {
        "status": "offline_smoke_only",
        "domain_after_filter": len(domain),
        "replay_after_filter": len(replay),
        "domain_filter_reasons": domain_reasons,
        "replay_filter_reasons": replay_reasons,
        "domain_dedup": domain_dedup,
        "replay_dedup": replay_dedup,
        "mixed_records": len(mixed),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "source_counts": {
            "domain": sum(row.get("source") == "domain" for row in mixed),
            "replay": sum(row.get("source") == "replay" for row in mixed),
        },
        "contamination": contamination,
        "illustrative_metric_formula": pareto_point(10.0, 8.0, 0.70, 0.68),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

