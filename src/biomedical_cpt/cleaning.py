"""Deterministic text quality, deduplication, splitting, and mixture helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    reason: str
    characters: int
    alpha_ratio: float
    url_ratio: float


def normalize_text(text: str) -> str:
    """Apply Unicode normalization and collapse whitespace without changing content order."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ")
    return _SPACE_RE.sub(" ", text).strip()


def quality_report(
    text: str,
    *,
    min_chars: int = 80,
    max_chars: int = 200_000,
    min_alpha_ratio: float = 0.20,
    max_url_ratio: float = 0.20,
) -> QualityReport:
    normalized = normalize_text(text)
    n = len(normalized)
    if n < min_chars:
        return QualityReport(False, "too_short", n, 0.0, 0.0)
    if n > max_chars:
        return QualityReport(False, "too_long", n, 0.0, 0.0)
    alpha = sum(ch.isalpha() for ch in normalized) / max(n, 1)
    url_ratio = len(_URL_RE.findall(normalized)) / max(n / 100.0, 1.0)
    if alpha < min_alpha_ratio:
        return QualityReport(False, "low_alpha_ratio", n, alpha, url_ratio)
    if url_ratio > max_url_ratio:
        return QualityReport(False, "url_heavy", n, alpha, url_ratio)
    # A very repetitive line is usually a scrape/template failure, not a useful abstract.
    words = normalized.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.08:
        return QualityReport(False, "repetitive", n, alpha, url_ratio)
    return QualityReport(True, "accepted", n, alpha, url_ratio)


def filter_quality(
    records: Iterable[Mapping[str, object]],
    *,
    min_chars: int = 80,
    max_chars: int = 200_000,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    accepted: list[dict[str, object]] = []
    reasons: dict[str, int] = {}
    for raw in records:
        row = dict(raw)
        text = normalize_text(str(row.get("text", "")))
        report = quality_report(text, min_chars=min_chars, max_chars=max_chars)
        if report.accepted:
            row["text"] = text
            accepted.append(row)
        else:
            reasons[report.reason] = reasons.get(report.reason, 0) + 1
    return accepted, reasons


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _shingles(text: str, size: int = 5) -> set[str]:
    words = text.lower().split()
    if len(words) <= size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _minhash(shingles: set[str], *, permutations: int = 32) -> tuple[int, ...]:
    if not shingles:
        return tuple([0] * permutations)
    signatures: list[int] = []
    for seed in range(permutations):
        signatures.append(
            min(int(_sha256(f"{seed}:{item}")[:16], 16) for item in shingles)
        )
    return tuple(signatures)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def deduplicate_records(
    records: Sequence[Mapping[str, object]],
    *,
    jaccard_threshold: float = 0.85,
    permutations: int = 32,
) -> tuple[list[dict[str, object]], dict[str, int | float]]:
    """Remove exact and near-duplicate records deterministically.

    The implementation uses exact hashes first and MinHash signatures to avoid comparing every pair.
    It is intentionally dependency-free for smoke tests; large corpora can swap this function for
    a datasketch/LSH implementation while preserving the metadata contract.
    """

    kept: list[dict[str, object]] = []
    exact_seen: set[str] = set()
    buckets: dict[tuple[int, int], list[int]] = {}
    shingles_by_index: list[set[str]] = []
    dropped_exact = 0
    dropped_near = 0
    for raw in records:
        row = dict(raw)
        text = normalize_text(str(row.get("text", "")))
        digest = _sha256(text)
        if digest in exact_seen:
            dropped_exact += 1
            continue
        shingles = _shingles(text)
        signature = _minhash(shingles, permutations=permutations)
        candidate_indices: set[int] = set()
        # Four bands give a cheap candidate index while retaining deterministic behavior.
        band_size = max(permutations // 4, 1)
        for start in range(0, permutations, band_size):
            band_bytes = repr(signature[start : start + band_size]).encode("ascii")
            band_hash = int(hashlib.sha256(band_bytes).hexdigest()[:16], 16)
            key = (start // band_size, band_hash)
            candidate_indices.update(buckets.get(key, []))
        is_near_duplicate = any(
            _jaccard(shingles, shingles_by_index[index]) >= jaccard_threshold
            for index in candidate_indices
        )
        if is_near_duplicate:
            dropped_near += 1
            continue
        row["text"] = text
        row["text_sha256"] = digest
        kept_index = len(kept)
        kept.append(row)
        exact_seen.add(digest)
        shingles_by_index.append(shingles)
        for start in range(0, permutations, band_size):
            band_bytes = repr(signature[start : start + band_size]).encode("ascii")
            band_hash = int(hashlib.sha256(band_bytes).hexdigest()[:16], 16)
            key = (start // band_size, band_hash)
            buckets.setdefault(key, []).append(kept_index)
    total = len(records)
    stats: dict[str, int | float] = {
        "input_records": total,
        "kept_records": len(kept),
        "dropped_exact": dropped_exact,
        "dropped_near": dropped_near,
        "dedup_rate": (total - len(kept)) / total if total else 0.0,
    }
    return kept, stats


def stable_split(record_id: object, *, validation_mod: int = 10, test_mod: int = 10) -> str:
    """Assign a record to train/validation/test using a stable SHA-1 bucket."""

    bucket = int(hashlib.sha1(str(record_id).encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < test_mod:
        return "test"
    if bucket < test_mod + validation_mod:
        return "validation"
    return "train"


def split_records(records: Iterable[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    splits = {"train": [], "validation": [], "test": []}
    for index, raw in enumerate(records):
        row = dict(raw)
        record_id = row.get("id", row.get("pmid", index))
        row["split"] = stable_split(record_id)
        splits[row["split"]].append(row)
    return splits


def mix_domain_and_replay(
    domain: Sequence[Mapping[str, object]],
    replay: Sequence[Mapping[str, object]],
    *,
    domain_ratio: float = 0.8,
    seed: int = 1337,
    target_records: int | None = None,
) -> list[dict[str, object]]:
    """Create a deterministic mixture while preserving source labels."""

    if not 0.0 < domain_ratio <= 1.0:
        raise ValueError("domain_ratio must be in (0, 1]")
    if not domain and not replay:
        return []
    if target_records is None:
        target_records = len(domain) + len(replay)
    domain_needed = round(target_records * domain_ratio)
    replay_needed = target_records - domain_needed
    # A tiny local RNG avoids importing numpy for data preparation.
    import random

    rng = random.Random(seed)
    domain_pool = [dict(row, source="domain") for row in domain]
    replay_pool = [dict(row, source="replay") for row in replay]
    rng.shuffle(domain_pool)
    rng.shuffle(replay_pool)

    def take(pool: list[dict[str, object]], count: int) -> list[dict[str, object]]:
        if not pool or count <= 0:
            return []
        if count <= len(pool):
            return pool[:count]
        return [dict(pool[i % len(pool)], repeated=True) for i in range(count)]

    mixed = take(domain_pool, domain_needed) + take(replay_pool, replay_needed)
    rng.shuffle(mixed)
    return mixed
