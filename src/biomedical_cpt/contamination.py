"""Simple, auditable n-gram contamination checks for evaluation text."""

from __future__ import annotations

from collections.abc import Iterable


def _ngrams(text: str, n: int) -> set[str]:
    tokens = " ".join(text.lower().split()).split()
    return {" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


def contamination_report(
    evaluation_texts: Iterable[str],
    corpus_texts: Iterable[str],
    *,
    n: int = 8,
) -> dict[str, float | int | list[int]]:
    """Return per-example and aggregate overlap rates.

    This is a screening audit, not proof of semantic independence. For production corpora, retain
    the matching n-grams and manually review borderline cases.
    """

    corpus_ngrams: set[str] = set()
    for text in corpus_texts:
        corpus_ngrams.update(_ngrams(text, n))
    overlap_counts: list[int] = []
    contaminated = 0
    for text in evaluation_texts:
        grams = _ngrams(text, n)
        overlap = len(grams & corpus_ngrams)
        overlap_counts.append(overlap)
        contaminated += int(overlap > 0)
    total = len(overlap_counts)
    return {
        "evaluation_examples": total,
        "contaminated_examples": contaminated,
        "contamination_rate": contaminated / total if total else 0.0,
        "overlap_ngram_counts": overlap_counts,
        "ngram_size": n,
    }

