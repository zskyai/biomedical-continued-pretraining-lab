"""Metrics used to report domain gain versus general-ability forgetting."""

from __future__ import annotations


def relative_domain_gain(base_value: float, adapted_value: float, *, lower_is_better: bool = True) -> float:
    """Return relative improvement, with a positive value meaning improvement."""

    if base_value == 0:
        return 0.0
    if lower_is_better:
        return (base_value - adapted_value) / abs(base_value)
    return (adapted_value - base_value) / abs(base_value)


def forgetting_delta(base_general_score: float, adapted_general_score: float) -> float:
    """Return general-score change; negative values indicate forgetting."""

    return adapted_general_score - base_general_score


def pareto_point(
    base_domain_ppl: float,
    adapted_domain_ppl: float,
    base_general_score: float,
    adapted_general_score: float,
) -> dict[str, float]:
    return {
        "domain_ppl_gain": relative_domain_gain(base_domain_ppl, adapted_domain_ppl),
        "general_score_delta": forgetting_delta(base_general_score, adapted_general_score),
    }

