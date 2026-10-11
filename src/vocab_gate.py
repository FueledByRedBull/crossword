from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VocabGateResult:
    passed: bool
    reason: str
    term_count: int
    min_required: int
    max_allowed: int | None


def evaluate_vocab_gate(
    term_count: int,
    *,
    min_required: int = 30,
    max_allowed: int | None = None,
) -> VocabGateResult:
    if term_count < min_required:
        reason = "insufficient_terms"
    elif max_allowed is not None and term_count > max_allowed:
        reason = "too_many_terms"
    else:
        reason = "ok"
    return VocabGateResult(
        passed=reason == "ok",
        reason=reason,
        term_count=term_count,
        min_required=min_required,
        max_allowed=max_allowed,
    )
