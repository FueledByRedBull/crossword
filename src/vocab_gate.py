from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VocabGateResult:
    passed: bool
    reason: str
    term_count: int
    min_required: int
    max_allowed: int


def evaluate_vocab_gate(
    term_count: int,
    *,
    min_required: int = 40,
    max_allowed: int = 80,
) -> VocabGateResult:
    if term_count < min_required:
        return VocabGateResult(
            passed=False,
            reason="insufficient_terms",
            term_count=term_count,
            min_required=min_required,
            max_allowed=max_allowed,
        )
    if term_count > max_allowed:
        return VocabGateResult(
            passed=False,
            reason="too_many_terms",
            term_count=term_count,
            min_required=min_required,
            max_allowed=max_allowed,
        )
    return VocabGateResult(
        passed=True,
        reason="ok",
        term_count=term_count,
        min_required=min_required,
        max_allowed=max_allowed,
    )
