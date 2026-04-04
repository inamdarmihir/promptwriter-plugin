"""
Drift Detector
──────────────
Detects semantic drift when a system prompt or long context prompt is modified.

When a SYS PROMPT changes, not all user queries are equally affected.
Some queries may become less aligned with the new prompt (regressions),
while others may improve. This module quantifies those shifts so prompt
engineers can make informed decisions before deploying changes.

Core algorithm:
  For each historical user query Q, compute:
    sim_old = cosine_similarity(embed(Q), embed(P_old))
    sim_new = cosine_similarity(embed(Q), embed(P_new))
    shift   = sim_new - sim_old   (positive = improved, negative = degraded)

  Queries where abs(shift) > threshold are flagged.

Questionnaire impact:
  Regressed queries are mapped to PromptClassification weakness categories
  (no_role, too_vague, no_context, no_format, ambiguous_intent), giving
  the user a structured view of which query types are most at risk.

Usage:
    detector = DriftDetector(embeddings)
    report = detector.analyze(old_prompt, new_prompt, historical_queries)
    print(report.severity)
    for r in report.regressions:
        print(r.query, r.shift)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from prompt_guidance.embeddings import BaseEmbeddings


# ─────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────

_HIGH_RISK_SHIFT = 0.20
_MEDIUM_RISK_SHIFT = 0.10

# Fallback queries used when no historical queries are provided.
# Cover a broad set of use-case types to stress-test any system prompt.
SAMPLE_QUERIES: list[str] = [
    "How do I get started?",
    "What are your capabilities?",
    "Can you help me write code?",
    "Write me a creative short story",
    "Summarize this document for me",
    "What is the weather like today?",
    "Help me debug this error message",
    "Explain machine learning simply",
    "Translate this paragraph to Spanish",
    "What are your limitations?",
    "How do I reset my password?",
    "Tell me a joke",
    "Give me a recipe for pasta",
    "What is the capital of France?",
    "Schedule a meeting for tomorrow",
]


# ─────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────

@dataclass
class QueryDrift:
    """Drift metrics for a single user query under a prompt change."""

    query: str
    sim_old: float        # cosine similarity with old prompt
    sim_new: float        # cosine similarity with new prompt
    shift: float          # sim_new - sim_old  (negative = degraded)
    risk_level: str       # "high" | "medium" | "low"
    direction: str        # "degraded" | "improved" | "neutral"

    @property
    def is_regression(self) -> bool:
        return self.direction == "degraded"

    @property
    def is_improvement(self) -> bool:
        return self.direction == "improved"


@dataclass
class DriftReport:
    """
    Full drift analysis comparing old_prompt vs new_prompt over a set of
    historical user queries.
    """

    old_prompt: str
    new_prompt: str
    total_queries: int
    threshold: float
    regressions: list[QueryDrift] = field(default_factory=list)
    improvements: list[QueryDrift] = field(default_factory=list)
    neutral: list[QueryDrift] = field(default_factory=list)
    severity: str = "none"            # "none" | "low" | "medium" | "high" | "critical"
    summary: str = ""
    questionnaire_impact: dict[str, list[str]] = field(default_factory=dict)
    # ^ maps PromptClassification weakness category → list of affected query strings

    @property
    def high_risk_count(self) -> int:
        return sum(1 for r in self.regressions if r.risk_level == "high")

    @property
    def regression_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return len(self.regressions) / self.total_queries

    @property
    def improvement_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return len(self.improvements) / self.total_queries


# ─────────────────────────────────────────────────────────────────
# Drift Detector
# ─────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Quantifies semantic drift when a prompt (especially a system prompt or
    long context prompt) is modified.

    The detector embeds both prompts and all historical user queries, then
    measures how each query's alignment shifts between the old and new prompt.
    """

    def __init__(self, embeddings: BaseEmbeddings):
        self.embeddings = embeddings

    def analyze(
        self,
        old_prompt: str,
        new_prompt: str,
        user_queries: Optional[list[str]] = None,
        threshold: float = 0.15,
    ) -> DriftReport:
        """
        Compare old_prompt vs new_prompt across user_queries.

        Args:
            old_prompt:   The original system/context prompt.
            new_prompt:   The modified or enhanced prompt to evaluate.
            user_queries: Historical user queries to test against.
                          Falls back to SAMPLE_QUERIES if not provided.
            threshold:    Min abs(shift) to flag a query as drifted.
                          Default 0.15 (15 % cosine distance change).

        Returns:
            DriftReport with regressions, improvements, severity and
            questionnaire_impact breakdown.
        """
        queries = user_queries if user_queries else SAMPLE_QUERIES

        # Embed prompts and all queries
        vec_old = self.embeddings.embed(old_prompt)
        vec_new = self.embeddings.embed(new_prompt)
        query_vecs = self.embeddings.embed_batch(queries)

        regressions: list[QueryDrift] = []
        improvements: list[QueryDrift] = []
        neutral_list: list[QueryDrift] = []

        for query, q_vec in zip(queries, query_vecs):
            sim_old = _cosine_similarity(q_vec, vec_old)
            sim_new = _cosine_similarity(q_vec, vec_new)
            shift = sim_new - sim_old
            abs_shift = abs(shift)

            # Risk level is determined by magnitude of degradation only
            if shift < 0 and abs_shift >= _HIGH_RISK_SHIFT:
                risk_level = "high"
            elif shift < 0 and abs_shift >= _MEDIUM_RISK_SHIFT:
                risk_level = "medium"
            else:
                risk_level = "low"

            if shift < -threshold:
                direction = "degraded"
            elif shift > threshold:
                direction = "improved"
            else:
                direction = "neutral"

            drift = QueryDrift(
                query=query,
                sim_old=round(sim_old, 4),
                sim_new=round(sim_new, 4),
                shift=round(shift, 4),
                risk_level=risk_level,
                direction=direction,
            )

            if direction == "degraded":
                regressions.append(drift)
            elif direction == "improved":
                improvements.append(drift)
            else:
                neutral_list.append(drift)

        # Sort regressions worst-first, improvements best-first
        regressions.sort(key=lambda x: x.shift)
        improvements.sort(key=lambda x: x.shift, reverse=True)

        severity = _compute_severity(regressions, len(queries))
        summary = _build_summary(regressions, improvements, len(queries), severity)
        impact = _compute_questionnaire_impact(regressions)

        return DriftReport(
            old_prompt=old_prompt,
            new_prompt=new_prompt,
            total_queries=len(queries),
            threshold=threshold,
            regressions=regressions,
            improvements=improvements,
            neutral=neutral_list,
            severity=severity,
            summary=summary,
            questionnaire_impact=impact,
        )


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (safe against zero-norm)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_severity(regressions: list[QueryDrift], total: int) -> str:
    """Map regression counts → severity label."""
    if total == 0 or not regressions:
        return "none"
    high_risk = sum(1 for r in regressions if r.risk_level == "high")
    rate = len(regressions) / total

    if high_risk >= 3 or rate > 0.5:
        return "critical"
    if high_risk >= 1 or rate > 0.3:
        return "high"
    if rate > 0.15:
        return "medium"
    return "low"


def _build_summary(
    regressions: list[QueryDrift],
    improvements: list[QueryDrift],
    total: int,
    severity: str,
) -> str:
    r = len(regressions)
    i = len(improvements)
    n = total - r - i
    parts = [
        f"{total} queries tested — {r} regressed, {i} improved, {n} neutral.",
        f"Drift severity: {severity.upper()}.",
    ]
    if regressions:
        worst = regressions[0]
        parts.append(
            f"Worst regression: '{worst.query[:60]}' (shift={worst.shift:+.3f})."
        )
    if improvements:
        best = improvements[0]
        parts.append(
            f"Best improvement: '{best.query[:60]}' (shift={best.shift:+.3f})."
        )
    return " ".join(parts)


# Maps PromptClassification weakness categories to keywords found in queries.
# Used to tell the user which questionnaire categories are most impacted.
_WEAKNESS_KEYWORDS: dict[str, list[str]] = {
    "no_role": ["role", "persona", "you are", "act as", "who are you"],
    "too_vague": ["what is", "explain", "tell me", "describe", "general"],
    "no_context": ["context", "background", "about", "regarding", "situation"],
    "no_format": ["format", "output", "structure", "json", "list", "table"],
    "ambiguous_intent": ["can you", "help me", "i need", "please", "how do i"],
}


def _compute_questionnaire_impact(regressions: list[QueryDrift]) -> dict[str, list[str]]:
    """
    Map regressed queries to PromptClassification weakness categories.

    Returns a dict like:
      {"ambiguous_intent": ["Can you help me write code?", ...], ...}
    """
    impact: dict[str, list[str]] = {}
    for drift in regressions:
        q_lower = drift.query.lower()
        for category, keywords in _WEAKNESS_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                impact.setdefault(category, []).append(drift.query)
    return impact


def is_system_prompt(prompt: str) -> bool:
    """
    Heuristic: returns True if the prompt looks like a system/instruction prompt
    rather than a short user query.

    Criteria:
      • Length ≥ 100 characters, AND
      • Contains at least one system-prompt marker keyword.
    """
    if len(prompt) < 100:
        return False
    lower = prompt.lower()
    markers = [
        "you are", "your role", "your task", "as an", "act as",
        "you will", "you must", "your job", "your goal",
        "instructions:", "system:", "###", "you should",
    ]
    return any(marker in lower for marker in markers)
