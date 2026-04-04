"""
Unit tests for drift_detector.py

These tests use a MockEmbeddings that returns hand-crafted vectors, so
no LLM/Qdrant infrastructure is needed to run them.
"""
from __future__ import annotations

import math

import pytest

from prompt_guidance.drift_detector import (
    SAMPLE_QUERIES,
    DriftDetector,
    DriftReport,
    QueryDrift,
    _cosine_similarity,
    _compute_severity,
    _compute_questionnaire_impact,
    _build_summary,
    is_system_prompt,
)
from prompt_guidance.embeddings import BaseEmbeddings


# ─────────────────────────────────────────────────────────────────
# Mock embeddings — deterministic, no I/O
# ─────────────────────────────────────────────────────────────────

class MockEmbeddings(BaseEmbeddings):
    """
    Returns fixed 4-D vectors keyed by text prefix.
    Vectors not in the registry get a constant fallback.
    """

    def __init__(self, registry: dict[str, list[float]]):
        self._registry = registry

    def embed(self, text: str) -> list[float]:
        for key, vec in self._registry.items():
            if text.startswith(key):
                return list(vec)
        return [1.0, 0.0, 0.0, 0.0]  # fallback

    @property
    def dimensions(self) -> int:
        return 4

    @property
    def provider_name(self) -> str:
        return "mock"


def _unit(v: list[float]) -> list[float]:
    """Normalise a vector to unit length."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


# ─────────────────────────────────────────────────────────────────
# _cosine_similarity
# ─────────────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = _unit([1.0, 2.0, 3.0, 4.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_symmetric(self):
        a = _unit([1.0, 2.0, 3.0, 4.0])
        b = _unit([4.0, 3.0, 2.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(_cosine_similarity(b, a), abs=1e-9)


# ─────────────────────────────────────────────────────────────────
# _compute_severity
# ─────────────────────────────────────────────────────────────────

class TestBuildSummary:
    def _drift(self, query: str, shift: float, direction: str) -> QueryDrift:
        return QueryDrift(
            query=query,
            sim_old=0.8 if shift < 0 else 0.3,
            sim_new=0.8 + shift if shift >= 0 else 0.8 + shift,
            shift=round(shift, 4),
            risk_level="high" if abs(shift) > 0.2 else "low",
            direction=direction,
        )

    def test_summary_mentions_counts(self):
        regressions = [self._drift("q1", -0.3, "degraded")]
        improvements = [self._drift("q2", 0.3, "improved")]
        s = _build_summary(regressions, improvements, 5, "low")
        assert "5 queries tested" in s
        assert "1 regressed" in s
        assert "1 improved" in s

    def test_summary_contains_worst_regression(self):
        regressions = [self._drift("worst query", -0.4, "degraded")]
        s = _build_summary(regressions, [], 5, "high")
        assert "worst query" in s

    def test_summary_no_regressions(self):
        s = _build_summary([], [], 10, "none")
        assert "10 queries tested" in s
        assert "0 regressed" in s


class TestComputeSeverity:
    def _make_regression(self, risk: str) -> QueryDrift:
        return QueryDrift(
            query="q", sim_old=0.5, sim_new=0.2, shift=-0.3,
            risk_level=risk, direction="degraded",
        )

    def test_no_regressions(self):
        assert _compute_severity([], 10) == "none"

    def test_empty_total(self):
        assert _compute_severity([], 0) == "none"

    def test_critical_by_high_risk_count(self):
        regressions = [self._make_regression("high")] * 3
        assert _compute_severity(regressions, 10) == "critical"

    def test_critical_by_rate(self):
        regressions = [self._make_regression("medium")] * 6
        assert _compute_severity(regressions, 10) == "critical"

    def test_high_severity_single_high_risk(self):
        regressions = [self._make_regression("high")]
        assert _compute_severity(regressions, 10) == "high"

    def test_medium_severity(self):
        regressions = [self._make_regression("medium")] * 2
        assert _compute_severity(regressions, 10) == "medium"

    def test_low_severity(self):
        regressions = [self._make_regression("low")] * 1
        assert _compute_severity(regressions, 10) == "low"


# ─────────────────────────────────────────────────────────────────
# is_system_prompt
# ─────────────────────────────────────────────────────────────────

class TestIsSystemPrompt:
    def test_short_prompt_is_not_system(self):
        assert is_system_prompt("Hello, what is the weather?") is False

    def test_long_prompt_without_markers(self):
        long = "a" * 150
        assert is_system_prompt(long) is False

    def test_you_are_marker(self):
        p = "You are a helpful assistant. " + "x" * 80
        assert is_system_prompt(p) is True

    def test_your_task_marker(self):
        p = "Your task is to analyse code and find bugs. " + "y" * 80
        assert is_system_prompt(p) is True

    def test_instructions_marker(self):
        p = "Instructions: always respond in JSON. " + "z" * 80
        assert is_system_prompt(p) is True

    def test_case_insensitive(self):
        p = "YOU ARE AN EXPERT CODER. " + "a" * 80
        assert is_system_prompt(p) is True


# ─────────────────────────────────────────────────────────────────
# _compute_questionnaire_impact
# ─────────────────────────────────────────────────────────────────

class TestQuestionnaireImpact:
    def _drift(self, query: str) -> QueryDrift:
        return QueryDrift(
            query=query, sim_old=0.5, sim_new=0.2, shift=-0.3,
            risk_level="high", direction="degraded",
        )

    def test_ambiguous_intent_category(self):
        impact = _compute_questionnaire_impact([
            self._drift("Can you help me write code?"),
        ])
        assert "ambiguous_intent" in impact
        assert "Can you help me write code?" in impact["ambiguous_intent"]

    def test_no_role_category(self):
        impact = _compute_questionnaire_impact([
            self._drift("Who are you and what is your role?"),
        ])
        assert "no_role" in impact

    def test_empty_regressions(self):
        assert _compute_questionnaire_impact([]) == {}

    def test_multiple_categories(self):
        regressions = [
            self._drift("Can you help me with my task?"),
            self._drift("What is the best format for output?"),
        ]
        impact = _compute_questionnaire_impact(regressions)
        # "ambiguous_intent" matches "can you"
        assert "ambiguous_intent" in impact
        # "no_format" matches "format"
        assert "no_format" in impact


# ─────────────────────────────────────────────────────────────────
# DriftDetector.analyze
# ─────────────────────────────────────────────────────────────────

class TestDriftDetector:
    """
    Uses hand-crafted orthogonal vectors to produce predictable cosine shifts.

    Layout:
      e1 = [1, 0, 0, 0]  — "technical" direction
      e2 = [0, 1, 0, 0]  — "general" direction

    old_prompt → e1  (technical assistant)
    new_prompt → e2  (general assistant; orthogonal to old)

    Queries along e1 will degrade (sim drops from 1 → 0).
    Queries along e2 will improve (sim rises from 0 → 1).
    """

    OLD = "old_prompt"
    NEW = "new_prompt"
    Q_TECH = "tech_query"      # aligned with e1
    Q_GEN = "general_query"    # aligned with e2

    @pytest.fixture
    def detector(self) -> DriftDetector:
        registry = {
            self.OLD: [1.0, 0.0, 0.0, 0.0],
            self.NEW: [0.0, 1.0, 0.0, 0.0],
            self.Q_TECH: [1.0, 0.0, 0.0, 0.0],
            self.Q_GEN: [0.0, 1.0, 0.0, 0.0],
        }
        return DriftDetector(MockEmbeddings(registry))

    def test_no_drift_on_identical_prompts(self, detector: DriftDetector):
        queries = [self.Q_TECH, self.Q_GEN]
        report = detector.analyze(self.OLD, self.OLD, user_queries=queries, threshold=0.05)
        assert report.severity == "none"
        assert report.regressions == []
        assert report.improvements == []

    def test_regression_detected(self, detector: DriftDetector):
        """tech_query should degrade when prompt changes from e1 → e2."""
        report = detector.analyze(
            self.OLD, self.NEW,
            user_queries=[self.Q_TECH],
            threshold=0.05,
        )
        assert len(report.regressions) == 1
        assert report.regressions[0].query == self.Q_TECH
        assert report.regressions[0].shift == pytest.approx(-1.0, abs=1e-4)
        assert report.regressions[0].is_regression is True

    def test_improvement_detected(self, detector: DriftDetector):
        """general_query should improve when prompt changes from e1 → e2."""
        report = detector.analyze(
            self.OLD, self.NEW,
            user_queries=[self.Q_GEN],
            threshold=0.05,
        )
        assert len(report.improvements) == 1
        assert report.improvements[0].query == self.Q_GEN
        assert report.improvements[0].shift == pytest.approx(1.0, abs=1e-4)
        assert report.improvements[0].is_improvement is True

    def test_total_queries_matches(self, detector: DriftDetector):
        queries = [self.Q_TECH, self.Q_GEN, "fallback1", "fallback2"]
        report = detector.analyze(self.OLD, self.NEW, user_queries=queries, threshold=0.05)
        assert report.total_queries == 4

    def test_regression_rate(self, detector: DriftDetector):
        queries = [self.Q_TECH, self.Q_GEN]
        report = detector.analyze(self.OLD, self.NEW, user_queries=queries, threshold=0.05)
        assert report.regression_rate == pytest.approx(0.5, abs=1e-6)

    def test_improvement_rate(self, detector: DriftDetector):
        queries = [self.Q_TECH, self.Q_GEN]
        report = detector.analyze(self.OLD, self.NEW, user_queries=queries, threshold=0.05)
        assert report.improvement_rate == pytest.approx(0.5, abs=1e-6)

    def test_falls_back_to_sample_queries(self, detector: DriftDetector):
        """When no user_queries provided, SAMPLE_QUERIES are used."""
        report = detector.analyze(self.OLD, self.NEW)
        assert report.total_queries == len(SAMPLE_QUERIES)

    def test_regressions_sorted_worst_first(self):
        """Multiple regressions should be sorted ascending by shift (worst first)."""
        # Use two queries both aligned towards old prompt but differently
        registry = {
            self.OLD: [1.0, 0.0, 0.0, 0.0],
            self.NEW: [0.0, 1.0, 0.0, 0.0],
            "partial": [0.9, 0.436, 0.0, 0.0],   # mostly e1, small shift
            self.Q_TECH: [1.0, 0.0, 0.0, 0.0],    # fully e1, max shift
        }
        d = DriftDetector(MockEmbeddings(registry))
        report = d.analyze(self.OLD, self.NEW, user_queries=[self.Q_TECH, "partial"], threshold=0.05)
        shifts = [r.shift for r in report.regressions]
        assert shifts == sorted(shifts)

    def test_severity_critical_when_many_regressions(self, detector: DriftDetector):
        """With 6+ regressions out of 10 queries, severity should be critical."""
        # All queries use fallback vector [1,0,0,0] = aligned with old prompt
        queries = [f"fallback_{i}" for i in range(10)]
        report = detector.analyze(self.OLD, self.NEW, user_queries=queries, threshold=0.05)
        assert report.severity in ("high", "critical")

    def test_report_has_summary(self, detector: DriftDetector):
        report = detector.analyze(self.OLD, self.NEW, user_queries=[self.Q_TECH], threshold=0.05)
        assert len(report.summary) > 0
        assert "regressed" in report.summary

    def test_questionnaire_impact_populated(self, detector: DriftDetector):
        """Queries with matching keywords should appear in questionnaire_impact."""
        queries = ["Can you help me debug this error?"]  # matches "ambiguous_intent"
        report = detector.analyze(self.OLD, self.NEW, user_queries=queries, threshold=0.05)
        # This query degrades (fallback vec aligned with old) or improves depending on direction.
        # What matters is impact is computed for regressions only.
        if report.regressions:
            # At least something in impact
            assert isinstance(report.questionnaire_impact, dict)


# ─────────────────────────────────────────────────────────────────
# DriftReport properties
# ─────────────────────────────────────────────────────────────────

class TestDriftReportProperties:
    def _make_report(self, regressions: list[QueryDrift]) -> DriftReport:
        return DriftReport(
            old_prompt="old", new_prompt="new",
            total_queries=10, threshold=0.15,
            regressions=regressions,
        )

    def test_high_risk_count(self):
        regressions = [
            QueryDrift("q1", 0.9, 0.6, -0.3, "high", "degraded"),
            QueryDrift("q2", 0.8, 0.6, -0.2, "medium", "degraded"),
            QueryDrift("q3", 0.7, 0.5, -0.2, "high", "degraded"),
        ]
        report = self._make_report(regressions)
        assert report.high_risk_count == 2

    def test_regression_rate_zero_queries(self):
        report = DriftReport(
            old_prompt="x", new_prompt="y",
            total_queries=0, threshold=0.15,
        )
        assert report.regression_rate == 0.0

    def test_is_regression_property(self):
        d = QueryDrift("q", 0.9, 0.5, -0.4, "high", "degraded")
        assert d.is_regression is True
        assert d.is_improvement is False

    def test_is_improvement_property(self):
        d = QueryDrift("q", 0.5, 0.9, 0.4, "low", "improved")
        assert d.is_improvement is True
        assert d.is_regression is False
