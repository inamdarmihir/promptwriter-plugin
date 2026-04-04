"""
Tests for PromptClassification drift-risk fields added in context_engineer.py.
No LLM or Qdrant calls needed.
"""
from __future__ import annotations

from prompt_guidance.context_engineer import PromptClassification


class TestPromptClassificationDriftFields:
    def test_default_drift_fields(self):
        pc = PromptClassification()
        assert pc.drift_risk is None
        assert pc.drift_affected_queries == 0

    def test_summary_without_drift(self):
        pc = PromptClassification(type="code_generation", domain="technical", complexity="medium")
        summary = pc.summary()
        assert "drift_risk" not in summary

    def test_summary_with_drift_risk_none(self):
        pc = PromptClassification(drift_risk="none", drift_affected_queries=0)
        summary = pc.summary()
        assert "drift_risk" not in summary

    def test_summary_with_low_drift_risk(self):
        pc = PromptClassification(drift_risk="low", drift_affected_queries=1)
        summary = pc.summary()
        assert "drift_risk=low" in summary
        assert "1 queries affected" in summary

    def test_summary_with_high_drift_risk(self):
        pc = PromptClassification(drift_risk="high", drift_affected_queries=5)
        summary = pc.summary()
        assert "drift_risk=high" in summary
        assert "5 queries affected" in summary

    def test_summary_with_critical_drift(self):
        pc = PromptClassification(
            type="instruction_following",
            domain="technical",
            complexity="complex",
            drift_risk="critical",
            drift_affected_queries=8,
        )
        summary = pc.summary()
        assert "drift_risk=critical" in summary
        assert "8 queries affected" in summary

    def test_from_json_does_not_set_drift_fields(self):
        """LLM output never contains drift fields — they are computed post-rewrite."""
        raw = '{"type": "code_generation", "domain": "technical", "complexity": "medium", "main_weaknesses": [], "retrieval_queries": []}'
        pc = PromptClassification.from_json(raw)
        assert pc.drift_risk is None
        assert pc.drift_affected_queries == 0

    def test_high_drift_risk_can_appear_in_main_weaknesses(self):
        pc = PromptClassification(main_weaknesses=["no_role", "high_drift_risk"])
        assert "high_drift_risk" in pc.main_weaknesses
        assert "no_role" in pc.summary()
