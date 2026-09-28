"""
test_feedback.py
================
Unit tests for feedback.py — override monitoring and candidate retraining.

Covers:
- record_override() records a feedback entry
- compute_override_rate() calculates correctly
- override_monitoring_report() threshold logic
- Threshold not exceeded
- Threshold exceeded — retraining recommended
- request_candidate_retrain() creates a pending candidate
- evaluate_candidate() updates status and cv_accuracy
- approve_candidate() sets status to approved
- reject_candidate() sets status to rejected
- activate_candidate() requires approved status
- Rejected candidate can never be activated
- Active model unchanged unless explicitly approved

Run with:
    pytest src/tests/test_feedback.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from feedback import (
    record_override,
    compute_override_rate,
    override_monitoring_report,
    request_candidate_retrain,
    evaluate_candidate,
    approve_candidate,
    reject_candidate,
    activate_candidate,
    get_active_candidate,
    candidate_status_summary,
    OverrideFeedback,
    CandidateModel,
    FEEDBACK_LOG,
    CANDIDATE_LOG,
    OVERRIDE_THRESHOLD_DEFAULT,
)
from models.evidence_item import EvidenceItem
from models.triage_models import TriageResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(item_id: str = "E001") -> EvidenceItem:
    return EvidenceItem(
        item_id=item_id,
        label="Test item",
        evidence_type="biological_dna",
        offence_type="homicide",
        probative_value=3,
        perishability=3,
        exclusionary_power=2,
        contamination_risk=2,
        specialist_required=1,
        testing_lead_time=5,
    )


def _make_result(item_id: str = "E001", tier: str = "Critical") -> TriageResult:
    return TriageResult(
        item_id=item_id,
        priority_tier=tier,
        priority_score=0.85,
        urgency_flag=True,
        explanation="test",
        decision_path="",
        model_importances=[],
        item_feature_values={"probative_value": 3},
        model_used="decision_tree",
        cv_accuracy=0.80,
    )


# ---------------------------------------------------------------------------
# Override recording
# ---------------------------------------------------------------------------

class TestRecordOverride:
    def test_record_override_returns_feedback(self):
        fb = record_override(
            evidence_id="E001",
            model_recommendation="High",
            investigator_decision="Critical",
            override_reason="Physical link confirmed",
            model_name="decision_tree",
            cv_accuracy=0.797,
        )
        assert isinstance(fb, OverrideFeedback)
        assert fb.evidence_id == "E001"
        assert fb.model_recommendation == "High"
        assert fb.investigator_decision == "Critical"

    def test_timestamp_auto_populated(self):
        fb = record_override("E002", "Low", "High")
        assert fb.timestamp
        assert len(fb.timestamp) > 10


# ---------------------------------------------------------------------------
# Override rate calculation
# ---------------------------------------------------------------------------

class TestComputeOverrideRate:
    def test_zero_total_returns_zero(self):
        assert compute_override_rate({}, 0) == 0.0

    def test_no_overrides_returns_zero(self):
        assert compute_override_rate({}, 10) == 0.0

    def test_all_overridden(self):
        overrides = {"E001": ("Critical", "reason"), "E002": ("High", "reason")}
        assert compute_override_rate(overrides, 2) == 1.0

    def test_partial_overrides(self):
        overrides = {"E001": ("Critical", "reason")}
        rate = compute_override_rate(overrides, 5)
        assert abs(rate - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Override monitoring report
# ---------------------------------------------------------------------------

class TestOverrideMonitoringReport:
    def test_report_structure(self):
        report = override_monitoring_report({}, 10, 0.20)
        assert "override_count" in report
        assert "total_decisions" in report
        assert "override_rate" in report
        assert "override_pct" in report
        assert "threshold" in report
        assert "threshold_exceeded" in report
        assert "retraining_recommended" in report
        assert "message" in report

    def test_threshold_not_exceeded(self):
        overrides = {"E001": ("Critical", "r")}
        report = override_monitoring_report(overrides, 10, 0.20)
        assert report["threshold_exceeded"] is False
        assert report["retraining_recommended"] is False

    def test_threshold_exceeded(self):
        overrides = {
            "E001": ("Critical", "r1"),
            "E002": ("High", "r2"),
            "E003": ("Standard", "r3"),
        }
        report = override_monitoring_report(overrides, 10, 0.20)
        assert report["threshold_exceeded"] is True
        assert report["retraining_recommended"] is True
        assert "retraining recommended" in report["message"].lower()

    def test_message_includes_counts(self):
        overrides = {"E001": ("Critical", "r"), "E002": ("High", "r")}
        report = override_monitoring_report(overrides, 8, 0.20)
        assert "2" in report["message"]
        assert "8" in report["message"]

    def test_threshold_boundary_exactly_at_threshold(self):
        # Exactly at threshold (not exceeded — rule is >, not >=)
        overrides = {"E001": ("Critical", "r"), "E002": ("High", "r")}
        report = override_monitoring_report(overrides, 10, 0.20)
        # 2/10 == 0.20, which is exactly at threshold, not ABOVE it
        assert report["threshold_exceeded"] is False


# ---------------------------------------------------------------------------
# Candidate retraining workflow
# ---------------------------------------------------------------------------

class TestCandidateWorkflow:
    def _fresh_candidate(self) -> CandidateModel:
        """Create a fresh candidate with unique ID."""
        item  = _make_item("EC001")
        result = _make_result("EC001", "High")
        candidate = request_candidate_retrain(
            model_name="decision_tree",
            base_accuracy=0.797,
            overrides={"EC001": ("Critical", "physical link confirmed")},
            items=[item],
            results=[result],
            session_id="test-session",
        )
        return candidate

    def test_candidate_created_pending(self):
        cand = self._fresh_candidate()
        assert isinstance(cand, CandidateModel)
        assert cand.status == "pending"

    def test_candidate_has_id(self):
        cand = self._fresh_candidate()
        assert cand.candidate_id.startswith("CAND-")

    def test_evaluate_candidate_updates_status(self):
        cand = self._fresh_candidate()
        evaluate_candidate(cand)
        assert cand.status == "evaluated"

    def test_evaluate_candidate_sets_cv_accuracy(self):
        cand = self._fresh_candidate()
        evaluate_candidate(cand)
        assert 0.0 <= cand.cv_accuracy <= 1.0

    def test_approve_evaluated_candidate(self):
        cand = self._fresh_candidate()
        evaluate_candidate(cand)
        approve_candidate(cand, "approver1")
        assert cand.status == "approved"
        assert cand.approved_by == "approver1"

    def test_reject_evaluated_candidate(self):
        cand = self._fresh_candidate()
        evaluate_candidate(cand)
        reject_candidate(cand, "reviewer1", "Does not improve metrics")
        assert cand.status == "rejected"

    def test_activate_approved_candidate(self):
        cand = self._fresh_candidate()
        evaluate_candidate(cand)
        approve_candidate(cand, "approver1")
        activate_candidate(cand)
        assert get_active_candidate() is not None
        assert get_active_candidate().candidate_id == cand.candidate_id

    def test_approve_requires_evaluated_status(self):
        cand = self._fresh_candidate()
        # pending — not yet evaluated
        with pytest.raises(ValueError):
            approve_candidate(cand, "approver1")

    def test_activate_requires_approved_status(self):
        cand = self._fresh_candidate()
        evaluate_candidate(cand)
        # not yet approved
        with pytest.raises(ValueError):
            activate_candidate(cand)

    def test_rejected_candidate_cannot_be_activated(self):
        cand = self._fresh_candidate()
        evaluate_candidate(cand)
        reject_candidate(cand, "reviewer1")
        with pytest.raises(ValueError):
            activate_candidate(cand)

    def test_candidate_status_summary_returns_dict(self):
        summary = candidate_status_summary()
        assert isinstance(summary, dict)
        assert "total_feedback" in summary
        assert "total_candidates" in summary
        assert "candidate_log" in summary
        assert "feedback_log" in summary
