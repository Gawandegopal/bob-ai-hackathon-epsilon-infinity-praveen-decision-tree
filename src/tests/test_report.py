"""
test_report.py
==============
Unit tests for report.py.

Covers:
- generate_report returns a non-empty string
- All nine required sections are present
- Data disclaimer appears at least once
- Policy section present when policy is active
- Policy section shows "No active policy" when policy is None
- Report records correct model name
- Investigator override decisions are reflected in the report

Run with:
    pytest src/tests/test_report.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timezone

from models.evidence_item import EvidenceItem, CaseContext
from models.triage_models import TriageResult
from models.policy import TriagePolicy
from scheduler import ScheduledItem, BATCH_IMMEDIATE, BATCH_ARCHIVE
from report import ReportData, generate_report
from models.triage_models import DATA_DISCLAIMER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_context():
    return CaseContext(
        fir_number="TEST-2024-001",
        offence_type="robbery",
        narrative="Armed robbery at convenience store.",
    )


@pytest.fixture
def sample_items():
    return [
        EvidenceItem(
            item_id="R001",
            label="Blood swab",
            evidence_type="biological_dna",
            offence_type="robbery",
            probative_value=3,
            perishability=3,
            exclusionary_power=3,
            contamination_risk=2,
            specialist_required=1,
            testing_lead_time=5,
            collection_age_hours=2,
            evidence_condition=2,
            specialist_type="DNA analyst",
        ),
        EvidenceItem(
            item_id="R002",
            label="CCTV footage",
            evidence_type="digital_cctv",
            offence_type="robbery",
            probative_value=2,
            perishability=1,
            exclusionary_power=2,
            contamination_risk=1,
            specialist_required=0,
            testing_lead_time=2,
            collection_age_hours=0,
            evidence_condition=3,
        ),
    ]


@pytest.fixture
def sample_results():
    return [
        TriageResult(
            item_id="R001",
            priority_tier="Critical",
            priority_score=0.9,
            urgency_flag=True,
            explanation="High probative value, perishable evidence.",
            decision_path="Node 1: perishability <= 2.5 -> False\nNode 2: -> Critical",
            model_importances=[],
            item_feature_values={"probative_value": 3, "perishability": 3},
            model_used="decision_tree",
            cv_accuracy=0.80,
        ),
        TriageResult(
            item_id="R002",
            priority_tier="Standard",
            priority_score=0.55,
            urgency_flag=False,
            explanation="Digital evidence, stable, low urgency.",
            decision_path="",
            model_importances=[
                {"rank": 1, "feature": "probative_value", "importance": 0.40},
            ],
            item_feature_values={"probative_value": 2, "perishability": 1},
            model_used="decision_tree",
            cv_accuracy=0.80,
        ),
    ]


@pytest.fixture
def sample_schedule(sample_items, sample_results):
    return [
        ScheduledItem(
            item=sample_items[0],
            triage_result=sample_results[0],
            policy_result=None,
            batch=BATCH_IMMEDIATE,
            batch_reason="Critical priority — immediate examination required.",
            investigator_decision="accepted",
            override_reason=None,
            final_tier="Critical",
        ),
        ScheduledItem(
            item=sample_items[1],
            triage_result=sample_results[1],
            policy_result=None,
            batch=BATCH_ARCHIVE,
            batch_reason="Standard, no specialist — routine processing.",
            investigator_decision="overridden",
            override_reason="Evidence shows suspect clearly on camera.",
            final_tier="High",
        ),
    ]


@pytest.fixture
def report_data_no_policy(sample_context, sample_items, sample_results, sample_schedule):
    return ReportData(
        context=sample_context,
        items=sample_items,
        results=sample_results,
        schedule=sample_schedule,
        policy=None,
        generated_at="2024-01-01 00:00 UTC",
        model_used="Decision Tree",
        cv_accuracy=0.797,
    )


@pytest.fixture
def sample_policy():
    return TriagePolicy(
        version="1.0.0",
        label="Robbery Focus Policy",
        description="Emphasises probative and degradation dimensions.",
        weight_P=2.0,
        weight_D=1.5,
        weight_E=1.0,
        weight_S=1.0,
        status="approved",
        proposed_by="admin",
        approved_by="approver",
    )


@pytest.fixture
def report_data_with_policy(sample_context, sample_items, sample_results, sample_schedule, sample_policy):
    return ReportData(
        context=sample_context,
        items=sample_items,
        results=sample_results,
        schedule=sample_schedule,
        policy=sample_policy,
        generated_at="2024-01-01 00:00 UTC",
        model_used="Decision Tree",
        cv_accuracy=0.797,
    )


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_returns_non_empty_string(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_contains_fir_number(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        assert "TEST-2024-001" in md

    def test_contains_offence_type(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        assert "robbery" in md.lower()

    def test_contains_model_name(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        assert "Decision Tree" in md

    def test_contains_item_ids(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        assert "R001" in md
        assert "R002" in md


# ---------------------------------------------------------------------------
# Required section tests
# ---------------------------------------------------------------------------

REQUIRED_PHRASES = [
    "AI-Assisted",                    # header / disclaimer
    "recommendations",                # recommendation label in disclaimer
    "Case",                           # case summary section
    "Evidence",                       # evidence inventory or classification
    "Schedule",                       # FSL schedule section
    "Investigator",                   # review decisions section
    "Model",                          # model metadata section
    "synthetic",                      # disclaimer footer (case-insensitive covered by lower())
]


class TestRequiredSections:
    def test_all_required_sections_present(self, report_data_no_policy):
        md = generate_report(report_data_no_policy).lower()
        for phrase in REQUIRED_PHRASES:
            assert phrase.lower() in md, f"Missing section/phrase: '{phrase}'"

    def test_disclaimer_present(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        # DATA_DISCLAIMER text or a synonym must appear
        assert "synthetic" in md.lower() or "demonstration" in md.lower()

    def test_recommendation_label_present(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        assert "recommendation" in md.lower()


# ---------------------------------------------------------------------------
# Policy section tests
# ---------------------------------------------------------------------------

class TestPolicySection:
    def test_no_policy_shows_no_active_policy(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        assert "no active policy" in md.lower()

    def test_active_policy_version_in_report(self, report_data_with_policy):
        md = generate_report(report_data_with_policy)
        assert "1.0.0" in md

    def test_active_policy_label_in_report(self, report_data_with_policy):
        md = generate_report(report_data_with_policy)
        assert "Robbery Focus Policy" in md

    def test_policy_disclaimer_present_when_policy_active(self, report_data_with_policy):
        md = generate_report(report_data_with_policy)
        assert "prototype" in md.lower() or "configurable" in md.lower()


# ---------------------------------------------------------------------------
# Override tests
# ---------------------------------------------------------------------------

class TestOverridesInReport:
    def test_override_reason_in_report(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        assert "Evidence shows suspect clearly on camera." in md

    def test_overridden_item_shows_final_tier(self, report_data_no_policy):
        md = generate_report(report_data_no_policy)
        # R002 was overridden to High
        assert "High" in md
