"""
test_triage_models.py
=====================
Unit tests for models/triage_models.py.

Covers:
- get_trained_model returns valid TrainingResult for all three model names
- predict_priority returns a TriageResult with all required fields
- Explanation content: DT returns decision_path, RF/GB return model_importances
- validate_item catches invalid field values
- compute_urgency_flag rules (from triage_models)
- DATA_DISCLAIMER constant is present

Run with:
    pytest src/tests/test_triage_models.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models.evidence_item import EvidenceItem, CaseContext
from models.triage_models import (
    get_trained_model,
    predict_priority,
    validate_item,
    compute_urgency_flag,
    DATA_DISCLAIMER,
    TrainingResult,
    TriageResult,
    PRIORITY_LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_item():
    return EvidenceItem(
        item_id="T001",
        label="Test blood swab",
        evidence_type="biological_dna",
        offence_type="homicide",
        probative_value=3,
        perishability=3,
        exclusionary_power=3,
        contamination_risk=2,
        specialist_required=1,
        testing_lead_time=5,
        collection_age_hours=2,
        evidence_condition=2,
        specialist_type="DNA analyst",
    )


@pytest.fixture(scope="module")
def dt_result():
    return get_trained_model("decision_tree")


@pytest.fixture(scope="module")
def rf_result():
    return get_trained_model("random_forest")


@pytest.fixture(scope="module")
def gb_result():
    return get_trained_model("gradient_boosting")


# ---------------------------------------------------------------------------
# TrainingResult tests
# ---------------------------------------------------------------------------

class TestGetTrainedModel:
    def test_dt_returns_training_result(self, dt_result):
        assert isinstance(dt_result, TrainingResult)

    def test_rf_returns_training_result(self, rf_result):
        assert isinstance(rf_result, TrainingResult)

    def test_gb_returns_training_result(self, gb_result):
        assert isinstance(gb_result, TrainingResult)

    def test_dt_model_name(self, dt_result):
        assert dt_result.model_name == "decision_tree"

    def test_rf_model_name(self, rf_result):
        assert rf_result.model_name == "random_forest"

    def test_gb_model_name(self, gb_result):
        assert gb_result.model_name == "gradient_boosting"

    def test_cv_accuracy_in_range(self, dt_result, rf_result, gb_result):
        for tr in [dt_result, rf_result, gb_result]:
            assert 0.0 < tr.cv_accuracy <= 1.0, f"{tr.model_name} cv_accuracy out of range"

    def test_data_disclaimer_in_training_result(self, dt_result):
        assert "synthetic" in dt_result.data_disclaimer.lower() or "demonstration" in dt_result.data_disclaimer.lower()

    def test_feature_names_not_empty(self, dt_result):
        assert len(dt_result.feature_names) > 0

    def test_cv_report_not_empty(self, dt_result):
        assert isinstance(dt_result.cv_report, str) and len(dt_result.cv_report) > 0

    def test_invalid_model_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_trained_model("not_a_model")


# ---------------------------------------------------------------------------
# TriageResult tests
# ---------------------------------------------------------------------------

class TestPredictPriority:
    def test_dt_returns_triage_result(self, sample_item, dt_result):
        r = predict_priority(sample_item, dt_result)
        assert isinstance(r, TriageResult)

    def test_rf_returns_triage_result(self, sample_item, rf_result):
        r = predict_priority(sample_item, rf_result)
        assert isinstance(r, TriageResult)

    def test_gb_returns_triage_result(self, sample_item, gb_result):
        r = predict_priority(sample_item, gb_result)
        assert isinstance(r, TriageResult)

    def test_priority_tier_is_valid_label(self, sample_item, dt_result):
        r = predict_priority(sample_item, dt_result)
        assert r.priority_tier in PRIORITY_LABELS

    def test_priority_score_in_range(self, sample_item, dt_result):
        r = predict_priority(sample_item, dt_result)
        assert 0.0 <= r.priority_score <= 1.0

    def test_urgency_flag_is_bool(self, sample_item, dt_result):
        r = predict_priority(sample_item, dt_result)
        assert isinstance(r.urgency_flag, bool)

    def test_model_used_field(self, sample_item, dt_result):
        r = predict_priority(sample_item, dt_result)
        assert r.model_used == "decision_tree"

    def test_item_feature_values_not_empty(self, sample_item, dt_result):
        r = predict_priority(sample_item, dt_result)
        assert isinstance(r.item_feature_values, dict)
        assert len(r.item_feature_values) > 0


# ---------------------------------------------------------------------------
# Explanation tests
# ---------------------------------------------------------------------------

class TestExplanations:
    def test_dt_has_decision_path(self, sample_item, dt_result):
        r = predict_priority(sample_item, dt_result)
        assert isinstance(r.decision_path, str)
        assert len(r.decision_path) > 0, "DT should produce a non-empty decision_path"

    def test_dt_model_importances_empty(self, sample_item, dt_result):
        r = predict_priority(sample_item, dt_result)
        # DT uses decision_path, not global importances
        assert r.model_importances == [] or r.model_importances is None or len(r.model_importances) == 0

    def test_rf_has_model_importances(self, sample_item, rf_result):
        r = predict_priority(sample_item, rf_result)
        assert isinstance(r.model_importances, list)
        assert len(r.model_importances) > 0

    def test_rf_decision_path_empty(self, sample_item, rf_result):
        r = predict_priority(sample_item, rf_result)
        assert r.decision_path == "" or r.decision_path is None

    def test_gb_has_model_importances(self, sample_item, gb_result):
        r = predict_priority(sample_item, gb_result)
        assert len(r.model_importances) > 0

    def test_model_importance_structure(self, sample_item, rf_result):
        r = predict_priority(sample_item, rf_result)
        for entry in r.model_importances:
            assert "feature" in entry
            assert "importance" in entry
            assert "rank" in entry


# ---------------------------------------------------------------------------
# validate_item tests
# ---------------------------------------------------------------------------

class TestValidateItem:
    def test_valid_item_returns_no_errors(self, sample_item):
        errors = validate_item(sample_item)
        assert errors == []

    def test_probative_value_out_of_range(self):
        item = EvidenceItem(
            item_id="V001", label="bad", evidence_type="biological_dna",
            offence_type="homicide", probative_value=5, perishability=1,
            exclusionary_power=1, contamination_risk=1, specialist_required=0,
            testing_lead_time=3,
        )
        errors = validate_item(item)
        assert any("probative" in e.lower() for e in errors)

    def test_perishability_out_of_range(self):
        item = EvidenceItem(
            item_id="V002", label="bad", evidence_type="biological_dna",
            offence_type="homicide", probative_value=1, perishability=0,
            exclusionary_power=1, contamination_risk=1, specialist_required=0,
            testing_lead_time=3,
        )
        errors = validate_item(item)
        assert any("perishability" in e.lower() or "degradation" in e.lower() for e in errors)

    def test_negative_lead_time(self):
        item = EvidenceItem(
            item_id="V003", label="bad", evidence_type="biological_dna",
            offence_type="homicide", probative_value=1, perishability=1,
            exclusionary_power=1, contamination_risk=1, specialist_required=0,
            testing_lead_time=-1,
        )
        errors = validate_item(item)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# compute_urgency_flag tests
# ---------------------------------------------------------------------------

def _urgency_item(perishability: int, collection_age_hours: int) -> EvidenceItem:
    """Build a minimal EvidenceItem for urgency flag testing."""
    return EvidenceItem(
        item_id="U000",
        label="test",
        evidence_type="trace_glass",
        offence_type="robbery",
        probative_value=1,
        perishability=perishability,
        exclusionary_power=1,
        contamination_risk=1,
        specialist_required=0,
        testing_lead_time=3,
        collection_age_hours=collection_age_hours,
    )


class TestComputeUrgencyFlag:
    def test_perishability_3_always_urgent(self):
        assert compute_urgency_flag(_urgency_item(3, 0)) is True
        assert compute_urgency_flag(_urgency_item(3, 100)) is True

    def test_perishability_2_within_6h_urgent(self):
        assert compute_urgency_flag(_urgency_item(2, 1)) is True
        assert compute_urgency_flag(_urgency_item(2, 5)) is True

    def test_perishability_2_at_6h_not_urgent(self):
        # boundary: age == 6 is NOT urgent (rule: 0 < age < 6, strictly less)
        assert compute_urgency_flag(_urgency_item(2, 6)) is False

    def test_perishability_2_age_0_unknown_not_urgent(self):
        # age == 0 means unknown — urgency not assumed
        assert compute_urgency_flag(_urgency_item(2, 0)) is False

    def test_perishability_1_never_urgent(self):
        assert compute_urgency_flag(_urgency_item(1, 1)) is False
        assert compute_urgency_flag(_urgency_item(1, 100)) is False


# ---------------------------------------------------------------------------
# DATA_DISCLAIMER constant
# ---------------------------------------------------------------------------

class TestDataDisclaimer:
    def test_disclaimer_not_empty(self):
        assert isinstance(DATA_DISCLAIMER, str)
        assert len(DATA_DISCLAIMER) > 20

    def test_disclaimer_mentions_synthetic(self):
        assert "synthetic" in DATA_DISCLAIMER.lower()
