"""
test_scheduler.py
=================
Unit tests for scheduler.py.

Covers:
- Critical item → Batch 1
- Low item → Batch 3
- Urgent (perishability=3) item → Batch 1
- High non-urgent item → Batch 2
- Standard + specialist → Batch 2
- Standard + no specialist → Batch 3
- Investigator override moves item to correct batch
- Within-batch sort: perishability DESC, lead time ASC
- build_schedule returns list[ScheduledItem] with correct fields

Run with:
    pytest src/tests/test_scheduler.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models.evidence_item import EvidenceItem
from models.triage_models import TriageResult
from scheduler import build_schedule, ScheduledItem, BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(item_id: str, perishability: int = 1, specialist_required: int = 0,
          testing_lead_time: int = 7, collection_age_hours: int = 0) -> EvidenceItem:
    return EvidenceItem(
        item_id=item_id,
        label=f"Item {item_id}",
        evidence_type="trace_glass",
        offence_type="robbery",
        probative_value=2,
        perishability=perishability,
        exclusionary_power=2,
        contamination_risk=1,
        specialist_required=specialist_required,
        testing_lead_time=testing_lead_time,
        collection_age_hours=collection_age_hours,
        evidence_condition=2,
    )


def _result(priority_tier: str, urgency_flag: bool = False,
            item_id: str = "X000") -> TriageResult:
    return TriageResult(
        item_id=item_id,
        priority_tier=priority_tier,
        priority_score=0.8,
        urgency_flag=urgency_flag,
        explanation="test explanation",
        decision_path="",
        model_importances=[],
        item_feature_values={},
        model_used="decision_tree",
        cv_accuracy=0.80,
    )


# ---------------------------------------------------------------------------
# Batch assignment tests
# ---------------------------------------------------------------------------

class TestBatchAssignment:
    def test_critical_item_goes_to_batch1(self):
        item = _item("C001")
        result = _result("Critical")
        schedule = build_schedule([item], [result], {}, None)
        assert len(schedule) == 1
        assert schedule[0].batch == BATCH_IMMEDIATE

    def test_low_item_goes_to_batch3(self):
        item = _item("L001")
        result = _result("Low")
        schedule = build_schedule([item], [result], {}, None)
        assert schedule[0].batch == BATCH_ARCHIVE

    def test_urgent_high_perishability_goes_to_batch1(self):
        # perishability=3 makes urgency_flag=True
        item = _item("U001", perishability=3)
        result = _result("Standard", urgency_flag=True)
        schedule = build_schedule([item], [result], {}, None)
        assert schedule[0].batch == BATCH_IMMEDIATE

    def test_high_non_urgent_goes_to_batch2(self):
        item = _item("H001", perishability=1)
        result = _result("High", urgency_flag=False)
        schedule = build_schedule([item], [result], {}, None)
        assert schedule[0].batch == BATCH_SECONDARY

    def test_standard_with_specialist_goes_to_batch2(self):
        item = _item("S001", specialist_required=1)
        result = _result("Standard", urgency_flag=False)
        schedule = build_schedule([item], [result], {}, None)
        assert schedule[0].batch == BATCH_SECONDARY

    def test_standard_no_specialist_goes_to_batch3(self):
        item = _item("S002", specialist_required=0)
        result = _result("Standard", urgency_flag=False)
        schedule = build_schedule([item], [result], {}, None)
        assert schedule[0].batch == BATCH_ARCHIVE


# ---------------------------------------------------------------------------
# Override tests
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_override_low_to_critical_goes_to_batch1(self):
        item = _item("O001")
        result = _result("Low")
        overrides = {"O001": ("Critical", "Physical link confirmed")}
        schedule = build_schedule([item], [result], overrides, None)
        si = schedule[0]
        assert si.batch == BATCH_IMMEDIATE
        assert si.final_tier == "Critical"
        assert si.investigator_decision == "overridden"
        assert si.override_reason == "Physical link confirmed"

    def test_override_tier_recorded_on_urgent_item(self):
        """
        Urgency flag is physical — it comes from the raw result and is not
        cleared by an investigator tier override. An override to "Low" on an
        urgent item changes final_tier but keeps the item in Batch 1 because
        the degradation risk is real regardless of the tier decision.
        This matches the documented scheduler behavior (see scheduler.py line
        "urgency is physical, not policy-weighted").
        """
        item = _item("O002", perishability=3)
        result = _result("Critical", urgency_flag=True, item_id="O002")
        overrides = {"O002": ("Low", "Evidence determined non-relevant")}
        schedule = build_schedule([item], [result], overrides, None)
        si = schedule[0]
        # final_tier reflects the override
        assert si.final_tier == "Low"
        assert si.investigator_decision == "overridden"
        # but the item stays urgent — urgency is physical
        assert si.triage_result.urgency_flag is True
        assert si.batch == BATCH_IMMEDIATE

    def test_no_override_is_accepted(self):
        item = _item("O003")
        result = _result("High")
        schedule = build_schedule([item], [result], {}, None)
        assert schedule[0].investigator_decision == "accepted"
        assert schedule[0].override_reason is None


# ---------------------------------------------------------------------------
# ScheduledItem structure tests
# ---------------------------------------------------------------------------

class TestScheduledItemStructure:
    def test_scheduled_item_has_required_fields(self):
        item = _item("F001")
        result = _result("Critical")
        schedule = build_schedule([item], [result], {}, None)
        si = schedule[0]
        assert isinstance(si, ScheduledItem)
        assert si.item is item
        assert si.triage_result is result
        assert isinstance(si.batch, str)
        assert isinstance(si.batch_reason, str) and len(si.batch_reason) > 0
        assert isinstance(si.final_tier, str)

    def test_batch_reason_not_empty(self):
        for tier in ["Critical", "High", "Standard", "Low"]:
            item = _item("F002", specialist_required=1 if tier == "Standard" else 0)
            result = _result(tier)
            schedule = build_schedule([item], [result], {}, None)
            assert len(schedule[0].batch_reason) > 0


# ---------------------------------------------------------------------------
# Multi-item sort tests
# ---------------------------------------------------------------------------

class TestWithinBatchSort:
    def test_higher_perishability_sorted_first_in_batch1(self):
        """Within Batch 1, items with higher perishability come first."""
        item_low_per  = _item("S001", perishability=1)
        item_high_per = _item("S002", perishability=3)
        result_low  = _result("Critical")
        result_high = _result("Critical", urgency_flag=True)
        schedule = build_schedule(
            [item_low_per, item_high_per],
            [result_low, result_high],
            {}, None,
        )
        batch1 = [si for si in schedule if si.batch == BATCH_IMMEDIATE]
        assert len(batch1) == 2
        assert batch1[0].item.perishability >= batch1[1].item.perishability

    def test_shorter_lead_time_sorted_first_when_perishability_equal(self):
        """When perishability is equal, shorter lead time comes first."""
        item_slow = _item("S003", perishability=3, testing_lead_time=14)
        item_fast = _item("S004", perishability=3, testing_lead_time=2)
        result_slow = _result("Critical", urgency_flag=True)
        result_fast = _result("Critical", urgency_flag=True)
        schedule = build_schedule(
            [item_slow, item_fast],
            [result_slow, result_fast],
            {}, None,
        )
        batch1 = [si for si in schedule if si.batch == BATCH_IMMEDIATE]
        assert batch1[0].item.testing_lead_time <= batch1[1].item.testing_lead_time
