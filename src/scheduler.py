"""
scheduler.py
============
EvidencePro — Deterministic FSL Examination Scheduler
-------------------------------------------------------

PURPOSE
-------
Converts a list of triage results into a ranked FSL examination schedule
with three batches. The scheduling logic is entirely rule-based — it is NOT
another ML model.

Rules are transparent, documented here, and must not be changed silently.
Any rule change should be reviewed and reflected in the report and UI.

BATCH ASSIGNMENT RULES
----------------------
The scheduler uses the `final_tier` (which may differ from the raw ML tier
if the investigator has overridden an item) and the `urgency_flag` from the
associated TriageResult.

  Batch 1 — Immediate:
    • final_tier == "Critical"
    • OR urgency_flag == True
      (urgency rule: perishability==3 always; perishability==2 within 6h
       of collection; perishability==1 never; collection_age_hours==0
       means unknown and does NOT trigger urgency)

  Batch 2 — Secondary:
    • final_tier == "High" AND NOT urgency_flag
    • OR final_tier == "Standard" AND item.specialist_required == 1

  Batch 3 — Archive:
    • final_tier == "Low"
    • OR final_tier == "Standard" AND item.specialist_required == 0

WITHIN-BATCH SORT
-----------------
  Primary key  : item.perishability  DESCENDING  (most perishable first)
  Secondary key: item.testing_lead_time ASCENDING (fastest turnaround first)

INVESTIGATOR OVERRIDES
----------------------
Overrides are supplied as a dict mapping item_id → (new_tier, reason).
An override changes `final_tier` and is recorded in the ScheduledItem.
Overrides affect batch assignment — an override to "Critical" promotes an
item to Batch 1 even if the ML said "Low".

Overrides are INDIVIDUAL item decisions, NOT system-wide policy changes.

POLICY INTERACTION
------------------
The scheduler accepts an optional policy-adjusted TriageResult (`policy_result`)
alongside the raw result. The batch assignment uses `final_tier` which comes
from the override or (if no override) from the policy-adjusted tier (if a
policy is active) or from the raw ML tier (if no policy is active).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.evidence_item import EvidenceItem
    from models.triage_models import TriageResult
    from models.policy import TriagePolicy


# ---------------------------------------------------------------------------
# BATCH LABELS
# ---------------------------------------------------------------------------

BATCH_IMMEDIATE  = "Batch 1: Immediate"
BATCH_SECONDARY  = "Batch 2: Secondary"
BATCH_ARCHIVE    = "Batch 3: Archive"

BATCH_ORDER = {
    BATCH_IMMEDIATE: 1,
    BATCH_SECONDARY: 2,
    BATCH_ARCHIVE:   3,
}


# ---------------------------------------------------------------------------
# SCHEDULED ITEM DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class ScheduledItem:
    """
    One evidence item as it appears in the FSL examination schedule.

    Attributes
    ----------
    item                 : The original EvidenceItem.
    triage_result        : Raw ML TriageResult (before any policy adjustment).
    policy_result        : Policy-adjusted TriageResult, or None if no policy active.
    batch                : Batch label string (BATCH_IMMEDIATE / SECONDARY / ARCHIVE).
    batch_reason         : One sentence explaining why this item was assigned
                           to this batch. Shown in the report and UI.
    investigator_decision: "accepted" if the investigator accepted the AI tier,
                           "overridden" if the investigator changed the tier, or
                           None if no decision has been recorded yet.
    override_reason      : Free-text reason supplied by the investigator when
                           overriding. None if not overridden.
    final_tier           : The tier used for scheduling — the investigator's
                           accepted/overridden decision, or (when no decision yet)
                           the policy-adjusted tier if a policy is active, or
                           the raw ML tier otherwise.
    """
    item:                  "EvidenceItem"
    triage_result:         "TriageResult"
    policy_result:         "TriageResult | None"
    batch:                 str
    batch_reason:          str
    investigator_decision: str | None
    override_reason:       str | None
    final_tier:            str


# ---------------------------------------------------------------------------
# BATCH ASSIGNMENT LOGIC
# ---------------------------------------------------------------------------

def _assign_batch(
    final_tier: str,
    urgency_flag: bool,
    specialist_required: int,
) -> tuple[str, str]:
    """
    Return (batch_label, batch_reason) for the given item characteristics.

    This function contains the single authoritative batch assignment logic.
    It is called by build_schedule() and must not be duplicated elsewhere.

    Parameters
    ----------
    final_tier          : The tier to use for scheduling (may be overridden).
    urgency_flag        : From TriageResult.urgency_flag (shared urgency rule).
    specialist_required : From EvidenceItem.specialist_required (0 or 1).

    Returns
    -------
    (batch_label, batch_reason) tuple.
    """
    # Batch 1: Critical tier OR any urgent item
    if final_tier == "Critical":
        return (
            BATCH_IMMEDIATE,
            f"Priority tier is Critical — immediate FSL examination required.",
        )
    if urgency_flag:
        return (
            BATCH_IMMEDIATE,
            f"Urgency flag is set (evidence is perishable or recently collected "
            f"and degrading) — moved to immediate batch regardless of tier ({final_tier}).",
        )

    # Batch 2: High-priority non-urgent, or Standard requiring specialist
    if final_tier == "High":
        return (
            BATCH_SECONDARY,
            f"Priority tier is High with no immediate urgency — secondary batch.",
        )
    if final_tier == "Standard" and specialist_required == 1:
        return (
            BATCH_SECONDARY,
            f"Priority tier is Standard but specialist analysis is required — "
            f"secondary batch to allocate specialist resource.",
        )

    # Batch 3: Low priority, or Standard with no specialist requirement
    if final_tier == "Low":
        return (
            BATCH_ARCHIVE,
            f"Priority tier is Low — archive batch for routine processing.",
        )
    # Standard without specialist
    return (
        BATCH_ARCHIVE,
        f"Priority tier is Standard with no specialist requirement — "
        f"archive batch for standard routine processing.",
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def build_schedule(
    items:     list["EvidenceItem"],
    results:   list["TriageResult"],
    overrides: dict[str, tuple[str, str]],
    policy:    "TriagePolicy | None" = None,
) -> list[ScheduledItem]:
    """
    Build a ranked FSL examination schedule from triage results.

    Parameters
    ----------
    items     : List of EvidenceItem objects (must correspond 1:1 with results).
    results   : List of TriageResult objects from predict_priority() (raw ML).
    overrides : Dict mapping item_id → (new_tier, reason).
                An override changes the item's final_tier and batch assignment.
                Overrides represent investigator decisions on individual items —
                they are NOT system-wide policy changes.
    policy    : The active TriagePolicy, or None.
                If not None, apply_policy() is called to get a policy_result
                for each item. The policy_result tier is used as final_tier
                unless the investigator has overridden the item.

    Returns
    -------
    A sorted list of ScheduledItem objects, ordered by batch then by
    (perishability DESC, testing_lead_time ASC) within each batch.

    Notes
    -----
    Items and results must be the same length and in the same order.
    The urgency_flag from the RAW triage_result is used (urgency is a
    physical property of the evidence, independent of policy weighting).
    """
    if len(items) != len(results):
        raise ValueError(
            f"items and results must have the same length. "
            f"Got {len(items)} items and {len(results)} results."
        )

    # Import apply_policy here to avoid circular import at module load
    from models.policy import apply_policy

    scheduled = []

    for item, raw_result in zip(items, results):
        # Compute policy-adjusted result if a policy is active
        if policy is not None:
            pol_result = apply_policy(raw_result, item, policy)
        else:
            pol_result = None

        # Determine final tier (override > policy > raw ML)
        iid = item.item_id
        if iid in overrides:
            override_tier, override_reason = overrides[iid]
            final_tier            = override_tier
            investigator_decision = "overridden"
        else:
            # No override — accept the recommendation
            final_tier            = pol_result.priority_tier if pol_result is not None else raw_result.priority_tier
            investigator_decision = "accepted"
            override_reason       = None

        # Use raw urgency_flag — urgency is physical, not policy-weighted
        urgency_flag = raw_result.urgency_flag

        batch_label, batch_reason = _assign_batch(
            final_tier,
            urgency_flag,
            item.specialist_required,
        )

        scheduled.append(ScheduledItem(
            item=item,
            triage_result=raw_result,
            policy_result=pol_result,
            batch=batch_label,
            batch_reason=batch_reason,
            investigator_decision=investigator_decision,
            override_reason=override_reason if iid in overrides else None,
            final_tier=final_tier,
        ))

    # Sort: batch order first, then perishability DESC, then lead time ASC
    scheduled.sort(key=lambda si: (
        BATCH_ORDER[si.batch],
        -si.item.perishability,
        si.item.testing_lead_time,
    ))

    return scheduled


def schedule_summary(scheduled: list[ScheduledItem]) -> dict[str, list[ScheduledItem]]:
    """
    Group a sorted schedule into a dict keyed by batch label.

    Returns
    -------
    OrderedDict-equivalent: {BATCH_IMMEDIATE: [...], BATCH_SECONDARY: [...], BATCH_ARCHIVE: [...]}
    """
    groups: dict[str, list[ScheduledItem]] = {
        BATCH_IMMEDIATE: [],
        BATCH_SECONDARY: [],
        BATCH_ARCHIVE:   [],
    }
    for si in scheduled:
        groups[si.batch].append(si)
    return groups
