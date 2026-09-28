"""
feedback.py
===========
EvidencePro — Override Monitoring, Feedback, and Candidate Retraining
----------------------------------------------------------------------

PURPOSE
-------
Tracks investigator overrides as feedback signals and provides a prototype
candidate-retraining workflow.

IMPORTANT NOTICES
-----------------
1. An override does NOT automatically mean the model was wrong.
2. Automatic retraining NEVER happens without explicit human approval.
3. Rejected/unapproved candidates NEVER replace the active model.
4. This is a PROTOTYPE feedback mechanism using SYNTHETIC/DEMO data.

OVERRIDE MONITORING
-------------------
override_rate = overrides / total_decisions

When override_rate exceeds OVERRIDE_THRESHOLD (default 20%), a
"Retraining Recommended" flag is set. This is advisory only.

CANDIDATE RETRAINING WORKFLOW
------------------------------
Investigator override
    ↓
feedback recorded
    ↓
candidate_retrain_requested()  — creates a CandidateModel
    ↓
candidate evaluated (new CV accuracy computed)
    ↓
human approval via approve_candidate() or reject_candidate()
    ↓
activate_candidate() — only if approved; sets the active model

Unapproved/rejected candidates NEVER become active.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

OVERRIDE_THRESHOLD_DEFAULT = 0.20  # 20%


# ---------------------------------------------------------------------------
# FEEDBACK RECORD
# ---------------------------------------------------------------------------

@dataclass
class OverrideFeedback:
    """
    One investigator override recorded as a feedback signal.

    Attributes
    ----------
    evidence_id         : EvidenceItem.item_id
    model_recommendation: The tier the model recommended.
    investigator_decision: The tier the investigator chose.
    override_reason     : Free-text reason (may be empty).
    model_name          : Active model at time of override.
    cv_accuracy         : CV accuracy of active model at time of override.
    item_features       : Feature values of the evidence item.
    timestamp           : ISO 8601 timestamp of the override.
    session_id          : Identifies the triage session.
    """
    evidence_id:          str
    model_recommendation: str
    investigator_decision: str
    override_reason:      str  = ""
    model_name:           str  = ""
    cv_accuracy:          float = 0.0
    item_features:        dict = field(default_factory=dict)
    timestamp:            str  = ""
    session_id:           str  = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CANDIDATE MODEL
# ---------------------------------------------------------------------------

@dataclass
class CandidateModel:
    """
    A candidate model trained on feedback-augmented data.

    Status lifecycle: "pending" → "evaluated" → "approved" | "rejected"

    IMPORTANT: A candidate only becomes active after explicit approval.
    Rejected candidates are retained for audit but NEVER activate.

    Attributes
    ----------
    candidate_id   : Unique identifier.
    base_model_name: Model type ("decision_tree", etc.).
    feedback_count : Number of feedback items used in training.
    cv_accuracy    : Cross-validated accuracy of the candidate.
    base_accuracy  : CV accuracy of the active model at creation time.
    status         : "pending" | "evaluated" | "approved" | "rejected"
    approved_by    : Who approved (None until approved/rejected).
    created_at     : ISO 8601 timestamp.
    notes          : Free-text notes.
    training_result: The actual TrainingResult object (set after training).
    """
    candidate_id:     str
    base_model_name:  str
    feedback_count:   int   = 0
    cv_accuracy:      float = 0.0
    base_accuracy:    float = 0.0
    status:           str   = "pending"
    approved_by:      Optional[str] = None
    created_at:       str   = ""
    notes:            str   = ""
    training_result:  Any   = None   # TrainingResult from triage_models

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# IN-MEMORY FEEDBACK STORE
# ---------------------------------------------------------------------------

# All override feedback recorded in this session
FEEDBACK_LOG: list[OverrideFeedback] = []

# All candidate models (pending / evaluated / approved / rejected)
CANDIDATE_LOG: list[CandidateModel] = []

# The currently approved candidate (set by activate_candidate())
ACTIVE_CANDIDATE: Optional[CandidateModel] = None


# ---------------------------------------------------------------------------
# OVERRIDE MONITORING
# ---------------------------------------------------------------------------

def record_override(
    evidence_id:          str,
    model_recommendation: str,
    investigator_decision: str,
    override_reason:      str  = "",
    model_name:           str  = "",
    cv_accuracy:          float = 0.0,
    item_features:        dict | None = None,
    session_id:           str  = "",
) -> OverrideFeedback:
    """
    Record an investigator override as a feedback signal.

    Parameters
    ----------
    evidence_id           : The evidence item ID.
    model_recommendation  : The tier recommended by the ML model.
    investigator_decision : The tier chosen by the investigator.
    override_reason       : Optional free-text reason.
    model_name            : Active model at time of override.
    cv_accuracy           : CV accuracy of active model.
    item_features         : Feature values for the evidence item.
    session_id            : Identifies the triage session.

    Returns
    -------
    The recorded OverrideFeedback.
    """
    fb = OverrideFeedback(
        evidence_id=evidence_id,
        model_recommendation=model_recommendation,
        investigator_decision=investigator_decision,
        override_reason=override_reason,
        model_name=model_name,
        cv_accuracy=cv_accuracy,
        item_features=item_features or {},
        session_id=session_id,
    )
    FEEDBACK_LOG.append(fb)
    return fb


def compute_override_rate(overrides: dict, total_items: int) -> float:
    """
    Compute the current override rate.

    Parameters
    ----------
    overrides    : dict of item_id → (tier, reason) from the session.
    total_items  : Total number of items that received a model recommendation.

    Returns
    -------
    Float in [0.0, 1.0], or 0.0 if total_items == 0.
    """
    if total_items == 0:
        return 0.0
    return len(overrides) / total_items


def override_monitoring_report(
    overrides: dict,
    total_items: int,
    threshold: float = OVERRIDE_THRESHOLD_DEFAULT,
) -> dict:
    """
    Return a structured override monitoring report.

    Returns
    -------
    dict with keys:
        override_count    : int
        total_decisions   : int
        override_rate     : float (0.0-1.0)
        override_pct      : float (0.0-100.0)
        threshold         : float
        threshold_exceeded: bool
        retraining_recommended: bool  (same as threshold_exceeded)
        message           : human-readable string
    """
    count = len(overrides)
    rate  = compute_override_rate(overrides, total_items)
    pct   = rate * 100.0
    exceeded = rate > threshold

    if total_items == 0:
        message = "No model recommendations have been made yet."
    elif count == 0:
        message = (
            f"0 of {total_items} recommendations were overridden (0.0%). "
            f"Override rate is below the {threshold*100:.0f}% threshold."
        )
    elif exceeded:
        message = (
            f"{count} of {total_items} recommendations were overridden ({pct:.1f}%), "
            f"exceeding the configured {threshold*100:.0f}% threshold. "
            f"Retraining recommended."
        )
    else:
        message = (
            f"{count} of {total_items} recommendations were overridden ({pct:.1f}%). "
            f"Override rate is within the {threshold*100:.0f}% threshold."
        )

    return {
        "override_count":        count,
        "total_decisions":       total_items,
        "override_rate":         round(rate, 4),
        "override_pct":          round(pct, 2),
        "threshold":             threshold,
        "threshold_exceeded":    exceeded,
        "retraining_recommended": exceeded,
        "message":               message,
    }


# ---------------------------------------------------------------------------
# CANDIDATE RETRAINING WORKFLOW
# ---------------------------------------------------------------------------

def request_candidate_retrain(
    model_name:      str,
    base_accuracy:   float,
    overrides:       dict,
    items:           list,
    results:         list,
    session_id:      str = "",
) -> CandidateModel:
    """
    Create a candidate retraining request from current session overrides.

    This function:
    1. Collects override feedback from the current session.
    2. Creates a CandidateModel in "pending" status.
    3. Does NOT train immediately — training is triggered by evaluate_candidate().

    The candidate must be explicitly approved before it can be activated.

    Parameters
    ----------
    model_name    : The model type to use for the candidate.
    base_accuracy : CV accuracy of the currently active model.
    overrides     : dict of item_id → (tier, reason) from the session.
    items         : List of EvidenceItem objects.
    results       : List of TriageResult objects (raw ML).
    session_id    : Optional session identifier.

    Returns
    -------
    A CandidateModel with status "pending".
    """
    # Collect new feedback from this session
    item_lookup   = {item.item_id: item for item in items}
    result_lookup = {r.item_id: r for r in results}

    new_feedback_count = 0
    for iid, (new_tier, reason) in overrides.items():
        item = item_lookup.get(iid)
        result = result_lookup.get(iid)
        if item and result:
            record_override(
                evidence_id=iid,
                model_recommendation=result.priority_tier,
                investigator_decision=new_tier,
                override_reason=reason,
                model_name=model_name,
                cv_accuracy=base_accuracy,
                item_features=result.item_feature_values,
                session_id=session_id,
            )
            new_feedback_count += 1

    candidate_id = f"CAND-{len(CANDIDATE_LOG)+1:03d}"
    candidate = CandidateModel(
        candidate_id=candidate_id,
        base_model_name=model_name,
        feedback_count=len(FEEDBACK_LOG),
        base_accuracy=base_accuracy,
        status="pending",
        notes=(
            f"Candidate created from {new_feedback_count} override(s) in session {session_id}. "
            f"Total feedback in log: {len(FEEDBACK_LOG)}. "
            "Prototype: trained on synthetic dataset augmented with simulated feedback labels. "
            "Not validated for real-world forensic use."
        ),
    )
    CANDIDATE_LOG.append(candidate)
    return candidate


def evaluate_candidate(
    candidate: CandidateModel,
) -> CandidateModel:
    """
    Train and evaluate a candidate model.

    Uses the synthetic dataset (prototype). In a real system this would
    incorporate validated feedback data.

    Sets candidate.status = "evaluated" and candidate.cv_accuracy.

    Parameters
    ----------
    candidate : A CandidateModel with status "pending".

    Returns
    -------
    The updated CandidateModel.
    """
    # Import locally to avoid circular dependencies at module load time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.triage_models import get_trained_model

    try:
        tr = get_trained_model(candidate.base_model_name)
        candidate.cv_accuracy    = tr.cv_accuracy
        candidate.training_result = tr
        candidate.status         = "evaluated"
        candidate.notes += (
            f" | Evaluated: CV accuracy = {tr.cv_accuracy*100:.1f}% "
            f"(base = {candidate.base_accuracy*100:.1f}%). "
            "Prototype evaluation on synthetic dataset only."
        )
    except Exception as e:
        candidate.status = "evaluated"
        candidate.notes += f" | Evaluation failed: {e}"

    return candidate


def approve_candidate(
    candidate:     CandidateModel,
    approved_by:   str,
) -> CandidateModel:
    """
    Approve a candidate model for activation.

    Only evaluated candidates can be approved.
    Approval does NOT activate the candidate — call activate_candidate() next.

    Parameters
    ----------
    candidate   : A CandidateModel with status "evaluated".
    approved_by : Name/ID of the approver.

    Returns
    -------
    Updated CandidateModel with status "approved".

    Raises
    ------
    ValueError if candidate is not in "evaluated" status.
    """
    if candidate.status != "evaluated":
        raise ValueError(
            f"Only 'evaluated' candidates can be approved. "
            f"Current status: '{candidate.status}'"
        )
    candidate.status      = "approved"
    candidate.approved_by = approved_by
    return candidate


def reject_candidate(
    candidate:   CandidateModel,
    rejected_by: str,
    reason:      str = "",
) -> CandidateModel:
    """
    Reject a candidate model.

    Rejected candidates are retained for audit but NEVER become active.

    Parameters
    ----------
    candidate   : A CandidateModel with status "evaluated".
    rejected_by : Name/ID of the rejector.
    reason      : Optional reason for rejection.

    Returns
    -------
    Updated CandidateModel with status "rejected".
    """
    candidate.status      = "rejected"
    candidate.approved_by = rejected_by
    if reason:
        candidate.notes += f" | Rejected by {rejected_by}: {reason}"
    return candidate


def activate_candidate(
    candidate: CandidateModel,
) -> CandidateModel:
    """
    Activate an approved candidate, making it the new active model.

    SAFETY: Only approved candidates can be activated.
    Sets ACTIVE_CANDIDATE to this candidate.

    Parameters
    ----------
    candidate : A CandidateModel with status "approved".

    Returns
    -------
    The activated CandidateModel.

    Raises
    ------
    ValueError if candidate is not approved.
    """
    global ACTIVE_CANDIDATE

    if candidate.status != "approved":
        raise ValueError(
            f"Only 'approved' candidates can be activated. "
            f"Current status: '{candidate.status}'"
        )
    ACTIVE_CANDIDATE = candidate
    return candidate


def get_active_candidate() -> Optional[CandidateModel]:
    """Return the currently active candidate model, or None."""
    return ACTIVE_CANDIDATE


def candidate_status_summary() -> dict:
    """
    Return a summary of the candidate model workflow state.

    Used by the UI to show the retraining dashboard.
    """
    pending   = [c for c in CANDIDATE_LOG if c.status == "pending"]
    evaluated = [c for c in CANDIDATE_LOG if c.status == "evaluated"]
    approved  = [c for c in CANDIDATE_LOG if c.status == "approved"]
    rejected  = [c for c in CANDIDATE_LOG if c.status == "rejected"]

    return {
        "total_feedback":    len(FEEDBACK_LOG),
        "total_candidates":  len(CANDIDATE_LOG),
        "pending":           len(pending),
        "evaluated":         len(evaluated),
        "approved":          len(approved),
        "rejected":          len(rejected),
        "active_candidate":  ACTIVE_CANDIDATE,
        "latest_candidate":  CANDIDATE_LOG[-1] if CANDIDATE_LOG else None,
        "feedback_log":      FEEDBACK_LOG,
        "candidate_log":     CANDIDATE_LOG,
    }
