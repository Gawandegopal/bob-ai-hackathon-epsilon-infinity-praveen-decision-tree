"""
policy.py
=========
EvidencePro — Optional PDES Policy Layer and Governance
---------------------------------------------------------

IMPORTANT DISCLAIMER
--------------------
The policy weights defined here are a CONFIGURABLE PROTOTYPE MECHANISM.
They are NOT official forensic standards, legally validated thresholds, or
scientifically peer-reviewed forensic triage guidelines. They exist solely
to demonstrate that the system can be configured with different emphasis on
the P/D/E/S dimensions.

Any organisation deploying a system based on this prototype would need to
establish their own validated, legally reviewed policy through appropriate
channels. The prototype governance workflow simulates a review cycle but
does not constitute a real governance or audit process.

Default behaviour
-----------------
DEFAULT_POLICY = None

When no policy is active the ML model's raw tier and score are used directly
by the scheduler and report. No weighting is applied.

Policy governance workflow (prototype simulation)
--------------------------------------------------
  POLICY_ADMIN  →  propose_policy()   →  status: "proposed"
  APPROVER      →  approve_policy()   →  status: "approved"  → becomes active
  APPROVER      →  reject_policy()    →  status: "rejected"

Only the most recently approved policy is active.
Policy history is maintained in POLICY_HISTORY (in-memory for the prototype).

Investigator overrides
----------------------
An investigator can override an individual evidence recommendation via the
human-review step. This is separate from a system-wide policy change:
  - Override   → changes one item's final_tier in one session.
  - Policy     → changes the weighting applied to all items system-wide.

apply_policy() computes a weighted PDES score and re-derives a priority tier.
Both the raw ML tier and the policy-adjusted tier are returned so they can
always be shown side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.evidence_item import EvidenceItem
    from models.triage_models import TriageResult


# ---------------------------------------------------------------------------
# ROLE ENUM
# ---------------------------------------------------------------------------

class PolicyRole(Enum):
    """Prototype simulation of user roles for policy governance."""
    INVESTIGATOR  = "investigator"
    POLICY_ADMIN  = "policy_admin"
    APPROVER      = "approver"


# ---------------------------------------------------------------------------
# POLICY TIER THRESHOLDS
# ---------------------------------------------------------------------------
# These thresholds map a normalised weighted PDES score [0, 1] to a priority
# tier. They are DOCUMENTED AND TRANSPARENT — not hidden in model weights.
#
# Score >= 0.75  →  Critical
# Score >= 0.50  →  High
# Score >= 0.25  →  Standard
# Score <  0.25  →  Low

POLICY_TIER_THRESHOLDS = [
    (0.75, "Critical"),
    (0.50, "High"),
    (0.25, "Standard"),
    (0.00, "Low"),
]

# S dimension: testing_lead_time normalisation ceiling (days)
# A lead time of 0 days → S_inv = 1.0 (fastest possible)
# A lead time >= LEAD_TIME_MAX → S_inv = 0.0 (slowest possible)
LEAD_TIME_MAX_DAYS = 21


# ---------------------------------------------------------------------------
# TRIAGE POLICY DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class TriagePolicy:
    """
    A named, versioned set of PDES dimension weights.

    Attributes
    ----------
    version      : Semantic version string; recorded in every report that uses
                   this policy, e.g. "1.0.0".
    label        : Short human-readable name, e.g. "Homicide Focus Policy".
    description  : Longer explanation of the policy's purpose and rationale.
    weight_P     : Weight for Probative Value (default 1.0).
    weight_D     : Weight for Degradation / Perishability Risk (default 1.0).
    weight_E     : Weight for Exclusionary Power (default 1.0).
    weight_S     : Weight for Processing Speed / Actionability (default 1.0).
    status       : "proposed" | "approved" | "rejected"
    proposed_by  : Name/ID of the Policy Admin who proposed this version.
    approved_by  : Name/ID of the Approver who approved/rejected, or None.

    DISCLAIMER: These weights are a configurable prototype mechanism only.
    They are not official forensic standards.
    """
    version:     str
    label:       str
    description: str   = ""
    weight_P:    float = 1.0
    weight_D:    float = 1.0
    weight_E:    float = 1.0
    weight_S:    float = 1.0
    status:      str   = "proposed"
    proposed_by: str   = ""
    approved_by: str | None = None


# ---------------------------------------------------------------------------
# IN-MEMORY POLICY STORE
# ---------------------------------------------------------------------------

DEFAULT_POLICY: TriagePolicy | None = None

# All proposed/approved/rejected policies are appended here.
# The active policy is the most recently approved entry.
POLICY_HISTORY: list[TriagePolicy] = []


# ---------------------------------------------------------------------------
# GOVERNANCE FUNCTIONS
# ---------------------------------------------------------------------------

def propose_policy(policy: TriagePolicy, role: PolicyRole) -> TriagePolicy:
    """
    Propose a new policy for review.

    Only a POLICY_ADMIN may propose. The policy is added to POLICY_HISTORY
    with status "proposed" and must be approved by an APPROVER before it
    becomes active.

    Parameters
    ----------
    policy : TriagePolicy with status "proposed" and proposed_by set.
    role   : The role of the user calling this function.

    Returns
    -------
    The same TriagePolicy (status unchanged — still "proposed").

    Raises
    ------
    ValueError if role is not POLICY_ADMIN.
    ValueError if policy already has status other than "proposed".
    """
    if role != PolicyRole.POLICY_ADMIN:
        raise ValueError(
            f"Only a POLICY_ADMIN may propose a policy. "
            f"Current role: {role.value}"
        )
    if policy.status != "proposed":
        raise ValueError(
            f"Policy must have status 'proposed' to be submitted. "
            f"Got: '{policy.status}'"
        )
    POLICY_HISTORY.append(policy)
    return policy


def approve_policy(
    policy: TriagePolicy,
    approver_name: str,
    role: PolicyRole,
) -> TriagePolicy:
    """
    Approve a proposed policy, making it the active policy.

    Only an APPROVER may approve. Sets status to "approved" and records
    the approver's name.

    Parameters
    ----------
    policy        : A TriagePolicy with status "proposed".
    approver_name : Name/ID of the approver.
    role          : The role of the user calling this function.

    Returns
    -------
    The updated TriagePolicy with status "approved".

    Raises
    ------
    ValueError if role is not APPROVER.
    ValueError if policy status is not "proposed".
    """
    if role != PolicyRole.APPROVER:
        raise ValueError(
            f"Only an APPROVER may approve a policy. "
            f"Current role: {role.value}"
        )
    if policy.status != "proposed":
        raise ValueError(
            f"Only a 'proposed' policy can be approved. "
            f"Got: '{policy.status}'"
        )
    policy.status      = "approved"
    policy.approved_by = approver_name
    return policy


def reject_policy(
    policy: TriagePolicy,
    approver_name: str,
    role: PolicyRole,
) -> TriagePolicy:
    """
    Reject a proposed policy.

    Only an APPROVER may reject. Sets status to "rejected".

    Parameters
    ----------
    policy        : A TriagePolicy with status "proposed".
    approver_name : Name/ID of the approver.
    role          : The role of the user calling this function.

    Returns
    -------
    The updated TriagePolicy with status "rejected".

    Raises
    ------
    ValueError if role is not APPROVER.
    ValueError if policy status is not "proposed".
    """
    if role != PolicyRole.APPROVER:
        raise ValueError(
            f"Only an APPROVER may reject a policy. "
            f"Current role: {role.value}"
        )
    if policy.status != "proposed":
        raise ValueError(
            f"Only a 'proposed' policy can be rejected. "
            f"Got: '{policy.status}'"
        )
    policy.status      = "rejected"
    policy.approved_by = approver_name
    return policy


def get_active_policy() -> TriagePolicy | None:
    """
    Return the most recently approved policy, or None if no policy is active.

    Scans POLICY_HISTORY from most recent to oldest.
    """
    for policy in reversed(POLICY_HISTORY):
        if policy.status == "approved":
            return policy
    return None


# ---------------------------------------------------------------------------
# POLICY APPLICATION
# ---------------------------------------------------------------------------

def apply_policy(
    result: "TriageResult",
    item: "EvidenceItem",
    policy: TriagePolicy | None,
) -> "TriageResult":
    """
    Apply the optional PDES weighting policy to a TriageResult.

    If policy is None, the result is returned unchanged — this is the default
    and recommended mode for unweighted ML operation.

    When a policy IS active:
      1. A weighted PDES score is computed from the item's feature values.
      2. The score is normalised to [0, 1].
      3. A new priority tier is derived from the documented POLICY_TIER_THRESHOLDS.
      4. A new TriageResult is returned with the adjusted tier and score, and
         policy_version set to the policy's version string.
      5. The ORIGINAL (raw ML) TriageResult is not modified — callers must keep
         both to show them side by side.

    Weighted score formula
    ----------------------
    P_norm  = (probative_value - 1) / 2           → [0, 1]
    D_norm  = (perishability - 1) / 2             → [0, 1]
    E_norm  = (exclusionary_power - 1) / 2        → [0, 1]
    S_inv   = 1 - min(testing_lead_time, MAX) / MAX → [0, 1] (fast = high)

    weighted_score = (P_norm * wP + D_norm * wD + E_norm * wE + S_inv * wS)
                     / (wP + wD + wE + wS)

    Tier is then re-derived from POLICY_TIER_THRESHOLDS (documented above).

    DISCLAIMER: This formula and these thresholds are a PROTOTYPE MECHANISM.
    They are not scientifically validated or legally endorsed.

    Parameters
    ----------
    result : The raw TriageResult from predict_priority().
    item   : The EvidenceItem that produced the result.
    policy : The active TriagePolicy, or None.

    Returns
    -------
    If policy is None → the original result unchanged.
    If policy is active → a new TriageResult with adjusted tier/score/policy_version.
    """
    if policy is None:
        return result

    # Normalise P, D, E to [0, 1] from ordinal 1–3
    p_norm = (item.probative_value   - 1) / 2.0
    d_norm = (item.perishability     - 1) / 2.0
    e_norm = (item.exclusionary_power - 1) / 2.0

    # S is inverse of testing_lead_time: shorter lead = higher actionability
    s_inv = 1.0 - min(item.testing_lead_time, LEAD_TIME_MAX_DAYS) / LEAD_TIME_MAX_DAYS

    total_weight = policy.weight_P + policy.weight_D + policy.weight_E + policy.weight_S
    if total_weight == 0:
        # Degenerate policy — all weights zero; return original result
        return result

    weighted_score = (
        p_norm * policy.weight_P
        + d_norm * policy.weight_D
        + e_norm * policy.weight_E
        + s_inv  * policy.weight_S
    ) / total_weight

    weighted_score = max(0.0, min(1.0, weighted_score))  # clamp to [0, 1]

    # Derive tier from documented thresholds
    new_tier = "Low"
    for threshold, tier_label in POLICY_TIER_THRESHOLDS:
        if weighted_score >= threshold:
            new_tier = tier_label
            break

    # Import here to avoid circular dependency at module level
    from dataclasses import replace as dc_replace
    adjusted = dc_replace(
        result,
        priority_tier=new_tier,
        priority_score=round(weighted_score, 4),
        policy_version=policy.version,
    )
    return adjusted


# ---------------------------------------------------------------------------
# POLICY SUMMARY (for reports and UI)
# ---------------------------------------------------------------------------

def policy_summary(policy: TriagePolicy | None) -> str:
    """
    Return a one-line human-readable policy status string for reports and UI.

    Examples
    --------
    "No active policy (unweighted ML)"
    "Active policy: Homicide Focus Policy v1.0.0 (P=2.0, D=1.5, E=1.0, S=1.0)"
    """
    if policy is None:
        return "No active policy (unweighted ML)"
    return (
        f"Active policy: {policy.label} v{policy.version} "
        f"(P={policy.weight_P}, D={policy.weight_D}, "
        f"E={policy.weight_E}, S={policy.weight_S})"
    )
