"""
phase2_demo.py
==============
Phase 2 validation script for EvidencePro.

Verifies the complete end-to-end pipeline:
  EvidenceItem + CaseContext
  → ML triage (Decision Tree)
  → Optional policy adjustment
  → FSL scheduling (with and without overrides)
  → Markdown report generation

Also verifies:
  - Policy governance (propose / approve / reject roles)
  - Policy None default (raw ML used)
  - Investigator override changes batch assignment
  - Report contains required sections and disclaimers
  - Scheduler is deterministic (two identical calls return identical output)

Run from the src/ directory:
    python phase2_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone

from models.evidence_item import EvidenceItem, CaseContext
from models.triage_models import get_trained_model, predict_priority, DATA_DISCLAIMER
from models.policy import (
    TriagePolicy, PolicyRole,
    POLICY_HISTORY, DEFAULT_POLICY,
    propose_policy, approve_policy, reject_policy, get_active_policy,
    apply_policy, policy_summary,
)
from scheduler import build_schedule, schedule_summary, BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE
from report import ReportData, generate_report

SEP = "=" * 64

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONTEXT = CaseContext(
    fir_number="FIR-2024-001",
    offence_type="homicide",
    narrative="Victim found at residential address. Multiple items collected from scene.",
)

ITEMS = [
    EvidenceItem(
        item_id="E001", label="Blood swab — victim's clothing",
        evidence_type="biological_dna", offence_type="homicide",
        probative_value=3, perishability=3, exclusionary_power=3,
        contamination_risk=3, specialist_required=1, testing_lead_time=5,
        collection_age_hours=3,
    ),
    EvidenceItem(
        item_id="E002", label="Latent fingerprint — door frame",
        evidence_type="fingerprint_latent", offence_type="homicide",
        probative_value=2, perishability=2, exclusionary_power=2,
        contamination_risk=2, specialist_required=0, testing_lead_time=5,
        collection_age_hours=0,
    ),
    EvidenceItem(
        item_id="E003", label="CCTV footage — street camera",
        evidence_type="digital_cctv", offence_type="homicide",
        probative_value=3, perishability=1, exclusionary_power=2,
        contamination_risk=1, specialist_required=0, testing_lead_time=2,
        collection_age_hours=0,
    ),
    EvidenceItem(
        item_id="E004", label="Soil trace — suspect's boot",
        evidence_type="trace_soil", offence_type="homicide",
        probative_value=1, perishability=1, exclusionary_power=1,
        contamination_risk=1, specialist_required=1, testing_lead_time=10,
        collection_age_hours=0,
    ),
    EvidenceItem(
        item_id="E005", label="Touch DNA — window handle",
        evidence_type="biological_touch_dna", offence_type="homicide",
        probative_value=2, perishability=3, exclusionary_power=3,
        contamination_risk=3, specialist_required=1, testing_lead_time=7,
        collection_age_hours=2,
    ),
]

# Train one model (DT for speed in validation)
section("Training model (Decision Tree)")
training = get_trained_model("decision_tree")
print(f"  CV accuracy: {training.cv_accuracy * 100:.1f}% ({DATA_DISCLAIMER})")

# Run triage for all items
RESULTS = [predict_priority(item, training) for item in ITEMS]
for item, result in zip(ITEMS, RESULTS):
    print(f"  {item.item_id}  {item.label[:40]:<40}  -> {result.priority_tier}  urgency={result.urgency_flag}")

print("  OK: all items triaged")

# ---------------------------------------------------------------------------
# 1. Policy governance
# ---------------------------------------------------------------------------
section("1. Policy Governance")

# Clear any residual state
POLICY_HISTORY.clear()

# a) Default is None
assert get_active_policy() is None, "Expected no active policy initially"
print("  OK: default policy is None")

# b) Investigator cannot propose
inv_policy = TriagePolicy(version="0.1.0", label="Test", proposed_by="investigator1")
try:
    propose_policy(inv_policy, PolicyRole.INVESTIGATOR)
    print("  FAIL: should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"  OK: Investigator cannot propose — {e}")

# c) Policy Admin can propose
admin_policy = TriagePolicy(
    version="1.0.0",
    label="Homicide Focus",
    description="Emphasises biological evidence degradation in homicide cases.",
    weight_P=2.0, weight_D=2.0, weight_E=1.0, weight_S=1.0,
    proposed_by="admin1",
)
propose_policy(admin_policy, PolicyRole.POLICY_ADMIN)
assert admin_policy.status == "proposed"
assert get_active_policy() is None, "Proposed policy should not yet be active"
print(f"  OK: Policy Admin proposed '{admin_policy.label}' — status: {admin_policy.status}")

# d) Investigator cannot approve
try:
    approve_policy(admin_policy, "investigator1", PolicyRole.INVESTIGATOR)
    print("  FAIL: should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"  OK: Investigator cannot approve — {e}")

# e) Approver approves
approve_policy(admin_policy, "approver1", PolicyRole.APPROVER)
assert admin_policy.status == "approved"
assert admin_policy.approved_by == "approver1"
active = get_active_policy()
assert active is not None and active.version == "1.0.0"
print(f"  OK: Approver approved — active policy: {policy_summary(active)}")

# f) Second proposal and rejection
policy_v2 = TriagePolicy(
    version="1.1.0", label="Balanced", weight_P=1.0, weight_D=1.0, weight_E=1.0, weight_S=1.0,
    proposed_by="admin1",
)
propose_policy(policy_v2, PolicyRole.POLICY_ADMIN)
reject_policy(policy_v2, "approver1", PolicyRole.APPROVER)
assert policy_v2.status == "rejected"
# Active policy should still be 1.0.0
assert get_active_policy().version == "1.0.0"
print(f"  OK: Rejected policy does not become active — still v{get_active_policy().version}")

# ---------------------------------------------------------------------------
# 2. apply_policy — raw vs adjusted
# ---------------------------------------------------------------------------
section("2. Policy Application (raw vs adjusted)")

blood_swab = ITEMS[0]
raw_result = RESULTS[0]

# With no policy
adj_none = apply_policy(raw_result, blood_swab, None)
assert adj_none is raw_result, "apply_policy(None) must return the original result unchanged"
print("  OK: apply_policy(None) returns original result unchanged")

# With active policy
active_pol = get_active_policy()
adj_pol = apply_policy(raw_result, blood_swab, active_pol)
assert adj_pol is not raw_result, "apply_policy should return a new TriageResult"
assert adj_pol.policy_version == "1.0.0"
print(f"  Raw tier: {raw_result.priority_tier}  score={raw_result.priority_score}")
print(f"  Adjusted: {adj_pol.priority_tier}  score={adj_pol.priority_score}  policy_version={adj_pol.policy_version}")
print("  OK: apply_policy returns a new TriageResult with policy_version set")

# ---------------------------------------------------------------------------
# 3. FSL Scheduler — no overrides, no policy
# ---------------------------------------------------------------------------
section("3. FSL Scheduler — no overrides, no policy")

POLICY_HISTORY.clear()
schedule_no_policy = build_schedule(ITEMS, RESULTS, overrides={}, policy=None)

assert len(schedule_no_policy) == len(ITEMS), "Schedule must contain one entry per item"

print("  Batch assignment (no policy, no overrides):")
for si in schedule_no_policy:
    print(f"    {si.item.item_id}  {si.item.label[:35]:<35}  {si.batch}  final={si.final_tier}  urgency={si.triage_result.urgency_flag}")

# E001 (Critical, urgent) must be Batch 1
e001 = next(si for si in schedule_no_policy if si.item.item_id == "E001")
assert e001.batch == BATCH_IMMEDIATE, f"E001 expected Batch 1, got {e001.batch}"

# E004 (Low, stable) must be Batch 3 or Batch 2 (specialist=1 → Batch 2)
e004 = next(si for si in schedule_no_policy if si.item.item_id == "E004")
print(f"  E004 tier={e004.final_tier} specialist={e004.item.specialist_required} batch={e004.batch}")

# Sort: within Batch 1, perishability DESC then lead_time ASC
batch1_items = [si for si in schedule_no_policy if si.batch == BATCH_IMMEDIATE]
for i in range(len(batch1_items) - 1):
    a, b = batch1_items[i], batch1_items[i + 1]
    assert (a.item.perishability, -a.item.testing_lead_time) >= (b.item.perishability, -b.item.testing_lead_time) or \
           a.item.perishability > b.item.perishability or \
           (a.item.perishability == b.item.perishability and a.item.testing_lead_time <= b.item.testing_lead_time), \
        f"Sort order violation: {a.item.item_id} before {b.item.item_id}"

print("  OK: Scheduler correct (no policy, no overrides)")

# ---------------------------------------------------------------------------
# 4. FSL Scheduler — with overrides
# ---------------------------------------------------------------------------
section("4. FSL Scheduler — with investigator overrides")

# Override E004 (would be Low/Standard) to Critical
overrides = {"E004": ("Critical", "Physical evidence connects directly to suspect — promoted to Critical.")}
schedule_with_override = build_schedule(ITEMS, RESULTS, overrides=overrides, policy=None)

e004_ov = next(si for si in schedule_with_override if si.item.item_id == "E004")
assert e004_ov.final_tier == "Critical", f"Expected Critical after override, got {e004_ov.final_tier}"
assert e004_ov.batch == BATCH_IMMEDIATE, f"Expected Batch 1 after override, got {e004_ov.batch}"
assert e004_ov.investigator_decision == "overridden"
assert "Physical evidence" in e004_ov.override_reason
print(f"  E004 overridden: final_tier={e004_ov.final_tier}  batch={e004_ov.batch}")
print(f"  Override reason: {e004_ov.override_reason}")
print("  OK: Override correctly promotes item to Batch 1")

# Non-overridden items unaffected
e001_ov = next(si for si in schedule_with_override if si.item.item_id == "E001")
assert e001_ov.investigator_decision == "accepted"
assert e001_ov.override_reason is None
print("  OK: Non-overridden items marked 'accepted' with no override_reason")

# ---------------------------------------------------------------------------
# 5. FSL Scheduler — with active policy
# ---------------------------------------------------------------------------
section("5. FSL Scheduler — with active policy")

# Re-add approved policy
POLICY_HISTORY.clear()
pol = TriagePolicy(version="1.0.0", label="Homicide Focus",
                   weight_P=2.0, weight_D=2.0, weight_E=1.0, weight_S=1.0,
                   proposed_by="admin1")
propose_policy(pol, PolicyRole.POLICY_ADMIN)
approve_policy(pol, "approver1", PolicyRole.APPROVER)
active_pol = get_active_policy()

schedule_with_policy = build_schedule(ITEMS, RESULTS, overrides={}, policy=active_pol)

print("  Batch assignment (with policy):")
for si in schedule_with_policy:
    raw_t = si.triage_result.priority_tier
    pol_t = si.policy_result.priority_tier if si.policy_result else "—"
    print(f"    {si.item.item_id}  raw={raw_t:<10} policy={pol_t:<10} final={si.final_tier:<10} {si.batch}")

# Verify all items have a policy_result
for si in schedule_with_policy:
    assert si.policy_result is not None, f"{si.item.item_id} missing policy_result"
    assert si.policy_result.policy_version == "1.0.0", f"{si.item.item_id} policy_version mismatch"

print("  OK: All items have policy_result with correct version")

# Scheduler determinism: identical call must return identical batches
schedule2 = build_schedule(ITEMS, RESULTS, overrides={}, policy=active_pol)
for s1, s2 in zip(schedule_with_policy, schedule2):
    assert s1.batch == s2.batch, "Scheduler is not deterministic"
print("  OK: Scheduler is deterministic (two identical calls give identical output)")

# ---------------------------------------------------------------------------
# 6. Report generation
# ---------------------------------------------------------------------------
section("6. Report Generation")

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
report_data = ReportData(
    context=CONTEXT,
    items=ITEMS,
    results=RESULTS,
    schedule=schedule_with_policy,
    policy=active_pol,
    generated_at=now,
    model_used="Decision Tree",
    cv_accuracy=training.cv_accuracy,
)

report_md = generate_report(report_data)

assert isinstance(report_md, str) and len(report_md) > 500, "Report is too short"
print(f"  Report length: {len(report_md)} characters")

# Required content checks
required_phrases = [
    "investigator is the final decision-maker",
    DATA_DISCLAIMER,
    "FIR-2024-001",
    "homicide",
    "Evidence Inventory",
    "Evidence Classification",
    "Priority Explanations",
    "FSL Examination Schedule",
    "Investigator Review Decisions",
    "Model and Policy Information",
    "CONFIGURABLE PROTOTYPE MECHANISM",
    "Batch 1",
    "Batch 2",
    "Batch 3",
    "Important Notices",
    "Homicide Focus",   # active policy label
    "1.0.0",            # policy version
]

all_ok = True
for phrase in required_phrases:
    if phrase not in report_md:
        print(f"  MISSING in report: '{phrase}'")
        all_ok = False
    else:
        print(f"  OK: found '{phrase[:60]}'")

assert all_ok, "Report is missing required content"
print("  OK: Report contains all required sections and disclaimers")

# Save report to file for inspection
report_path = os.path.join(os.path.dirname(__file__), "phase2_report_sample.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"  Report saved to: {report_path}")

# ---------------------------------------------------------------------------
# 7. No-policy report (verify no policy artefacts leak into the report)
# ---------------------------------------------------------------------------
section("7. No-policy report")

schedule_nopol = build_schedule(ITEMS, RESULTS, overrides={}, policy=None)
report_nopol = generate_report(ReportData(
    context=CONTEXT, items=ITEMS, results=RESULTS,
    schedule=schedule_nopol, policy=None,
    generated_at=now, model_used="Decision Tree",
    cv_accuracy=training.cv_accuracy,
))

assert "No active policy" in report_nopol, "Expected 'No active policy' in no-policy report"
assert "1.0.0" not in report_nopol, "Policy version should not appear in no-policy report"
print("  OK: No-policy report correctly omits policy details")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
section("Phase 2 Validation Complete")
print("  All checks passed.")
print(f"  Items triaged      : {len(ITEMS)}")
print(f"  Policy checks      : propose / approve / reject / get_active all verified")
print(f"  Scheduler checks   : no-policy, with-override, with-policy, determinism")
print(f"  Report checks      : {len(required_phrases)} required phrases verified")
print(f"\n  REMINDER: {DATA_DISCLAIMER}")
print()
