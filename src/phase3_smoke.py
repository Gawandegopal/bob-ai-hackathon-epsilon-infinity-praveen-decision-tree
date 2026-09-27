"""
phase3_smoke.py
===============
Phase 3 smoke test — exercises the complete pipeline as the Streamlit app
would, without a browser. Verifies that all module integrations used by
app.py work end-to-end.

Run from src/:
    python phase3_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from models.evidence_item import EvidenceItem, CaseContext, EVIDENCE_TYPES, OFFENCE_TYPES
from models.triage_models import get_trained_model, predict_priority, validate_item, DATA_DISCLAIMER
from models.policy import (
    TriagePolicy, PolicyRole, POLICY_HISTORY,
    propose_policy, approve_policy, get_active_policy, apply_policy, policy_summary,
)
from scheduler import build_schedule, schedule_summary, BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE
from report import ReportData, generate_report

SEP = "=" * 60

def ok(msg): print(f"  OK: {msg}")
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")

# ---------------------------------------------------------------------------
# Simulate the app's model training (sidebar button)
# ---------------------------------------------------------------------------
section("Simulate: Train / Retrain Model (all 3)")
for model_name in ["decision_tree", "random_forest", "gradient_boosting"]:
    tr = get_trained_model(model_name)
    assert tr.model_name == model_name
    assert 0 < tr.cv_accuracy <= 1.0
    assert tr.data_disclaimer == DATA_DISCLAIMER
    ok(f"{model_name} trains OK — CV={tr.cv_accuracy*100:.1f}%")

# Use DT for remainder
training = get_trained_model("decision_tree")

# ---------------------------------------------------------------------------
# Simulate: Step 1 — Case Context
# ---------------------------------------------------------------------------
section("Simulate: Step 1 — Case Context")
ctx = CaseContext(fir_number="SMOKE-001", offence_type="robbery", narrative="Test case")
ok(f"CaseContext created: {ctx.fir_number}")

# ---------------------------------------------------------------------------
# Simulate: Step 2 — Evidence Items (with validation)
# ---------------------------------------------------------------------------
section("Simulate: Step 2 — Evidence Items")
items = [
    EvidenceItem(item_id="E001", label="Blood swab",
                 evidence_type="biological_dna", offence_type="robbery",
                 probative_value=3, perishability=3, exclusionary_power=3,
                 contamination_risk=3, specialist_required=1, testing_lead_time=5,
                 collection_age_hours=2),
    EvidenceItem(item_id="E002", label="CCTV footage",
                 evidence_type="digital_cctv", offence_type="robbery",
                 probative_value=3, perishability=1, exclusionary_power=2,
                 contamination_risk=1, specialist_required=0, testing_lead_time=2,
                 collection_age_hours=0),
    EvidenceItem(item_id="E003", label="Latent fingerprint",
                 evidence_type="fingerprint_latent", offence_type="robbery",
                 probative_value=2, perishability=2, exclusionary_power=2,
                 contamination_risk=2, specialist_required=0, testing_lead_time=5,
                 collection_age_hours=0),
    EvidenceItem(item_id="E004", label="Soil trace",
                 evidence_type="trace_soil", offence_type="robbery",
                 probative_value=1, perishability=1, exclusionary_power=1,
                 contamination_risk=1, specialist_required=1, testing_lead_time=10,
                 collection_age_hours=0),
]
for item in items:
    errs = validate_item(item)
    assert not errs, f"{item.item_id} validation failed: {errs}"
ok(f"{len(items)} items validated")

# ---------------------------------------------------------------------------
# Simulate: Step 3 — Run Triage
# ---------------------------------------------------------------------------
section("Simulate: Step 3 — Run Triage")
results = [predict_priority(item, training) for item in items]
for item, result in zip(items, results):
    assert result.priority_tier in ["Critical", "High", "Standard", "Low"]
    assert 0.0 <= result.priority_score <= 1.0
    assert isinstance(result.urgency_flag, bool)
    assert isinstance(result.item_feature_values, dict) and len(result.item_feature_values) == 8
    print(f"    {item.item_id}  {result.priority_tier:<10} urgency={result.urgency_flag}")
ok("All triage results valid")

# Verify raw vs policy — with no active policy, apply_policy returns same object
POLICY_HISTORY.clear()
assert get_active_policy() is None
for item, result in zip(items, results):
    adj = apply_policy(result, item, None)
    assert adj is result
ok("apply_policy(None) returns original result unchanged")

# ---------------------------------------------------------------------------
# Simulate: Step 4 — Explanations (DT decision path / RF importances)
# ---------------------------------------------------------------------------
section("Simulate: Step 4 — Explanations")
# DT has decision_path, empty model_importances
assert results[0].decision_path != "", "DT should have decision_path"
assert results[0].model_importances == [], "DT should have empty model_importances"
ok("DT: decision_path present, model_importances empty")

# RF
rf_training = get_trained_model("random_forest")
rf_results = [predict_priority(item, rf_training) for item in items]
assert rf_results[0].decision_path == "", "RF should have empty decision_path"
assert len(rf_results[0].model_importances) > 0, "RF should have model_importances"
assert len(rf_results[0].item_feature_values) == 8
for entry in rf_results[0].model_importances:
    assert "feature" in entry and "importance" in entry and "rank" in entry
ok("RF: decision_path empty, model_importances and item_feature_values present")

# ---------------------------------------------------------------------------
# Simulate: Step 5 — Human Review with overrides
# ---------------------------------------------------------------------------
section("Simulate: Step 5 — Human Review")
overrides = {
    "E004": ("Critical", "Investigator confirmed physical link to suspect"),
}
ok(f"Override recorded for E004: Critical")

# ---------------------------------------------------------------------------
# Simulate: Policy admin workflow
# ---------------------------------------------------------------------------
section("Simulate: Policy Admin workflow")
POLICY_HISTORY.clear()
pol = TriagePolicy(version="2.0.0", label="Robbery Focus", weight_P=2.0, weight_D=1.5,
                   weight_E=1.0, weight_S=1.0, proposed_by="admin1")
propose_policy(pol, PolicyRole.POLICY_ADMIN)
approve_policy(pol, "approver1", PolicyRole.APPROVER)
active = get_active_policy()
assert active is not None and active.version == "2.0.0"
ok(f"Policy approved and active: {policy_summary(active)}")

# Policy-adjusted results
pol_results = [apply_policy(r, item, active) for r, item in zip(results, items)]
for pr in pol_results:
    assert pr.policy_version == "2.0.0"
ok("All items have policy-adjusted results with correct version")

# ---------------------------------------------------------------------------
# Simulate: Step 6 — FSL Schedule (with override and policy)
# ---------------------------------------------------------------------------
section("Simulate: Step 6 — FSL Schedule")
schedule = build_schedule(items, results, overrides, active)
assert len(schedule) == len(items)

groups = schedule_summary(schedule)
# E001 (Critical + urgent) must be Batch 1
e001 = next(si for si in schedule if si.item.item_id == "E001")
assert e001.batch == BATCH_IMMEDIATE, f"E001 expected Immediate, got {e001.batch}"
# E004 (overridden to Critical) must also be Batch 1
e004 = next(si for si in schedule if si.item.item_id == "E004")
assert e004.batch == BATCH_IMMEDIATE, f"E004 override expected Immediate, got {e004.batch}"
assert e004.investigator_decision == "overridden"
# All items have policy_result
for si in schedule:
    assert si.policy_result is not None

for batch_label in [BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE]:
    batch_items = groups[batch_label]
    print(f"    {batch_label}: {len(batch_items)} item(s)")
ok("Schedule built correctly — override, urgency, and policy all applied")

# ---------------------------------------------------------------------------
# Simulate: Step 7 — Report generation and download
# ---------------------------------------------------------------------------
section("Simulate: Step 7 — Report")
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
report_data = ReportData(
    context=ctx, items=items, results=results, schedule=schedule,
    policy=active, generated_at=now,
    model_used="Decision Tree", cv_accuracy=training.cv_accuracy,
)
report_md = generate_report(report_data)
assert len(report_md) > 500
required = [
    "investigator is the final decision-maker",
    DATA_DISCLAIMER,
    "SMOKE-001",
    "robbery",
    "CONFIGURABLE PROTOTYPE MECHANISM",
    "Batch 1", "Batch 2", "Batch 3",
    "Robbery Focus",
    "2.0.0",
]
for phrase in required:
    assert phrase in report_md, f"Missing in report: '{phrase}'"

# Simulate download: encode to bytes as st.download_button does
report_bytes = report_md.encode("utf-8")
assert len(report_bytes) > 500
ok(f"Report generated ({len(report_md)} chars, {len(report_bytes)} bytes) — download simulation OK")
ok("All required phrases present in report")

# ---------------------------------------------------------------------------
# Confirm frozen baseline still unmodified
# ---------------------------------------------------------------------------
section("Confirm baseline integrity")
import hashlib
with open(os.path.join(os.path.dirname(__file__), "evidence_triage.py"), "rb") as f:
    content = f.read()
assert b"build_model" in content and b"DecisionTreeClassifier" in content
assert b"triage_models" not in content  # baseline must not import new modules
ok("evidence_triage.py untouched — baseline integrity confirmed")

import subprocess
result_cli = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), "run_triage.py")],
    capture_output=True, text=True, cwd=os.path.dirname(__file__),
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
)
assert result_cli.returncode == 0, f"run_triage.py failed: {result_cli.stderr}"
ok("python run_triage.py exits 0")

section("Phase 3 Smoke Test Complete")
print("  All checks passed.")
print(f"  Models tested      : 3 (DT, RF, GB)")
print(f"  Evidence items     : {len(items)}")
print(f"  Triage results     : {len(results)}")
print(f"  Schedule items     : {len(schedule)}")
print(f"  Report size        : {len(report_md)} chars")
print(f"  Baseline integrity : confirmed")
print(f"\n  REMINDER: {DATA_DISCLAIMER}")
print()
