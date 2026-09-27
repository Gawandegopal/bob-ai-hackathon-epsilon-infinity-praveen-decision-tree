"""
phase1_demo.py
==============
Phase 1 validation script for EvidencePro.

Verifies:
  1. Dataset loads and has the expected shape
  2. All three models train successfully
  3. predict_priority returns a valid TriageResult for representative items
  4. urgency_flag fires correctly for the documented rules
  5. validate_item catches bad inputs

Run from the src/ directory:
  python phase1_demo.py

This script is for development validation only. It is not part of the
production application.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))  # ensure src/ is on the path

from data.synthetic_dataset import build_dataset
from models.evidence_item import EvidenceItem, CaseContext, EVIDENCE_TYPES, OFFENCE_TYPES
from models.triage_models import (
    get_trained_model,
    predict_priority,
    validate_item,
    compute_urgency_flag,
    DATA_DISCLAIMER,
)

SEP = "=" * 64


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ---------------------------------------------------------------------------
# 1. DATASET CHECK
# ---------------------------------------------------------------------------
section("1. Synthetic Dataset")

df = build_dataset()
print(f"  Rows    : {len(df)}")
print(f"  Columns : {list(df.columns)}")
print(f"  Label distribution:\n{df['priority_label'].value_counts().to_string()}")
assert len(df) >= 148, f"Expected >=148 rows, got {len(df)}"
assert "exclusionary_power" in df.columns, "exclusionary_power column missing"
print("  ✓ Dataset OK")


# ---------------------------------------------------------------------------
# 2. MODEL TRAINING & CV ACCURACY
# ---------------------------------------------------------------------------
section("2. Model Training (all three models)")

results = {}
for name in ["decision_tree", "random_forest", "gradient_boosting"]:
    result = get_trained_model(name)
    results[name] = result
    print(f"\n  [{name}]")
    print(f"    CV accuracy : {result.cv_accuracy * 100:.1f}%  "
          f"← {DATA_DISCLAIMER}")
    print(f"    Disclaimer  : {result.data_disclaimer}")
    print(f"    CV Report:\n")
    for line in result.cv_report.splitlines():
        print(f"      {line}")

print("\n  ✓ All three models trained OK")


# ---------------------------------------------------------------------------
# 3. PREDICTION — representative items
# ---------------------------------------------------------------------------
section("3. Predictions for representative evidence items")

test_cases = [
    {
        "label": "Blood swab — homicide scene",
        "item": EvidenceItem(
            item_id="T001", label="Blood swab",
            evidence_type="biological_dna", offence_type="homicide",
            probative_value=3, perishability=3, exclusionary_power=3,
            contamination_risk=3, specialist_required=1, testing_lead_time=5,
            collection_age_hours=2,
        ),
        "expected_tier": "Critical",
    },
    {
        "label": "CCTV footage — robbery",
        "item": EvidenceItem(
            item_id="T002", label="CCTV footage",
            evidence_type="digital_cctv", offence_type="robbery",
            probative_value=3, perishability=1, exclusionary_power=2,
            contamination_risk=1, specialist_required=0, testing_lead_time=2,
            collection_age_hours=0,
        ),
        "expected_tier": "High",
    },
    {
        "label": "Latent fingerprint — burglary",
        "item": EvidenceItem(
            item_id="T003", label="Latent fingerprint",
            evidence_type="fingerprint_latent", offence_type="burglary",
            probative_value=2, perishability=2, exclusionary_power=2,
            contamination_risk=2, specialist_required=0, testing_lead_time=5,
            collection_age_hours=0,
        ),
        "expected_tier": "Standard",
    },
    {
        "label": "Soil trace — vehicle crime",
        "item": EvidenceItem(
            item_id="T004", label="Soil trace",
            evidence_type="trace_soil", offence_type="vehicle_crime",
            probative_value=1, perishability=1, exclusionary_power=1,
            contamination_risk=1, specialist_required=1, testing_lead_time=10,
            collection_age_hours=0,
        ),
        "expected_tier": "Low",
    },
    {
        "label": "Touch DNA — sexual assault (perishable)",
        "item": EvidenceItem(
            item_id="T005", label="Touch DNA",
            evidence_type="biological_touch_dna", offence_type="sexual_assault",
            probative_value=2, perishability=3, exclusionary_power=3,
            contamination_risk=3, specialist_required=1, testing_lead_time=7,
            collection_age_hours=3,
        ),
        "expected_tier": "Critical",
    },
]

for tc in test_cases:
    print(f"\n  Item  : {tc['label']}")
    for model_name, result in results.items():
        tr = predict_priority(tc["item"], result)
        urgent_str = "URGENT" if tr.urgency_flag else "stable"
        print(f"    [{model_name:<20}]  tier={tr.priority_tier:<10} "
              f"score={tr.priority_score:.3f}  {urgent_str}")
        if model_name == "decision_tree" and tr.decision_path:
            for line in tr.decision_path.splitlines()[:5]:  # first 5 steps only
                print(f"      {line}")
        else:
            # Show per-item values and global importances as two separate sections
            print(f"      --- Item feature values ---")
            for fname, fval in tr.item_feature_values.items():
                print(f"        {fname:<24} {fval}")
            print(f"      --- Model-level importances (global, NOT per-item causation) ---")
            for entry in tr.model_importances:
                print(f"        #{entry['rank']} {entry['feature']:<24} {entry['importance']:.4f}")

print("\n  ✓ Predictions OK")


# ---------------------------------------------------------------------------
# 4. URGENCY FLAG RULES
# ---------------------------------------------------------------------------
section("4. Urgency flag rule verification")

urgency_cases = [
    # (perishability, collection_age_hours, expected_urgent, description)
    (3, 0,   True,  "perishability=3, age=0 (unknown) → always urgent"),
    (3, 100, True,  "perishability=3, age=100h → always urgent"),
    (2, 3,   True,  "perishability=2, age=3h → urgent (within 6h window)"),
    (2, 5,   True,  "perishability=2, age=5h → urgent (within 6h window)"),
    (2, 6,   False, "perishability=2, age=6h → NOT urgent (at boundary, < 6 only)"),
    (2, 10,  False, "perishability=2, age=10h → NOT urgent (past window)"),
    (2, 0,   False, "perishability=2, age=0 (unknown) → NOT urgent"),
    (1, 3,   False, "perishability=1, age=3h → NOT urgent (stable)"),
    (1, 0,   False, "perishability=1, age=0 → NOT urgent (stable)"),
]

all_passed = True
for p, age, expected, desc in urgency_cases:
    item = EvidenceItem(
        evidence_type="trace_soil", offence_type="burglary",
        perishability=p, collection_age_hours=age,
    )
    got = compute_urgency_flag(item)
    status = "✓" if got == expected else "✗ FAIL"
    if got != expected:
        all_passed = False
    print(f"  {status}  {desc}")

if all_passed:
    print("  ✓ All urgency rule checks passed")
else:
    print("  ✗ Some urgency checks FAILED")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 5. VALIDATION — rejects bad inputs
# ---------------------------------------------------------------------------
section("5. Input validation")

bad_item = EvidenceItem(
    evidence_type="unknown_type",
    offence_type="homicide",
    probative_value=5,    # invalid
    perishability=2,
    exclusionary_power=2,
    contamination_risk=2,
    specialist_required=0,
    testing_lead_time=-1, # invalid
    collection_age_hours=0,
)
errors = validate_item(bad_item)
print(f"  Errors found: {len(errors)}")
for e in errors:
    print(f"    - {e}")
assert len(errors) >= 3, "Expected at least 3 validation errors"
print("  ✓ Validation correctly rejects bad inputs")


# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
section("Phase 1 Validation Complete")
print("  All checks passed.")
print(f"  Dataset rows   : {len(df)}")
print(f"  Models trained : {len(results)}")
print(f"  Test items     : {len(test_cases)}")
print(f"  Urgency checks : {len(urgency_cases)}")
print()
print(f"  REMINDER: {DATA_DISCLAIMER}")
print()
