"""
phase4_demo.py
==============
EvidencePro — Phase 4 validation script.

Verifies that extractor.py works correctly across three evidence description
types (biological, digital, trace). Runs in heuristic mode only — no API key
required. All checks must pass for Phase 4 to be considered complete.

Usage:
    python src/phase4_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from extractor import (
    extract_features,
    build_extraction_prompt,
    is_watsonx_available,
    extraction_mode_label,
    EXTRACTABLE_FIELDS,
)
from models.evidence_item import CaseContext

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

def check(label: str, condition: bool):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    results.append(condition)

# ---------------------------------------------------------------------------
# Test context
# ---------------------------------------------------------------------------
ctx = CaseContext(
    fir_number="DEMO-001",
    offence_type="homicide",
    narrative="Victim found at scene with signs of struggle.",
)

print("\n=== Phase 4 — extractor.py validation ===\n")

# ---------------------------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------------------------
print("1. Utility functions")
check("is_watsonx_available() returns bool",   isinstance(is_watsonx_available(), bool))
check("extraction_mode_label() returns str",   isinstance(extraction_mode_label(), str))
check("EXTRACTABLE_FIELDS is a non-empty list", isinstance(EXTRACTABLE_FIELDS, list) and len(EXTRACTABLE_FIELDS) > 0)
check("'priority_tier' not in EXTRACTABLE_FIELDS", "priority_tier" not in EXTRACTABLE_FIELDS)

# ---------------------------------------------------------------------------
# 2. Biological evidence — blood swab
# ---------------------------------------------------------------------------
print("\n2. Biological evidence (blood swab)")
r_bio = extract_features("Blood swab from victim's clothing, collected at scene", ctx)
check("returns dict",                          isinstance(r_bio, dict))
check("_source key present",                   "_source" in r_bio)
check("evidence_type == biological_dna",       r_bio.get("evidence_type") == "biological_dna")
check("perishability == 3 (high degradation)", r_bio.get("perishability") == 3)
check("specialist_required == 1",              r_bio.get("specialist_required") == 1)

# ---------------------------------------------------------------------------
# 3. Digital evidence — CCTV footage
# ---------------------------------------------------------------------------
print("\n3. Digital evidence (CCTV footage)")
r_dig = extract_features("CCTV footage from camera near the entrance", ctx)
check("returns dict",                          isinstance(r_dig, dict))
check("evidence_type == digital_cctv",         r_dig.get("evidence_type") == "digital_cctv")
check("perishability == 1 (stable)",           r_dig.get("perishability") == 1)

# ---------------------------------------------------------------------------
# 4. Trace evidence — glass fragments
# ---------------------------------------------------------------------------
print("\n4. Trace evidence (glass fragments)")
r_trace = extract_features("Broken glass fragments found at entry point", ctx)
check("returns dict",                          isinstance(r_trace, dict))
check("evidence_type == trace_glass",          r_trace.get("evidence_type") == "trace_glass")

# ---------------------------------------------------------------------------
# 5. Empty description — graceful fallback
# ---------------------------------------------------------------------------
print("\n5. Empty description graceful fallback")
r_empty = extract_features("", ctx)
check("returns dict for empty input",          isinstance(r_empty, dict))
check("_source == 'none' for empty input",     r_empty.get("_source") == "none")

# ---------------------------------------------------------------------------
# 6. Prompt builder
# ---------------------------------------------------------------------------
print("\n6. build_extraction_prompt()")
prompt = build_extraction_prompt("blood swab", ctx)
check("prompt is non-empty string",            isinstance(prompt, str) and len(prompt) > 50)
check("prompt contains JSON instruction",      "JSON" in prompt)
check("prompt contains evidence_type field",   "evidence_type" in prompt)
check("prompt does NOT mention priority_tier", "priority_tier" not in prompt)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 44)
passed = sum(results)
total  = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  [ALL PASS] Phase 4 validation PASSED")
else:
    print("  [FAILURE]  Phase 4 validation FAILED -- see above")
    sys.exit(1)
