"""
run_triage.py
=============
Forensic Evidence Triage – Entry Point
---------------------------------------
IMPORTANT: This prototype uses SYNTHETIC / CURATED DEMONSTRATION DATA only.
Do not use for real forensic decisions.

Run with:
    python run_triage.py

To test your own evidence item, add a dict to the TEST_ITEMS list at the
bottom of this file, or call triage_item() directly from your own script.
"""

from evidence_triage import (
    build_model,
    predict_priority,
    explain_decision,
    EVIDENCE_TYPES,
    OFFENCE_TYPES,
    PRIORITY_LABELS,
)

# ---------------------------------------------------------------------------
# TEST EVIDENCE ITEMS
# ---------------------------------------------------------------------------
# Add or modify items here to test new scenarios.
#
# Required keys and allowed values:
#   evidence_type       : see EVIDENCE_TYPES in evidence_triage.py
#   offence_type        : see OFFENCE_TYPES  in evidence_triage.py
#   probative_value     : 1 (low), 2 (medium), 3 (high)
#   perishability       : 1 (stable), 2 (days), 3 (hours)
#   contamination_risk  : 1 (low), 2 (medium), 3 (high)
#   specialist_required : 0 (no) or 1 (yes)
#   testing_lead_time   : days (positive integer)

TEST_ITEMS = [
    {
        "label": "Blood swab from homicide scene",
        "item": {
            "evidence_type":       "biological_dna",
            "offence_type":        "homicide",
            "probative_value":     3,
            "perishability":       3,
            "contamination_risk":  3,
            "specialist_required": 1,
            "testing_lead_time":   5,
        },
    },
    {
        "label": "CCTV footage from robbery",
        "item": {
            "evidence_type":       "digital_cctv",
            "offence_type":        "robbery",
            "probative_value":     3,
            "perishability":       1,
            "contamination_risk":  1,
            "specialist_required": 0,
            "testing_lead_time":   2,
        },
    },
    {
        "label": "Latent fingerprint from burglary",
        "item": {
            "evidence_type":       "fingerprint_latent",
            "offence_type":        "burglary",
            "probative_value":     2,
            "perishability":       2,
            "contamination_risk":  2,
            "specialist_required": 0,
            "testing_lead_time":   5,
        },
    },
    {
        "label": "Tyre impression from vehicle crime",
        "item": {
            "evidence_type":       "impression_tyre",
            "offence_type":        "vehicle_crime",
            "probative_value":     1,
            "perishability":       1,
            "contamination_risk":  2,
            "specialist_required": 0,
            "testing_lead_time":   7,
        },
    },
    {
        "label": "Drug-supply blood toxicology",
        "item": {
            "evidence_type":       "toxicology_blood",
            "offence_type":        "drug_supply",
            "probative_value":     3,
            "perishability":       3,
            "contamination_risk":  2,
            "specialist_required": 1,
            "testing_lead_time":   3,
        },
    },
    {
        "label": "Touch DNA from sexual assault",
        "item": {
            "evidence_type":       "biological_touch_dna",
            "offence_type":        "sexual_assault",
            "probative_value":     2,
            "perishability":       3,
            "contamination_risk":  3,
            "specialist_required": 1,
            "testing_lead_time":   7,
        },
    },
    {
        "label": "Questioned document from fraud",
        "item": {
            "evidence_type":       "document_questioned",
            "offence_type":        "fraud",
            "probative_value":     2,
            "perishability":       1,
            "contamination_risk":  1,
            "specialist_required": 1,
            "testing_lead_time":   14,
        },
    },
    {
        "label": "Soil trace from burglary",
        "item": {
            "evidence_type":       "trace_soil",
            "offence_type":        "burglary",
            "probative_value":     1,
            "perishability":       1,
            "contamination_risk":  1,
            "specialist_required": 1,
            "testing_lead_time":   10,
        },
    },
]


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def triage_item(label: str, item: dict, clf, encoders, show_path: bool = True):
    """Print the triage result (and optionally the decision path) for one item."""
    priority = predict_priority(item, clf, encoders)

    # Colour-code the priority label using ANSI escape codes (terminals only)
    colours = {
        "Critical": "\033[91m",   # bright red
        "High":     "\033[93m",   # bright yellow
        "Standard": "\033[94m",   # bright blue
        "Low":      "\033[92m",   # bright green
    }
    reset = "\033[0m"
    colour = colours.get(priority, "")

    print("=" * 60)
    print(f"Evidence : {label}")
    print(f"Priority : {colour}{priority}{reset}")
    print()

    # Echo the key feature values for quick reference
    print("Features submitted:")
    for k, v in item.items():
        print(f"  {k:<24} {v}")
    print()

    if show_path:
        path_text = explain_decision(item, clf, encoders)
        print(path_text)
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  FORENSIC EVIDENCE TRIAGE – DECISION TREE PROTOTYPE")
    print("  *** SYNTHETIC / DEMONSTRATION DATA ONLY ***")
    print("=" * 60)
    print()

    # Train the model
    print("Training DecisionTreeClassifier on synthetic dataset...")
    clf, encoders, accuracy = build_model()
    print(f"Training-set accuracy: {accuracy * 100:.1f}%  "
          f"(train = test here; prototype only)\n")

    print(f"Priority levels: {', '.join(PRIORITY_LABELS)}")
    print(f"Evidence types : {len(EVIDENCE_TYPES)} categories")
    print(f"Offence types  : {len(OFFENCE_TYPES)} categories")
    print()

    # Run all test items
    for entry in TEST_ITEMS:
        triage_item(entry["label"], entry["item"], clf, encoders, show_path=True)

    print("=" * 60)
    print("Done. To test your own item, add a dict to TEST_ITEMS in run_triage.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
