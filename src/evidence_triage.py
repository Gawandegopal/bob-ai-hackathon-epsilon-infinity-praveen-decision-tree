"""
evidence_triage.py
==================
Forensic Evidence Triage – Decision Tree Prototype
----------------------------------------------------
IMPORTANT: The dataset in this module is SYNTHETIC / CURATED DEMONSTRATION DATA.
It is not derived from real casework and must not be used in any real forensic
triage decision.

This module provides:
  - build_dataset()      : returns a labelled synthetic dataset as a DataFrame
  - build_model()        : trains a DecisionTreeClassifier on that dataset
  - predict_priority()   : predicts the triage priority for one evidence item
  - explain_decision()   : prints the decision path taken through the tree
  - encode_item()        : converts a plain-dict evidence item into model input
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# 1. FEATURE DEFINITIONS
# ---------------------------------------------------------------------------
# Each feature is described here so the rest of the code stays readable.
#
# evidence_type        : category of physical evidence
# offence_type         : category of crime being investigated
# probative_value      : how likely the evidence is to prove/disprove a fact
#                        1 = low, 2 = medium, 3 = high
# perishability        : how quickly the evidence degrades
#                        1 = stable, 2 = degrades over days, 3 = degrades in hours
# contamination_risk   : likelihood of cross-contamination altering the evidence
#                        1 = low, 2 = medium, 3 = high
# specialist_required  : does analysis require specialist equipment/expertise?
#                        0 = no, 1 = yes
# testing_lead_time    : approximate laboratory turnaround in days (continuous)
#
# priority_label       : TARGET – one of: Critical, High, Standard, Low

EVIDENCE_TYPES = [
    "biological_dna", "biological_touch_dna", "fingerprint_latent",
    "fingerprint_patent", "digital_device", "digital_cctv",
    "firearm_ballistic", "firearm_residue", "toxicology_blood",
    "toxicology_hair", "document_questioned", "trace_fibre",
    "trace_glass", "trace_soil", "impression_footwear", "impression_tyre",
]

OFFENCE_TYPES = [
    "homicide", "sexual_assault", "robbery", "burglary",
    "drug_supply", "fraud", "arson", "vehicle_crime",
]

PRIORITY_LABELS = ["Critical", "High", "Standard", "Low"]


# ---------------------------------------------------------------------------
# 2. SYNTHETIC DATASET
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """
    Returns a DataFrame of 48 synthetic evidence scenarios.

    Each row represents ONE piece of evidence in ONE crime scenario.
    The 'priority_label' column is the ground-truth triage decision.

    Rationale for labelling:
      Critical  – time-sensitive, high probative value, serious offence
      High      – important but slightly more stable or less serious
      Standard  – routine processing; no immediate urgency
      Low       – supplementary; unlikely to change outcome alone
    """
    rows = [
        # --- CRITICAL: Homicide + biological DNA ---
        ("biological_dna",      "homicide",       3, 3, 3, 1, 5,  "Critical"),
        ("biological_touch_dna","homicide",        3, 3, 3, 1, 7,  "Critical"),
        ("toxicology_blood",    "homicide",        3, 3, 2, 1, 3,  "Critical"),
        ("firearm_residue",     "homicide",        3, 3, 3, 1, 4,  "Critical"),
        ("firearm_ballistic",   "homicide",        3, 2, 1, 1, 10, "Critical"),
        ("biological_dna",      "sexual_assault",  3, 3, 3, 1, 5,  "Critical"),
        ("toxicology_blood",    "sexual_assault",  3, 3, 2, 1, 3,  "Critical"),
        ("biological_touch_dna","sexual_assault",  2, 3, 3, 1, 7,  "Critical"),
        ("digital_device",      "homicide",        3, 1, 1, 1, 14, "Critical"),
        ("digital_cctv",        "homicide",        3, 2, 1, 0, 2,  "Critical"),

        # --- HIGH: Serious offences, good probative value, less perishable ---
        ("fingerprint_latent",  "homicide",        2, 2, 2, 0, 5,  "High"),
        ("fingerprint_latent",  "sexual_assault",  2, 2, 2, 0, 5,  "High"),
        ("biological_dna",      "robbery",         3, 2, 2, 1, 5,  "High"),
        ("digital_device",      "sexual_assault",  3, 1, 1, 1, 14, "High"),
        ("digital_cctv",        "robbery",         3, 1, 1, 0, 2,  "High"),
        ("toxicology_blood",    "drug_supply",     3, 3, 2, 1, 3,  "High"),
        ("firearm_ballistic",   "robbery",         3, 2, 1, 1, 10, "High"),
        ("firearm_residue",     "robbery",         2, 2, 2, 1, 4,  "High"),
        ("digital_device",      "fraud",           3, 1, 1, 1, 14, "High"),
        ("trace_fibre",         "homicide",        2, 2, 2, 1, 7,  "High"),

        # --- STANDARD: Burglary, arson, vehicle crime – moderate urgency ---
        ("fingerprint_latent",  "burglary",        2, 2, 2, 0, 5,  "Standard"),
        ("fingerprint_patent",  "burglary",        2, 1, 1, 0, 3,  "Standard"),
        ("biological_dna",      "burglary",        2, 1, 1, 1, 5,  "Standard"),
        ("trace_glass",         "burglary",        2, 2, 1, 0, 5,  "Standard"),
        ("trace_fibre",         "burglary",        1, 2, 2, 1, 7,  "Standard"),
        ("impression_footwear", "burglary",        2, 1, 2, 0, 7,  "Standard"),
        ("digital_cctv",        "burglary",        2, 1, 1, 0, 2,  "Standard"),
        ("digital_device",      "drug_supply",     2, 1, 1, 1, 14, "Standard"),
        ("toxicology_hair",     "drug_supply",     2, 1, 1, 1, 21, "Standard"),
        ("impression_tyre",     "arson",           2, 1, 2, 0, 7,  "Standard"),
        ("trace_soil",          "arson",           1, 1, 1, 1, 10, "Standard"),
        ("digital_cctv",        "vehicle_crime",   2, 1, 1, 0, 2,  "Standard"),
        ("impression_footwear", "robbery",         1, 1, 2, 0, 7,  "Standard"),
        ("document_questioned", "fraud",           2, 1, 1, 1, 14, "Standard"),

        # --- LOW: Low probative value, stable, routine ---
        ("trace_soil",          "burglary",        1, 1, 1, 1, 10, "Low"),
        ("trace_fibre",         "vehicle_crime",   1, 1, 2, 1, 7,  "Low"),
        ("impression_tyre",     "vehicle_crime",   1, 1, 2, 0, 7,  "Low"),
        ("fingerprint_patent",  "vehicle_crime",   1, 1, 1, 0, 3,  "Low"),
        ("document_questioned", "vehicle_crime",   1, 1, 1, 1, 14, "Low"),
        ("trace_glass",         "vehicle_crime",   1, 1, 1, 0, 5,  "Low"),
        ("toxicology_hair",     "fraud",           1, 1, 1, 1, 21, "Low"),
        ("impression_footwear", "vehicle_crime",   1, 1, 2, 0, 7,  "Low"),
        ("trace_fibre",         "fraud",           1, 1, 1, 1, 7,  "Low"),
        ("trace_soil",          "vehicle_crime",   1, 1, 1, 1, 10, "Low"),
        ("fingerprint_latent",  "vehicle_crime",   1, 1, 1, 0, 5,  "Low"),
        ("digital_cctv",        "fraud",           1, 1, 1, 0, 2,  "Low"),
        ("impression_tyre",     "burglary",        1, 1, 2, 0, 7,  "Low"),
        ("document_questioned", "burglary",        1, 1, 1, 1, 14, "Low"),
    ]

    columns = [
        "evidence_type", "offence_type",
        "probative_value", "perishability", "contamination_risk",
        "specialist_required", "testing_lead_time",
        "priority_label",
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# 3. ENCODERS  (fit once, reused for new items)
# ---------------------------------------------------------------------------

class TriageEncoders:
    """Holds fitted LabelEncoders for the two categorical features."""

    def __init__(self):
        self.evidence_enc = LabelEncoder().fit(EVIDENCE_TYPES)
        self.offence_enc  = LabelEncoder().fit(OFFENCE_TYPES)
        self.priority_enc = LabelEncoder().fit(PRIORITY_LABELS)

    def encode_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame of raw features into a numeric matrix."""
        et = self.evidence_enc.transform(df["evidence_type"])
        ot = self.offence_enc.transform(df["offence_type"])
        return np.column_stack([
            et, ot,
            df["probative_value"].values,
            df["perishability"].values,
            df["contamination_risk"].values,
            df["specialist_required"].values,
            df["testing_lead_time"].values,
        ])

    def encode_labels(self, series: pd.Series) -> np.ndarray:
        return self.priority_enc.transform(series)

    def decode_label(self, index: int) -> str:
        return self.priority_enc.inverse_transform([index])[0]


FEATURE_NAMES = [
    "evidence_type", "offence_type",
    "probative_value", "perishability", "contamination_risk",
    "specialist_required", "testing_lead_time",
]


# ---------------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------------

def build_model(
    max_depth: int = 6,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, TriageEncoders, float]:
    """
    Train a DecisionTreeClassifier on the synthetic dataset.

    Returns
    -------
    clf       : fitted DecisionTreeClassifier
    encoders  : TriageEncoders instance (needed for new predictions)
    accuracy  : training-set accuracy (informational only for this prototype)
    """
    df       = build_dataset()
    encoders = TriageEncoders()

    X = encoders.encode_features(df)
    y = encoders.encode_labels(df["priority_label"])

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion="gini",
        random_state=random_state,
    )
    clf.fit(X, y)

    accuracy = clf.score(X, y)
    return clf, encoders, accuracy


# ---------------------------------------------------------------------------
# 5. PREDICTION & EXPLANATION
# ---------------------------------------------------------------------------

def encode_item(item: dict, encoders: TriageEncoders) -> np.ndarray:
    """
    Convert a plain-dict evidence item into a 1-row numeric array.

    item keys (all required):
      evidence_type       (str)  – one of EVIDENCE_TYPES
      offence_type        (str)  – one of OFFENCE_TYPES
      probative_value     (int)  – 1, 2, or 3
      perishability       (int)  – 1, 2, or 3
      contamination_risk  (int)  – 1, 2, or 3
      specialist_required (int)  – 0 or 1
      testing_lead_time   (int)  – days (positive integer)
    """
    row = pd.DataFrame([item])
    return encoders.encode_features(row)


def predict_priority(
    item: dict,
    clf: DecisionTreeClassifier,
    encoders: TriageEncoders,
) -> str:
    """Return the predicted priority label for one evidence item."""
    X    = encode_item(item, encoders)
    idx  = clf.predict(X)[0]
    return encoders.decode_label(idx)


def explain_decision(
    item: dict,
    clf: DecisionTreeClassifier,
    encoders: TriageEncoders,
) -> str:
    """
    Return a human-readable explanation of the decision path through the tree.

    The explanation lists each decision node the item passed through, showing
    the feature that was tested, the threshold, and which branch was taken.
    """
    X        = encode_item(item, encoders)
    node_ids = clf.decision_path(X).indices   # nodes visited for this sample
    feature  = clf.tree_.feature
    threshold= clf.tree_.threshold
    children_left  = clf.tree_.children_left
    children_right = clf.tree_.children_right

    lines = ["Decision path:"]
    step  = 1
    for node in node_ids:
        if children_left[node] == children_right[node]:
            # leaf node
            label = encoders.decode_label(clf.predict(X)[0])
            lines.append(f"  Step {step}: LEAF → predicted priority = {label}")
        else:
            feat_idx   = feature[node]
            feat_name  = FEATURE_NAMES[feat_idx]
            thresh     = threshold[node]
            feat_val   = float(X[0, feat_idx])

            # Decode categorical values back to strings for readability
            if feat_name == "evidence_type":
                display_val = encoders.evidence_enc.inverse_transform([int(feat_val)])[0]
                display_thr = encoders.evidence_enc.inverse_transform([int(thresh)])[0]
                condition   = f"{feat_name} = '{display_val}'"
                branch      = "≤" if feat_val <= thresh else ">"
                branch_desc = f"code {int(feat_val)} {branch} threshold {int(thresh)} ('{display_thr}')"
            elif feat_name == "offence_type":
                display_val = encoders.offence_enc.inverse_transform([int(feat_val)])[0]
                display_thr = encoders.offence_enc.inverse_transform([int(thresh)])[0]
                condition   = f"{feat_name} = '{display_val}'"
                branch      = "≤" if feat_val <= thresh else ">"
                branch_desc = f"code {int(feat_val)} {branch} threshold {int(thresh)} ('{display_thr}')"
            else:
                condition   = f"{feat_name} = {feat_val:.0f}"
                branch      = "≤" if feat_val <= thresh else ">"
                branch_desc = f"{feat_val:.0f} {branch} {thresh:.1f}"

            lines.append(
                f"  Step {step}: [{condition}]  →  {branch_desc}"
            )
        step += 1

    return "\n".join(lines)
