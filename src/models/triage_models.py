"""
triage_models.py
================
EvidencePro — Unified Multi-Model ML Module
--------------------------------------------
This is the single authoritative ML implementation for EvidencePro.

It supersedes the prototype in evidence_triage.py (which is retained as a
frozen reference baseline and is NOT imported by any new application code).

WARNING — SYNTHETIC DATA
------------------------
All models in this module are trained on a SYNTHETIC / CURATED demonstration
dataset. Cross-validated accuracy figures are DEMONSTRATIVE ONLY and do NOT
represent real-world forensic triage performance. See data/synthetic_dataset.py.

Public API
----------
  get_trained_model(model_name, hyperparams=None) -> TrainingResult
  predict_priority(item, training_result)          -> TriageResult
  validate_item(item)                              -> list[str]
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from models.evidence_item import (
    EVIDENCE_TYPES,
    EvidenceItem,
    OFFENCE_TYPES,
    PRIORITY_LABELS,
)
from data.synthetic_dataset import build_dataset

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DATA_DISCLAIMER = (
    "Accuracy figures are from a synthetic/demonstration dataset and do not "
    "represent real-world forensic performance."
)

#: Features that are passed into the ML model.
#: Order matters — must match TriageEncoders.encode_features() output.
ML_FEATURE_NAMES = [
    "evidence_type",       # categorical (label-encoded)
    "offence_type",        # categorical (label-encoded)
    "probative_value",     # int 1-3  (P)
    "perishability",       # int 1-3  (D)
    "exclusionary_power",  # int 1-3  (E)
    "contamination_risk",  # int 1-3
    "specialist_required", # binary 0/1
    "testing_lead_time",   # int days (proxy for S, inverted)
]

#: These EvidenceItem fields are NOT ML features.
#: They are used for urgency, scheduling, reporting, and provenance only.
NON_ML_FIELDS = [
    "collection_age_hours",
    "evidence_condition",
    "specialist_type",
    "quantity",
    "ai_extracted",
    "investigator_corrected",
]

SUPPORTED_MODELS = Literal["decision_tree", "random_forest", "gradient_boosting"]


# ---------------------------------------------------------------------------
# URGENCY RULE
# Defined once here, consumed identically by scheduler.py.
# ---------------------------------------------------------------------------

def compute_urgency_flag(item: "EvidenceItem") -> bool:
    """
    Return True when the evidence item is considered urgent.

    Rules (transparent and documented):
      perishability == 3   → always urgent (degrades in hours)
      perishability == 2   → urgent ONLY IF collection_age_hours is known
                             (> 0) and within the last 6 hours; meaningful
                             degradation is possible within that window
      perishability == 1   → never urgent via collection age (stable evidence)
      collection_age_hours == 0 → unknown / not entered; urgency NOT assumed

    These rules are specific to the prototype. They are NOT official forensic
    standards or legally validated urgency thresholds.
    """
    p = item.perishability
    age = item.collection_age_hours

    if p == 3:
        return True
    if p == 2 and 0 < age < 6:
        return True
    return False


# ---------------------------------------------------------------------------
# ENCODERS
# ---------------------------------------------------------------------------

class TriageEncoders:
    """
    Holds fitted LabelEncoders for the two categorical features and the target.

    fit() is called once at module level on the fixed vocabulary lists so that
    all models share identical encoding.
    """

    def __init__(self):
        self.evidence_enc = LabelEncoder().fit(EVIDENCE_TYPES)
        self.offence_enc  = LabelEncoder().fit(OFFENCE_TYPES)
        self.priority_enc = LabelEncoder().fit(PRIORITY_LABELS)

    def encode_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame row(s) into a numeric feature matrix."""
        et = self.evidence_enc.transform(df["evidence_type"])
        ot = self.offence_enc.transform(df["offence_type"])
        return np.column_stack([
            et,
            ot,
            df["probative_value"].values,
            df["perishability"].values,
            df["exclusionary_power"].values,
            df["contamination_risk"].values,
            df["specialist_required"].values,
            df["testing_lead_time"].values,
        ])

    def encode_item(self, item: "EvidenceItem") -> np.ndarray:
        """Convert a single EvidenceItem into a 1-row feature array."""
        row = pd.DataFrame([{
            "evidence_type":      item.evidence_type,
            "offence_type":       item.offence_type,
            "probative_value":    item.probative_value,
            "perishability":      item.perishability,
            "exclusionary_power": item.exclusionary_power,
            "contamination_risk": item.contamination_risk,
            "specialist_required": item.specialist_required,
            "testing_lead_time":  item.testing_lead_time,
        }])
        return self.encode_features(row)

    def encode_labels(self, series: pd.Series) -> np.ndarray:
        return self.priority_enc.transform(series)

    def decode_label(self, index: int) -> str:
        return self.priority_enc.inverse_transform([index])[0]


# ---------------------------------------------------------------------------
# RESULT DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """
    Everything produced by training one model on the synthetic dataset.

    Attributes
    ----------
    model_name   : Human-readable model identifier.
    clf          : Fitted sklearn estimator (the actual model object).
    encoders     : TriageEncoders instance used to encode training data.
    feature_names: Ordered list of ML feature names (= ML_FEATURE_NAMES).
    cv_accuracy  : Mean cross-validated accuracy (StratifiedKFold, k=5).
    cv_report    : Full sklearn classification_report string from CV folds.
    data_disclaimer : Fixed warning that accuracy is from synthetic data.
    """
    model_name: str
    clf: Any
    encoders: TriageEncoders
    feature_names: list
    cv_accuracy: float
    cv_report: str
    data_disclaimer: str = DATA_DISCLAIMER


@dataclass
class TriageResult:
    """
    The complete prediction output for one evidence item.

    Attributes
    ----------
    item_id             : Matches EvidenceItem.item_id for traceability.
    priority_tier       : Critical / High / Standard / Low
    priority_score      : Model's probability for the predicted class (0.0-1.0).
    urgency_flag        : True if the evidence is time-sensitive (see compute_urgency_flag).
    explanation         : Human-readable explanation of the recommendation.
    decision_path       : Node-by-node path string (Decision Tree only; empty string for
                          RF/GB, which do not have a single interpretable path).
    model_importances   : (RF/GB only) Global model-level feature importances — ranked list
                          of dicts {"feature": str, "importance": float, "rank": int}.
                          These are a property of the TRAINED MODEL, not of this specific
                          item. They describe which features the model relies on most across
                          all training examples. They do NOT explain why this particular
                          item received its prediction. Empty list for Decision Tree.
    item_feature_values : All ML feature values for this specific item as a decoded dict
                          {feature_name: value}. Shown alongside model_importances so the
                          investigator can see what values this item had and compare them
                          against the features the model generally considers important.
    model_used          : Model identifier string.
    cv_accuracy         : Cross-validated accuracy reported at training time.
    policy_version      : Active policy version string, or "No active policy".
    """
    item_id: str
    priority_tier: str
    priority_score: float
    urgency_flag: bool
    explanation: str
    decision_path: str
    model_importances: list
    item_feature_values: dict
    model_used: str
    cv_accuracy: float
    policy_version: str = "No active policy"


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_item(item: "EvidenceItem") -> list:
    """
    Validate that an EvidenceItem has acceptable values for ML inference.

    Returns a list of error message strings. An empty list means the item
    is valid. Does not raise — callers decide how to handle errors.
    """
    errors = []

    if item.evidence_type not in EVIDENCE_TYPES:
        errors.append(
            f"evidence_type '{item.evidence_type}' is not in the allowed list. "
            f"Choose one of: {', '.join(EVIDENCE_TYPES)}"
        )
    if item.offence_type not in OFFENCE_TYPES:
        errors.append(
            f"offence_type '{item.offence_type}' is not in the allowed list. "
            f"Choose one of: {', '.join(OFFENCE_TYPES)}"
        )
    for name, val in [
        ("probative_value", item.probative_value),
        ("perishability", item.perishability),
        ("exclusionary_power", item.exclusionary_power),
        ("contamination_risk", item.contamination_risk),
    ]:
        if val not in (1, 2, 3):
            errors.append(f"{name} must be 1, 2, or 3 (got {val!r}).")
    if item.specialist_required not in (0, 1):
        errors.append(f"specialist_required must be 0 or 1 (got {item.specialist_required!r}).")
    if not isinstance(item.testing_lead_time, int) or item.testing_lead_time < 1:
        errors.append(
            f"testing_lead_time must be a positive integer in days "
            f"(got {item.testing_lead_time!r})."
        )
    if not isinstance(item.collection_age_hours, int) or item.collection_age_hours < 0:
        errors.append(
            f"collection_age_hours must be a non-negative integer "
            f"(got {item.collection_age_hours!r}). Use 0 if unknown."
        )
    return errors


# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------

def get_trained_model(
    model_name: str,
    hyperparams: dict | None = None,
) -> TrainingResult:
    """
    Train a model on the full synthetic dataset and return a TrainingResult.

    Parameters
    ----------
    model_name  : One of "decision_tree", "random_forest", "gradient_boosting".
    hyperparams : Optional dict of sklearn constructor kwargs. If None, sensible
                  defaults are used.

    Returns
    -------
    TrainingResult with a fitted estimator and StratifiedKFold CV metrics.

    Notes
    -----
    Cross-validation (k=5) is used for accuracy reporting.
    The final model is then re-fitted on the FULL dataset for inference.
    CV accuracy on ~180 synthetic rows has high variance; treat it as
    demonstrative only.
    """
    hp = hyperparams or {}

    if model_name == "decision_tree":
        clf = DecisionTreeClassifier(
            max_depth=hp.get("max_depth", 6),
            criterion=hp.get("criterion", "gini"),
            random_state=hp.get("random_state", 42),
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            random_state=hp.get("random_state", 42),
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", 4),
            learning_rate=hp.get("learning_rate", 0.1),
            random_state=hp.get("random_state", 42),
        )
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            "Choose one of: decision_tree, random_forest, gradient_boosting"
        )

    df       = build_dataset()
    encoders = TriageEncoders()

    X = encoders.encode_features(df)
    y = encoders.encode_labels(df["priority_label"])

    # --- Cross-validation for accuracy reporting ---
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    cv_mean   = float(cv_scores.mean())

    # Full CV report using all predictions (train on k-1, predict on 1)
    from sklearn.model_selection import cross_val_predict
    y_pred_cv = cross_val_predict(clf, X, y, cv=cv)
    label_names = encoders.priority_enc.classes_.tolist()
    cv_report_str = classification_report(y, y_pred_cv, target_names=label_names)

    # --- Final model trained on full dataset for inference ---
    clf.fit(X, y)

    return TrainingResult(
        model_name=model_name,
        clf=clf,
        encoders=encoders,
        feature_names=list(ML_FEATURE_NAMES),
        cv_accuracy=cv_mean,
        cv_report=cv_report_str,
    )


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

def predict_priority(
    item: "EvidenceItem",
    training_result: TrainingResult,
) -> TriageResult:
    """
    Run inference for one EvidenceItem and return a TriageResult.

    The TriageResult includes:
    - priority_tier and priority_score from the ML model
    - urgency_flag from the shared urgency rule (compute_urgency_flag)
    - human-readable explanation appropriate for the model type
    - decision_path (Decision Tree only) or top_features (RF / GB)

    Raises ValueError if the item fails validation.
    """
    errors = validate_item(item)
    if errors:
        raise ValueError("EvidenceItem failed validation:\n" + "\n".join(errors))

    clf      = training_result.clf
    encoders = training_result.encoders
    model    = training_result.model_name

    X = encoders.encode_item(item)

    # Raw prediction
    pred_idx = clf.predict(X)[0]
    tier     = encoders.decode_label(int(pred_idx))

    # Probability of predicted class
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        score = float(proba[pred_idx])
    else:
        score = 1.0  # fallback (shouldn't occur with our three estimators)

    # Urgency
    urgent = compute_urgency_flag(item)

    # Per-item feature values (always computed for all model types)
    item_vals = _item_feature_values(item, encoders, X, training_result.feature_names)

    # Explanation
    if model == "decision_tree":
        path_text   = _explain_decision_tree(item, clf, encoders, X)
        importances = []
        explanation = _build_explanation(
            item, tier, urgent, model, path=path_text, item_vals=item_vals
        )
    else:
        path_text   = ""
        importances = _global_feature_importances(clf, training_result.feature_names)
        explanation = _build_explanation(
            item, tier, urgent, model,
            importances=importances, item_vals=item_vals,
        )

    return TriageResult(
        item_id=item.item_id,
        priority_tier=tier,
        priority_score=round(score, 4),
        urgency_flag=urgent,
        explanation=explanation,
        decision_path=path_text,
        model_importances=importances,
        item_feature_values=item_vals,
        model_used=model,
        cv_accuracy=training_result.cv_accuracy,
    )


# ---------------------------------------------------------------------------
# EXPLAINABILITY — DECISION TREE
# Ported and adapted from evidence_triage.py explain_decision()
# ---------------------------------------------------------------------------

def _explain_decision_tree(
    item: "EvidenceItem",
    clf: DecisionTreeClassifier,
    encoders: TriageEncoders,
    X: np.ndarray,
) -> str:
    """
    Return a human-readable decision path through the Decision Tree.

    For each internal node traversed, reports the feature tested, its value
    for this item, the threshold, and which branch was taken.
    Categorical features (evidence_type, offence_type) are decoded back to
    their string names for readability.
    """
    node_ids       = clf.decision_path(X).indices
    feat_arr       = clf.tree_.feature
    threshold      = clf.tree_.threshold
    children_left  = clf.tree_.children_left
    children_right = clf.tree_.children_right

    lines = ["Decision path:"]
    step  = 1
    for node in node_ids:
        if children_left[node] == children_right[node]:
            # Leaf node
            label = encoders.decode_label(clf.predict(X)[0])
            lines.append(f"  Step {step}: LEAF → predicted priority = {label}")
        else:
            feat_idx  = feat_arr[node]
            feat_name = ML_FEATURE_NAMES[feat_idx]
            thresh    = threshold[node]
            feat_val  = float(X[0, feat_idx])

            if feat_name == "evidence_type":
                display_val = encoders.evidence_enc.inverse_transform([int(feat_val)])[0]
                display_thr = encoders.evidence_enc.inverse_transform([int(thresh)])[0]
                condition   = f"{feat_name} = '{display_val}'"
                branch      = "≤" if feat_val <= thresh else ">"
                branch_desc = (
                    f"code {int(feat_val)} {branch} threshold {int(thresh)}"
                    f" ('{display_thr}')"
                )
            elif feat_name == "offence_type":
                display_val = encoders.offence_enc.inverse_transform([int(feat_val)])[0]
                display_thr = encoders.offence_enc.inverse_transform([int(thresh)])[0]
                condition   = f"{feat_name} = '{display_val}'"
                branch      = "≤" if feat_val <= thresh else ">"
                branch_desc = (
                    f"code {int(feat_val)} {branch} threshold {int(thresh)}"
                    f" ('{display_thr}')"
                )
            else:
                condition   = f"{feat_name} = {feat_val:.0f}"
                branch      = "≤" if feat_val <= thresh else ">"
                branch_desc = f"{feat_val:.0f} {branch} {thresh:.1f}"

            lines.append(f"  Step {step}: [{condition}]  →  {branch_desc}")
        step += 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EXPLAINABILITY — RANDOM FOREST / GRADIENT BOOSTING
# ---------------------------------------------------------------------------

def _global_feature_importances(
    clf: Any,
    feature_names: list,
    top_n: int = 5,
) -> list:
    """
    Return the top N globally important features for this estimator.

    IMPORTANT: These importances are a property of the TRAINED MODEL computed
    across all training examples. They describe which features the model relied
    on most during training. They do NOT explain why any specific evidence item
    received its prediction.

    Returns a list of dicts sorted by importance descending:
      [{"feature": str, "importance": float, "rank": int}, ...]
    """
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    return [
        {
            "feature":    feature_names[i],
            "importance": round(float(importances[i]), 4),
            "rank":       rank + 1,
        }
        for rank, i in enumerate(indices)
    ]


def _item_feature_values(
    item: "EvidenceItem",
    encoders: TriageEncoders,
    X: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Return the decoded feature values for this specific evidence item.

    Categorical features (evidence_type, offence_type) are decoded back to
    their original string names. Numeric features are returned as integers.

    This is the per-item counterpart to _global_feature_importances(). The
    investigator sees what values this item actually had, and can compare them
    against which features the model generally considers important — without
    any claim that the importances caused this specific prediction.
    """
    result = {}
    for i, name in enumerate(feature_names):
        val = float(X[0, i])
        if name == "evidence_type":
            decoded = encoders.evidence_enc.inverse_transform([int(val)])[0]
        elif name == "offence_type":
            decoded = encoders.offence_enc.inverse_transform([int(val)])[0]
        else:
            decoded = int(val) if val == int(val) else round(val, 2)
        result[name] = decoded
    return result


# ---------------------------------------------------------------------------
# HUMAN-READABLE EXPLANATION BUILDER
# ---------------------------------------------------------------------------

def _build_explanation(
    item: "EvidenceItem",
    tier: str,
    urgent: bool,
    model: str,
    path: str = "",
    importances: list | None = None,
    item_vals: dict | None = None,
) -> str:
    """
    Compose a plain-English explanation of the triage recommendation.
    Suitable for display to an investigator without ML background.

    For Decision Tree:
      Includes the actual per-item decision path — each rule that was tested
      for this specific item leading to the predicted tier.

    For Random Forest / Gradient Boosting:
      Includes two clearly separated sections:
        1. This item's feature values — what the model actually received.
        2. Model-level feature importances — which features the model generally
           relies on most across all training data. These are explicitly labelled
           as global model information and are NOT described as the reason this
           specific item received its prediction.
    """
    lines = []

    lines.append(
        f"AI RECOMMENDATION (EvidencePro — {model.replace('_', ' ').title()}): "
        f"Priority {tier}."
    )

    if urgent:
        if item.perishability == 3:
            lines.append(
                "URGENT: This evidence degrades within hours. "
                "Immediate processing is recommended."
            )
        else:
            lines.append(
                f"URGENT: This evidence was collected {item.collection_age_hours} hour(s) "
                "ago and may be degrading. Early processing is recommended."
            )

    # PDES dimension summary — always shown, always per-item values
    p_label = {1: "low", 2: "medium", 3: "high"}
    lines.append(
        f"Probative value (P): {p_label[item.probative_value]} | "
        f"Degradation risk (D): {p_label[item.perishability]} | "
        f"Exclusionary power (E): {p_label[item.exclusionary_power]} | "
        f"Processing speed (S): "
        f"{'fast' if item.testing_lead_time <= 3 else 'moderate' if item.testing_lead_time <= 10 else 'slow'} "
        f"(est. {item.testing_lead_time} day(s))"
    )

    if model == "decision_tree":
        lines.append(
            "How the Decision Tree reached this recommendation "
            "(actual rules applied to this specific item):"
        )
        lines.append(path)

    else:
        # Section 1: per-item values — what the model actually received
        if item_vals:
            lines.append(
                "Feature values for this item "
                "(the values the model used to classify this specific evidence):"
            )
            for fname, fval in item_vals.items():
                lines.append(f"  {fname:<24} {fval}")

        # Section 2: global model importances — clearly separated and labelled
        if importances:
            lines.append(
                "Model-level feature importance (GLOBAL — property of the trained model, "
                "NOT specific to this item):"
            )
            lines.append(
                "  The following shows which features this model generally relies on most "
                "across all training examples. This is not an explanation of why this "
                "particular item was classified as it was."
            )
            for entry in importances:
                lines.append(
                    f"  #{entry['rank']}  {entry['feature']:<24} "
                    f"importance = {entry['importance']:.4f}"
                )

    lines.append(
        "Note: This is an AI-assisted recommendation. "
        "The investigator is the final decision-maker."
    )
    return "\n".join(lines)
