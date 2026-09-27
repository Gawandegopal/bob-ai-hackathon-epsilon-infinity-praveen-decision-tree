"""
extractor.py
============
EvidencePro — AI/NLP Evidence Feature Extraction
--------------------------------------------------

PURPOSE
-------
Accepts a free-text evidence description and returns a dict of candidate
EvidenceItem field values proposed by an LLM (IBM watsonx.ai).

The returned dict is ALWAYS presented to the investigator for review and
correction before being used to construct an EvidenceItem. The extractor
NEVER assigns a final priority tier, overrides the ML model, produces an
FSL schedule, or makes any investigator decision.

PIPELINE POSITION
-----------------
Natural-language description
  → extract_features()        ← this module
  → investigator review/edit  (app.py Tab 2)
  → EvidenceItem              (models/evidence_item.py)
  → ML triage                 (models/triage_models.py)
  → policy layer              (models/policy.py)
  → human review              (app.py Tab 5)
  → FSL scheduling            (scheduler.py)
  → report                    (report.py)

GRACEFUL FALLBACK
-----------------
The extractor has three operating modes:

  Mode 1 — watsonx.ai (live):
    Requires WATSONX_API_KEY, WATSONX_PROJECT_ID, and the
    ibm-watsonx-ai package installed. Makes a real LLM call.

  Mode 2 — rule-based heuristics (no API key / SDK unavailable):
    A lightweight keyword-matching approach that can populate
    evidence_type and specialist_required from the description text.
    Produces partial, lower-confidence suggestions.
    Clearly labelled as "heuristic" in the returned metadata.

  Mode 3 — empty (no extraction possible):
    Returns an empty dict. The form in app.py appears blank for
    fully manual entry. The app works identically with or without
    an API key.

The application NEVER depends on this module for its core workflow.
All modes leave every field editable and require explicit investigator
confirmation before the item enters the pipeline.

CREDENTIALS
-----------
  WATSONX_API_KEY      — IBM Cloud API key
  WATSONX_PROJECT_ID   — watsonx.ai project ID
  WATSONX_URL          — Inference endpoint (default: us-south)

Never hard-coded. Loaded from environment variables or a .env file via
python-dotenv. If credentials are absent, Mode 2 or Mode 3 is used.

DISCLAIMER
----------
All extracted field values are AI-generated suggestions. They must be
reviewed and confirmed by the investigator before being used in triage.
They do not constitute final forensic assessments.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.evidence_item import CaseContext

# Load .env if present (no-op if file does not exist or python-dotenv unavailable)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models.evidence_item import EVIDENCE_TYPES, OFFENCE_TYPES

# ---------------------------------------------------------------------------
# EXTRACTABLE FIELDS
# The extractor only proposes values for these fields.
# item_id, label, offence_type, ai_extracted, investigator_corrected are
# set by the calling code (app.py), not by the extractor.
# ---------------------------------------------------------------------------

EXTRACTABLE_FIELDS = [
    "evidence_type",       # str — one of EVIDENCE_TYPES
    "probative_value",     # int 1-3
    "perishability",       # int 1-3
    "exclusionary_power",  # int 1-3
    "contamination_risk",  # int 1-3
    "specialist_required", # int 0 or 1
    "testing_lead_time",   # int days
    "evidence_condition",  # int 1-3
    "specialist_type",     # str free text
    "quantity",            # str free text
]

# Field value constraints used for validation after extraction
FIELD_CONSTRAINTS = {
    "evidence_type":      {"type": str,  "allowed": EVIDENCE_TYPES},
    "probative_value":    {"type": int,  "min": 1, "max": 3},
    "perishability":      {"type": int,  "min": 1, "max": 3},
    "exclusionary_power": {"type": int,  "min": 1, "max": 3},
    "contamination_risk": {"type": int,  "min": 1, "max": 3},
    "specialist_required":{"type": int,  "min": 0, "max": 1},
    "testing_lead_time":  {"type": int,  "min": 1, "max": 365},
    "evidence_condition": {"type": int,  "min": 1, "max": 3},
    "specialist_type":    {"type": str},
    "quantity":           {"type": str},
}

# Extraction source labels — included in the returned metadata
SOURCE_WATSONX   = "watsonx.ai"
SOURCE_HEURISTIC = "heuristic"
SOURCE_NONE      = "none"


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def extract_features(
    description: str,
    context:     "CaseContext",
    api_key:     str | None = None,
    project_id:  str | None = None,
    api_url:     str | None = None,
) -> dict:
    """
    Propose structured EvidenceItem field values from a natural-language
    evidence description.

    Parameters
    ----------
    description : Free-text evidence description entered by the investigator.
    context     : CaseContext for the current session (offence_type, narrative).
    api_key     : IBM watsonx.ai API key. If None, read from WATSONX_API_KEY
                  environment variable.
    project_id  : watsonx.ai project ID. If None, read from WATSONX_PROJECT_ID.
    api_url     : Inference URL. If None, read from WATSONX_URL.

    Returns
    -------
    dict with zero or more of the EXTRACTABLE_FIELDS as keys, plus:
      "_source"  : one of SOURCE_WATSONX | SOURCE_HEURISTIC | SOURCE_NONE
      "_partial" : True if only some fields were extracted (heuristic mode)

    An empty dict (or dict with only "_source"/"_partial" keys) means the form
    should be shown blank for fully manual entry.

    The returned values are SUGGESTIONS. Every field must be reviewed by the
    investigator before the item enters the ML pipeline.

    Raises
    ------
    Never raises. All errors are caught and result in graceful fallback.
    """
    description = (description or "").strip()
    if not description:
        return {"_source": SOURCE_NONE, "_partial": True}

    # Resolve credentials from args or environment
    _api_key    = api_key    or os.environ.get("WATSONX_API_KEY",    "").strip()
    _project_id = project_id or os.environ.get("WATSONX_PROJECT_ID", "").strip()
    _api_url    = api_url    or os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()

    # --- Mode 1: watsonx.ai ---
    if _api_key and _project_id:
        result = _extract_via_watsonx(description, context, _api_key, _project_id, _api_url)
        if result is not None:
            result["_source"]  = SOURCE_WATSONX
            result["_partial"] = False
            return result
        # Fall through to heuristic if the LLM call failed

    # --- Mode 2: heuristic ---
    result = _extract_via_heuristics(description, context)
    result["_source"]  = SOURCE_HEURISTIC
    result["_partial"] = True
    return result


def build_extraction_prompt(description: str, context: "CaseContext") -> str:
    """
    Build the LLM prompt used in watsonx.ai extraction mode.

    The prompt instructs the LLM to return ONLY a JSON object with the
    extractable fields. It explicitly prohibits the LLM from assigning
    priority tiers or making forensic decisions.
    """
    evidence_list  = "\n".join(f"  - {e}" for e in EVIDENCE_TYPES)
    offence_list   = "\n".join(f"  - {o}" for o in OFFENCE_TYPES)

    return f"""\
You are an AI assistant supporting forensic evidence triage.

Your ONLY task is to extract structured attributes from the evidence description below.
You must NOT assign a priority tier, make forensic decisions, or recommend laboratory scheduling.
All values you provide are suggestions that the investigator will review and may correct.

CASE CONTEXT:
  Offence type : {context.offence_type}
  Narrative    : {context.narrative or "(not provided)"}

EVIDENCE DESCRIPTION:
  {description}

Return ONLY a valid JSON object with any of these fields that you can confidently extract.
Omit a field entirely if you are not confident about its value.
Do NOT include any explanation, markdown, or text outside the JSON object.

FIELD DEFINITIONS AND ALLOWED VALUES:

evidence_type (string) — must be exactly one of:
{evidence_list}

probative_value (integer 1-3) — how strongly this evidence tends to prove/disprove a fact:
  1 = low, 2 = medium, 3 = high

perishability (integer 1-3) — how quickly this evidence degrades without processing:
  1 = stable, 2 = degrades over days, 3 = degrades within hours

exclusionary_power (integer 1-3) — how effectively this evidence can exclude suspects:
  1 = low, 2 = medium, 3 = high

contamination_risk (integer 1-3) — risk of cross-contamination altering the evidence:
  1 = low, 2 = medium, 3 = high

specialist_required (integer 0 or 1) — whether specialist equipment or expertise is needed:
  0 = no, 1 = yes

testing_lead_time (integer, days) — approximate laboratory turnaround time in days

evidence_condition (integer 1-3) — physical condition of the evidence:
  1 = poor, 2 = fair, 3 = good

specialist_type (string) — the specific forensic discipline required, e.g. "DNA analyst"

quantity (string) — descriptive quantity, e.g. "1 swab", "3 samples"

EXAMPLE OUTPUT FORMAT (values are illustrative only):
{{"evidence_type": "biological_dna", "probative_value": 3, "perishability": 3, "specialist_required": 1, "specialist_type": "DNA analyst"}}

JSON output:"""


# ---------------------------------------------------------------------------
# MODE 1 — watsonx.ai
# ---------------------------------------------------------------------------

def _extract_via_watsonx(
    description: str,
    context:     "CaseContext",
    api_key:     str,
    project_id:  str,
    api_url:     str,
) -> dict | None:
    """
    Call the watsonx.ai inference API and parse the JSON response.

    Returns a validated dict on success, or None on any failure.
    Failures are caught silently — the caller falls back to heuristics.
    """
    try:
        from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
        from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params  # type: ignore
    except ImportError:
        # SDK not installed — silent fallback
        return None

    try:
        credentials = Credentials(url=api_url, api_key=api_key)
        client      = APIClient(credentials)

        model = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",
            api_client=client,
            project_id=project_id,
            params={
                Params.MAX_NEW_TOKENS:  512,
                Params.TEMPERATURE:     0.0,
                Params.STOP_SEQUENCES:  ["\n\n", "```"],
            },
        )

        prompt   = build_extraction_prompt(description, context)
        response = model.generate_text(prompt=prompt)

        return _parse_and_validate_json(response)

    except Exception:
        # Network error, auth failure, quota exceeded, etc. — fall back silently
        return None


# ---------------------------------------------------------------------------
# MODE 2 — Rule-based heuristics
# ---------------------------------------------------------------------------

# Keyword → evidence_type mappings (longest-match wins)
_EVIDENCE_KEYWORDS: list[tuple[list[str], str]] = [
    (["touch dna", "contact dna"],              "biological_touch_dna"),
    (["dna", "blood", "saliva", "semen",
      "swab", "biological", "tissue", "cell"],  "biological_dna"),
    (["latent fingerprint", "latent print",
      "fingerprint powder", "latent fp"],        "fingerprint_latent"),
    (["patent fingerprint", "visible fingerprint",
      "bloody fingerprint"],                     "fingerprint_patent"),
    (["cctv", "camera", "surveillance",
      "video footage", "dashcam", "footage"],    "digital_cctv"),
    (["phone", "mobile", "laptop", "tablet",
      "computer", "device", "digital",
      "hard drive", "usb", "memory"],            "digital_device"),
    (["bullet", "cartridge", "casing",
      "ballistic", "firearm", "gun", "weapon",
      "projectile"],                             "firearm_ballistic"),
    (["gsr", "gunshot residue",
      "firearm residue"],                        "firearm_residue"),
    (["blood toxicology", "blood sample",
      "blood alcohol", "drug screen",
      "toxicology blood"],                       "toxicology_blood"),
    (["hair toxicology", "hair sample",
      "hair analysis", "toxicology hair"],       "toxicology_hair"),
    (["document", "cheque", "check",
      "signature", "handwriting",
      "questioned document", "forged"],         "document_questioned"),
    (["fibre", "fiber", "thread",
      "textile", "clothing trace"],              "trace_fibre"),
    (["glass", "broken glass",
      "glass fragment", "glass shard"],          "trace_glass"),
    (["soil", "dirt", "mud",
      "earth", "ground"],                        "trace_soil"),
    (["footwear", "shoe", "boot",
      "footprint", "shoe impression"],           "impression_footwear"),
    (["tyre", "tire", "tyre mark",
      "wheel track"],                            "impression_tyre"),
]

# Keyword → specialist_type mappings
_SPECIALIST_KEYWORDS: list[tuple[list[str], str]] = [
    (["dna", "biological", "touch dna",
      "blood", "saliva", "semen"],               "DNA analyst"),
    (["fingerprint", "latent", "patent"],        "Fingerprint specialist"),
    (["ballistic", "bullet", "firearm",
      "weapon", "gsr", "residue"],               "Ballistics/firearms examiner"),
    (["toxicology", "drug", "alcohol",
      "poison", "chemical"],                     "Toxicologist"),
    (["digital", "phone", "laptop", "device",
      "computer", "cctv", "camera"],             "Digital forensics examiner"),
    (["document", "handwriting", "signature",
      "forged", "cheque"],                       "Document examiner"),
    (["fibre", "fiber", "textile", "glass",
      "soil", "trace"],                          "Trace evidence examiner"),
    (["footwear", "tyre", "impression",
      "shoe", "boot", "tire"],                   "Impression evidence specialist"),
]

# Evidence type → expected specialist_required (default 0 = no)
_SPECIALIST_REQUIRED_MAP: dict[str, int] = {
    "biological_dna":      1,
    "biological_touch_dna":1,
    "toxicology_blood":    1,
    "toxicology_hair":     1,
    "firearm_ballistic":   1,
    "firearm_residue":     1,
    "document_questioned": 1,
    "trace_fibre":         1,
    "trace_glass":         1,
    "trace_soil":          1,
    "digital_device":      1,
    "digital_cctv":        0,
    "fingerprint_latent":  0,
    "fingerprint_patent":  0,
    "impression_footwear": 0,
    "impression_tyre":     0,
}

# Evidence type → default perishability
_PERISHABILITY_MAP: dict[str, int] = {
    "biological_dna":       3,
    "biological_touch_dna": 3,
    "toxicology_blood":     3,
    "firearm_residue":      3,
    "toxicology_hair":      2,
    "trace_fibre":          2,
    "fingerprint_latent":   2,
    "trace_glass":          1,
    "trace_soil":           1,
    "fingerprint_patent":   1,
    "digital_device":       1,
    "digital_cctv":         1,
    "firearm_ballistic":    1,
    "document_questioned":  1,
    "impression_footwear":  1,
    "impression_tyre":      1,
}

# Evidence type → default testing_lead_time (days)
_LEAD_TIME_MAP: dict[str, int] = {
    "biological_dna":       5,
    "biological_touch_dna": 7,
    "toxicology_blood":     3,
    "toxicology_hair":      21,
    "firearm_ballistic":    10,
    "firearm_residue":      4,
    "fingerprint_latent":   5,
    "fingerprint_patent":   3,
    "digital_device":       14,
    "digital_cctv":         2,
    "document_questioned":  14,
    "trace_fibre":          7,
    "trace_glass":          5,
    "trace_soil":           10,
    "impression_footwear":  7,
    "impression_tyre":      7,
}


def _extract_via_heuristics(description: str, context: "CaseContext") -> dict:
    """
    Extract a partial set of fields using keyword matching.

    Only populates fields where reasonable confidence is possible from
    keywords. Returns a partial dict — the form will show remaining
    fields with their default values for the investigator to complete.
    """
    text   = description.lower()
    result: dict = {}

    # Match evidence_type
    evidence_type = None
    for keywords, etype in _EVIDENCE_KEYWORDS:
        if any(kw in text for kw in keywords):
            evidence_type = etype
            break

    if evidence_type:
        result["evidence_type"]       = evidence_type
        result["perishability"]       = _PERISHABILITY_MAP.get(evidence_type, 1)
        result["testing_lead_time"]   = _LEAD_TIME_MAP.get(evidence_type, 7)
        result["specialist_required"] = _SPECIALIST_REQUIRED_MAP.get(evidence_type, 0)

    # Match specialist_type
    for keywords, stype in _SPECIALIST_KEYWORDS:
        if any(kw in text for kw in keywords):
            result["specialist_type"] = stype
            break

    # Quantity hints
    qty_match = re.search(r"\b(\d+)\s*(swab|sample|bag|item|piece|fragment|vial)", text)
    if qty_match:
        result["quantity"] = f"{qty_match.group(1)} {qty_match.group(2)}(s)"

    # Condition hints
    if any(w in text for w in ["damaged", "contaminated", "degraded", "poor"]):
        result["evidence_condition"] = 1
    elif any(w in text for w in ["intact", "sealed", "good", "pristine", "clean"]):
        result["evidence_condition"] = 3

    return result


# ---------------------------------------------------------------------------
# JSON PARSING AND VALIDATION
# ---------------------------------------------------------------------------

def _parse_and_validate_json(raw: str) -> dict | None:
    """
    Parse the LLM's raw text output and validate field values.

    Returns a dict of valid fields (invalid fields are silently dropped),
    or None if no valid JSON is found.
    """
    if not raw:
        return None

    # Extract JSON object from the response (LLMs sometimes add preamble text)
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    validated: dict = {}
    for field, value in parsed.items():
        if field not in EXTRACTABLE_FIELDS:
            continue  # ignore unknown fields (e.g. priority_tier injected by LLM)
        constraint = FIELD_CONSTRAINTS.get(field, {})
        validated_value = _validate_field_value(field, value, constraint)
        if validated_value is not None:
            validated[field] = validated_value

    return validated if validated else None


def _validate_field_value(field: str, value, constraint: dict):
    """
    Validate and coerce a single field value from LLM output.
    Returns the coerced value or None if invalid.
    """
    expected_type = constraint.get("type")

    # Coerce type
    if expected_type == int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
    elif expected_type == str:
        value = str(value).strip()
        if not value:
            return None

    # Check allowed values
    allowed = constraint.get("allowed")
    if allowed is not None and value not in allowed:
        return None

    # Check numeric range
    if expected_type == int:
        if "min" in constraint and value < constraint["min"]:
            return None
        if "max" in constraint and value > constraint["max"]:
            return None

    return value


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def is_watsonx_available() -> bool:
    """
    Return True if the watsonx.ai SDK is installed AND credentials are present.
    Used by app.py to show/hide the 'Extract with AI' button label.
    """
    try:
        import importlib
        if importlib.util.find_spec("ibm_watsonx_ai") is None:
            return False
    except Exception:
        return False

    api_key    = os.environ.get("WATSONX_API_KEY",    "").strip()
    project_id = os.environ.get("WATSONX_PROJECT_ID", "").strip()
    return bool(api_key and project_id)


def extraction_mode_label() -> str:
    """
    Return a short human-readable string describing the current extraction mode.
    Used in the app UI to inform the investigator what to expect.
    """
    if is_watsonx_available():
        return "AI extraction via IBM watsonx.ai"
    return "Heuristic extraction (no API key configured)"
