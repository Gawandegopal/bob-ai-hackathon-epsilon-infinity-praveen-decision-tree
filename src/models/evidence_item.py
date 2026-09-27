"""
evidence_item.py
================
Canonical data models for EvidencePro.

Defines the two dataclasses that flow through every component of the system:

  CaseContext  — case-level information (shared across all evidence items in a session)
  EvidenceItem — one piece of forensic evidence and all its structured attributes

PDES dimension mapping
----------------------
  P = Probative Value     → EvidenceItem.probative_value     (int 1-3)
  D = Degradation Risk    → EvidenceItem.perishability        (int 1-3)
                            ("perishability" is the internal name; D is the PDES label)
  E = Exclusionary Power  → EvidenceItem.exclusionary_power   (int 1-3)
  S = Processing Speed    → inverse of EvidenceItem.testing_lead_time
                            (a short lead time = high processing speed)
                            No separate S column is stored; the inverse relationship
                            is documented here and applied at scoring time.

ML features vs. operational features
--------------------------------------
The following EvidenceItem fields are used as ML model input features:
  evidence_type, offence_type, probative_value, perishability,
  exclusionary_power, contamination_risk, specialist_required, testing_lead_time

The following fields are NOT ML features. They are used for urgency computation,
FSL scheduling, reporting, and provenance tracking only:
  collection_age_hours, evidence_condition, specialist_type, quantity,
  ai_extracted, investigator_corrected
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# VALID VALUES — kept here so other modules can import and validate against them
# ---------------------------------------------------------------------------

EVIDENCE_TYPES = [
    "biological_dna",
    "biological_touch_dna",
    "fingerprint_latent",
    "fingerprint_patent",
    "digital_device",
    "digital_cctv",
    "firearm_ballistic",
    "firearm_residue",
    "toxicology_blood",
    "toxicology_hair",
    "document_questioned",
    "trace_fibre",
    "trace_glass",
    "trace_soil",
    "impression_footwear",
    "impression_tyre",
]

OFFENCE_TYPES = [
    "homicide",
    "sexual_assault",
    "robbery",
    "burglary",
    "drug_supply",
    "fraud",
    "arson",
    "vehicle_crime",
]

PRIORITY_LABELS = ["Critical", "High", "Standard", "Low"]


# ---------------------------------------------------------------------------
# CaseContext
# ---------------------------------------------------------------------------

@dataclass
class CaseContext:
    """
    Case-level information shared across all evidence items in a session.

    Attributes
    ----------
    fir_number : str
        The First Information Report or case reference number.
    offence_type : str
        Crime category — must be one of OFFENCE_TYPES.
    narrative : str
        Free-text investigator description of the incident.
    """
    fir_number: str
    offence_type: str
    narrative: str = ""


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """
    One piece of forensic evidence and all its structured attributes.

    ML features (passed to the model)
    ----------------------------------
    evidence_type       : str   — one of EVIDENCE_TYPES
    offence_type        : str   — one of OFFENCE_TYPES (propagated from CaseContext)
    probative_value     : int   — 1 (low) / 2 (medium) / 3 (high)              [P]
    perishability       : int   — 1 (stable) / 2 (days) / 3 (hours)            [D]
    exclusionary_power  : int   — 1 (low) / 2 (medium) / 3 (high)              [E]
    contamination_risk  : int   — 1 (low) / 2 (medium) / 3 (high)
    specialist_required : int   — 0 (no) / 1 (yes)
    testing_lead_time   : int   — approximate lab turnaround in days (proxy for S)

    Operational features (NOT ML features)
    ----------------------------------------
    collection_age_hours : int  — hours elapsed since evidence was collected.
                                  0 = unknown / not entered.
                                  Used to compute urgency_flag (see triage_models.py).
    evidence_condition   : int  — 1 (poor) / 2 (fair) / 3 (good)
    specialist_type      : str  — free-text forensic discipline, e.g. "DNA analyst"
    quantity             : str  — descriptive, e.g. "1 swab" or "3 samples"
    item_id              : str  — unique identifier within the session
    label                : str  — investigator's short description of the item

    Provenance flags
    ----------------
    ai_extracted         : bool — True if field values were suggested by AI extraction
    investigator_corrected : bool — True if the investigator modified AI-extracted values
    """

    # Identity
    item_id: str = ""
    label: str = ""

    # ML features
    evidence_type: str = ""
    offence_type: str = ""
    probative_value: int = 1
    perishability: int = 1
    exclusionary_power: int = 1
    contamination_risk: int = 1
    specialist_required: int = 0
    testing_lead_time: int = 7

    # Operational features
    collection_age_hours: int = 0
    evidence_condition: int = 2
    specialist_type: str = ""
    quantity: str = ""

    # Provenance
    ai_extracted: bool = False
    investigator_corrected: bool = False
