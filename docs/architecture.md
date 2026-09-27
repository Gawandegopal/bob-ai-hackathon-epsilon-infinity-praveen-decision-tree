# EvidencePro — System Architecture

## Overview

EvidencePro is a single-process Python application. There is no separate
frontend/backend split, no microservices, no database, and no message queue.
The Streamlit framework renders both the UI and orchestrates module calls in
the same process. All persistence is in `st.session_state` (in-memory per
browser session).

## Component Map

| Component | File | Responsibility |
|---|---|---|
| Presentation / orchestration | `src/app.py` | Streamlit UI, session state, tab rendering. No ML, policy, or scheduling logic defined here. |
| AI/NLP extraction | `src/extractor.py` | Accepts a free-text evidence description; calls watsonx.ai or falls back to heuristic keyword matching; returns validated field suggestions. |
| Evidence data model | `src/models/evidence_item.py` | `EvidenceItem` and `CaseContext` dataclasses; allowed value lists; ML vs non-ML field separation. |
| ML training & prediction | `src/models/triage_models.py` | `get_trained_model()` (DT/RF/GB), `predict_priority()`, `compute_urgency_flag()`, `validate_item()`, `DATA_DISCLAIMER`. |
| Policy layer | `src/models/policy.py` | `TriagePolicy` dataclass, governance workflow (propose / approve / reject), `apply_policy()`. |
| Synthetic training data | `src/data/synthetic_dataset.py` | 148-row expanded dataset; `build_dataset()` returns a DataFrame. |
| FSL scheduler | `src/scheduler.py` | `build_schedule()` — deterministic, rule-based batch assignment. No ML. |
| Report generator | `src/report.py` | `generate_report()` — assembles a nine-section Markdown report. |
| Frozen baseline | `src/evidence_triage.py` | Original 48-row DT prototype — **never modified**; kept as reference artifact. |
| Baseline CLI | `src/run_triage.py` | Original CLI harness — **never modified**; still runs independently. |

## Data Flow

```
Investigator (browser)
    |
    | 1. Case context (FIR number, offence type, narrative)
    v
app.py  →  st.session_state.case_context  (CaseContext)
    |
    | 2. Evidence description (free text, optional)
    v
extractor.py
    ├─ Mode 1: ibm_watsonx_ai SDK  →  granite-13b-instruct-v2  →  JSON response
    ├─ Mode 2: heuristic keyword match  →  partial dict
    └─ Mode 3: empty dict (no extraction)
    |
    | 3. Investigator reviews/corrects pre-filled form
    v
EvidenceItem dataclass  →  st.session_state.evidence_items
    |
    | 4. "Run Triage" button
    v
models/triage_models.py
    ├─ get_trained_model(model_name)  →  TrainingResult (DT / RF / GB)
    └─ predict_priority(item, training_result)  →  TriageResult
           ├─ priority_tier (Critical / High / Standard / Low)
           ├─ priority_score (model probability)
           ├─ urgency_flag (compute_urgency_flag — physical rule, not ML)
           ├─ decision_path (DT only — node-by-node path)
           └─ model_importances + item_feature_values (RF/GB only)
    |
    | 5. Optional: policy layer
    v
models/policy.py
    └─ apply_policy(result, item, policy)  →  adjusted TriageResult
           weighted_score = (P*wP + D*wD + E*wE + S_inv*wS) / sum(weights)
           re-derive tier from thresholds: >=0.75=Critical, >=0.50=High, >=0.25=Standard
    |
    | 6. Investigator accepts or overrides each item (mandatory reason for overrides)
    v
st.session_state.overrides  {item_id → (tier, reason)}
    |
    | 7. "Build Schedule" button
    v
scheduler.py
    └─ build_schedule(items, results, overrides, policy)  →  list[ScheduledItem]
           Batch 1 (Immediate): Critical tier OR urgency_flag==True
           Batch 2 (Secondary): High non-urgent OR Standard+specialist
           Batch 3 (Archive):   Low OR Standard no-specialist
           Within-batch sort: perishability DESC, testing_lead_time ASC
    |
    | 8. "Generate Report" button
    v
report.py
    └─ generate_report(ReportData)  →  Markdown string
           9 sections including disclaimers, per-item explanations, schedule,
           investigator decisions, model metadata, synthetic data warning
    |
    v
Streamlit download button  →  EvidencePro_<FIR>.md
```

## Module Dependency Graph

```
app.py
  ├── extractor.py
  │     └── models/evidence_item.py  (EVIDENCE_TYPES, OFFENCE_TYPES)
  ├── models/evidence_item.py
  ├── models/triage_models.py
  │     ├── models/evidence_item.py
  │     └── data/synthetic_dataset.py
  ├── models/policy.py
  │     └── models/triage_models.py  (TriageResult)
  ├── scheduler.py
  │     ├── models/evidence_item.py
  │     └── models/triage_models.py  (TriageResult, PRIORITY_LABELS)
  └── report.py
        ├── models/evidence_item.py
        ├── models/triage_models.py  (DATA_DISCLAIMER)
        └── scheduler.py  (ScheduledItem)

evidence_triage.py  ←─ isolated, imported only by run_triage.py
run_triage.py       ←─ standalone CLI; does NOT import any new module
```

## PDES Forensic Framework

EvidencePro uses the PDES framework to structure evidence assessment:

| Dimension | Field | Notes |
|---|---|---|
| **P** — Probative Value | `probative_value` (int 1-3) | How strongly the evidence tends to prove or disprove a fact |
| **D** — Degradation Risk | `perishability` (int 1-3) | How quickly the evidence degrades without processing |
| **E** — Exclusionary Power | `exclusionary_power` (int 1-3) | How effectively the evidence can exclude suspects |
| **S** — Processing Speed | `testing_lead_time` (inverse, int days) | Shorter lead time = higher S dimension |

`collection_age_hours`, `specialist_required`, `contamination_risk`, and
`evidence_condition` are secondary features: they are used for urgency,
scheduling, and reporting but are not all ML features (see
`ML_FEATURE_NAMES` in `triage_models.py`).

## ML Architecture

```
data/synthetic_dataset.py
    build_dataset()  →  DataFrame (148 rows, 8 ML features, 4 labels)

models/triage_models.py
    TriageEncoders.fit(EVIDENCE_TYPES, OFFENCE_TYPES, PRIORITY_LABELS)
    TriageEncoders.encode_features(EvidenceItem)  →  np.ndarray [1 x 8]

    get_trained_model("decision_tree" | "random_forest" | "gradient_boosting")
        StratifiedKFold(k=5) for cv_accuracy
        final model trained on full dataset
        →  TrainingResult

    predict_priority(item, training_result)
        →  TriageResult

    Explainability:
        Decision Tree   →  decision_path  (walk tree_ nodes for this item)
        Random Forest   →  model_importances (global) + item_feature_values
        Gradient Boost  →  model_importances (global) + item_feature_values
```

## Security Notes

- API keys (`WATSONX_API_KEY`, `WATSONX_PROJECT_ID`) are loaded from
  environment variables or a `.env` file and are never hard-coded or logged.
- The `.env` file is in `.gitignore`.
- Role selection in the Policy Admin tab is a prototype simulation only;
  there is no real authentication in this prototype.

## Scalability Notes

This is a single-session prototype. For production use:

- Session state would move to a persistent database (PostgreSQL or similar).
- The ML models would be trained offline and serialised to disk (e.g., joblib).
- Authentication would replace the role simulation.
- The watsonx.ai extraction would be rate-limited and have retry logic.
- Multi-investigator collaboration would require a shared session store.
