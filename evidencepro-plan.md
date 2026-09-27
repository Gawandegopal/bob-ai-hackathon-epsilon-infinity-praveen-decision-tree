# EvidencePro — Revised Implementation Plan

> **Status:** DRAFT — awaiting approval before any code changes.
>
> This plan is grounded in a full inspection of the existing source code.
> No files have been modified.

---

## Top-Level Overview

EvidencePro is an AI-assisted forensic evidence triage system built on top of an
existing Decision Tree prototype. The goal is to evolve the two-file Python CLI
into a complete Streamlit-based investigator platform that:

- accepts natural-language case and evidence descriptions
- extracts structured evidence features with AI assistance
- runs three comparable ML models (Decision Tree, Random Forest, Gradient Boosting)
- produces transparent, investigator-readable priority recommendations and explanations
- applies an optional configurable policy layer
- supports investigator accept/override with recorded reasons
- builds a deterministic FSL examination schedule
- generates a human-readable downloadable report

The existing `evidence_triage.py` is the validated baseline. Its logic is
preserved and refactored — not replaced.

---

## Current State

### Files that contain real implementation

| File | What it does | Disposition |
|---|---|---|
| `src/evidence_triage.py` | Dataset, encoders, DT model, prediction, explanation | FROZEN temporarily as comparison baseline — do not modify. Once triage_models.py is complete and verified, this file becomes a reference artifact only. It is not imported by any new code. |
| `src/run_triage.py` | CLI harness, TEST_ITEMS, triage_item() print helper | Keep as CLI; thin-adapter after refactor |
| `requirements.txt` | scikit-learn, pandas, numpy | Extend — add streamlit, python-dotenv, ibm-watsonx-ai |

### Files that are templates only

All docs, README, submission.yaml, demo artifacts — unfilled templates.
These will be completed in Phase 8.

### What the existing prototype already provides

- 7 structured evidence features: evidence_type, offence_type, probative_value,
  perishability, contamination_risk, specialist_required, testing_lead_time
- 16 evidence type categories, 8 offence categories, 4 priority labels
- 48-row synthetic dataset with rationale-based labelling
- Fitted LabelEncoders for categorical features
- predict_priority() returning a string label
- explain_decision() returning a human-readable decision path string
- A working CLI that produces coloured, explained output

### Gaps identified

- No EvidenceItem or CaseContext data model (evidence is passed as plain dicts)
- No multi-model support (DT only; explain_decision uses clf.tree_ — DT-specific)
- build_model() returns train=test accuracy on 48 rows — misleading metric
- No urgency flag distinct from priority tier
- No exclusionary_power (E) or processing_speed (S) — two of the four PDES dimensions are absent
- No collection_age feature — urgency cannot reflect real-time evidence degradation
- No AI/NLP extraction layer
- No optional policy/weighting layer
- No FSL scheduler
- No report generation
- No Streamlit UI
- No human-in-the-loop review
- triage_item() only prints; prediction results have no structured return type

---

## Proposed File Structure

```
src/
  app.py                   NEW — Streamlit entry point
  extractor.py             NEW — AI/NLP feature extraction from free text
  scheduler.py             NEW — deterministic FSL batch scheduler
  report.py                NEW — Markdown report assembly
  run_triage.py            KEEP (minor adapter only) — CLI harness still works
  evidence_triage.py       FREEZE — do not modify; documented baseline

  models/
    __init__.py            NEW
    evidence_item.py       NEW — EvidenceItem + CaseContext dataclasses
    triage_models.py       NEW — multi-model ML module (replaces build_model)
    policy.py              NEW — optional PDES policy layer + governance

  data/
    synthetic_dataset.py   NEW — expanded dataset (migrated + extended from evidence_triage.py)

  tests/
    test_triage_models.py  NEW
    test_scheduler.py      NEW
    test_extractor.py      NEW
    test_report.py         NEW

requirements.txt           EXTEND — add streamlit, python-dotenv, ibm-watsonx-ai
.env.example               UPDATE — remove unused DATABASE_URL, SLACK_WEBHOOK_URL
```

---

## Sub-Task 1 — Evidence Schema and Expanded Dataset

**Intent**
Establish the single canonical data model used throughout the entire system.
Expand the synthetic dataset to support multi-model comparison with cross-validation.

**Expected Outcomes**
- `models/evidence_item.py` exists with EvidenceItem and CaseContext dataclasses
- `data/synthetic_dataset.py` exists with an expanded dataset (target ~150-200 rows)
- The cross-validated accuracy from this dataset is **demonstrative only**. It shows
  that the pipeline is functional and that model comparison is possible. It does NOT
  represent real-world forensic triage accuracy. This limitation must be stated in
  the UI, in the report, and in all documentation.
- All seven original features are present plus the new PDES-aligned ones
- The dataset module has clear SYNTHETIC DATA warnings and documented rationale

**Todo List**
1. Create `src/models/__init__.py` (empty)
2. Create `src/models/evidence_item.py`
   - CaseContext dataclass: fir_number (str), offence_type (str), narrative (str)
   - EvidenceItem dataclass: all seven existing fields plus:
     - exclusionary_power (int 1-3) — maps to new E dimension
     - collection_age_hours (int) — hours since evidence was collected, drives urgency
     - evidence_condition (int 1-3) — 1=poor, 2=fair, 3=good
     - specialist_type (str) — descriptive discipline name, NOT an ML feature
     - quantity (str) — descriptive, NOT an ML feature; scheduling hint only
     - ai_extracted (bool) — was this row filled by AI or manually?
     - investigator_corrected (bool) — did the investigator change extracted values?
   - Note: processing_speed (S) is the inverse of testing_lead_time — document this
     mapping explicitly; do not add a redundant column
   - Note: perishability IS the D dimension — rename is documentary only
3. Create `src/data/synthetic_dataset.py`
   - Copy build_dataset() logic from evidence_triage.py as the starting point
   - Add exclusionary_power column with rationale-based values
   - Add collection_age_hours column (representative values per scenario type)
   - Add evidence_condition column
   - Expand from 48 rows to ~180 rows by systematically varying:
     - probative_value across 1/2/3
     - perishability across 1/2/3
     - exclusionary_power across 1/2/3
     - collection_age_hours across realistic ranges
     - across all 8 offence types, not only the currently covered ones
   - Keep the original 48 rows intact as the first block
   - Document the labelling rationale in comments
   - Include module-level SYNTHETIC DATA warning

**Relevant Context**
- Existing EVIDENCE_TYPES, OFFENCE_TYPES, PRIORITY_LABELS in `evidence_triage.py`
- Dataset rationale comments in build_dataset() docstring
- The urgency_flag (derived from collection_age_hours + perishability) is
  computed at prediction time, not stored in the dataset

**Status:** [x] done

---

## Sub-Task 2 — Multi-Model ML Module

**Intent**
Create a single ML module that supports Decision Tree, Random Forest, and
Gradient Boosting through a uniform interface, so the rest of the application
never needs to know which model is active.

**Expected Outcomes**
- `models/triage_models.py` exists
- All three models train from the same feature set and return the same result type
- explain_decision works for all three models (path for DT, top features for RF/GB)
- predict_priority returns a structured TriageResult dataclass, not just a string
- Cross-validated accuracy is used, not train=test accuracy
- TriageEncoders handles the expanded feature set

**Todo List**
1. Create `src/models/triage_models.py` with:
   a. Copy EVIDENCE_TYPES, OFFENCE_TYPES, PRIORITY_LABELS from evidence_triage.py
      (source of truth moves here; evidence_triage.py is frozen)
   b. ML_FEATURE_NAMES constant listing only the features actually used in ML:
      evidence_type, offence_type, probative_value, perishability,
      contamination_risk, specialist_required, testing_lead_time,
      exclusionary_power
      (collection_age_hours, specialist_type, quantity stay in EvidenceItem
       but are NOT ML features — they drive scheduling and urgency logic)
   c. TriageEncoders class (equivalent to existing, extended for new features)
   d. TriageResult dataclass:
      - priority_tier (str)    — Critical/High/Standard/Low
      - priority_score (float) — raw model probability for predicted class
      - urgency_flag (bool)    — True when:
          perishability == 3 (always urgent regardless of collection age), OR
          perishability == 2 AND collection_age_hours > 0 AND collection_age_hours < 6
            (medium-perishability evidence collected within the last 6 hours is flagged
             urgent because meaningful degradation is possible within that window).
          perishability == 1 items are NEVER flagged urgent via collection age.
          collection_age_hours == 0 means unknown / not entered; urgency is NOT
          assumed — the flag does not fire on zero.
      - explanation (str)      — human-readable text
      - decision_path (str)    — node-by-node path (DT only, else empty)
      - top_features (list)    — top contributing features (RF/GB)
      - model_used (str)       — "decision_tree" | "random_forest" | "gradient_boosting"
      - cv_accuracy (float)    — cross-validated accuracy reported at training
   e. TrainingResult dataclass:
      - model_name (str)
      - clf (Any)
      - encoders (TriageEncoders)
      - feature_names (list)
      - cv_accuracy (float)
      - cv_report (str)        — sklearn classification_report from cross-validation
      - data_disclaimer (str)  — fixed string embedded at construction time:
          "Accuracy figures are from a synthetic/demonstration dataset and do not
           represent real-world forensic performance."
          Surfaced automatically in the UI and in every generated report.
   f. get_trained_model(model_name, hyperparams=None) -> TrainingResult
      - "decision_tree"     — DecisionTreeClassifier, max_depth=6, gini
      - "random_forest"     — RandomForestClassifier, n_estimators=100
      - "gradient_boosting" — GradientBoostingClassifier, n_estimators=100
      - Uses StratifiedKFold cross-validation (k=5) for cv_accuracy
      - Trains final model on full dataset for inference
   g. predict_priority(item: EvidenceItem, result: TrainingResult) -> TriageResult
   h. explain_decision for DT: walk decision_path nodes (port from evidence_triage.py)
   i. explain_decision for RF/GB: report top 3 features by importance with values
   j. validate_item(item: EvidenceItem) -> list[str]
      - returns list of validation error strings (empty if valid)

**Relevant Context**
- Existing explain_decision() in evidence_triage.py lines 248-302 — port the
  DT path-walk logic, do not rewrite from scratch
- existing encode_item() — equivalent logic moves into TriageEncoders
- sklearn cross_val_score / cross_validate for CV
- RandomForestClassifier.feature_importances_ for RF explanation
- GradientBoostingClassifier.feature_importances_ for GB explanation

**Status:** [x] done

---

## Sub-Task 3 — Optional Policy Layer and Governance

**Intent**
Implement the configurable PDES weighting layer as a clearly separate,
optional step that does not affect the ML model internals. Support simple
role-based policy governance for the prototype.

**Expected Outcomes**
- `models/policy.py` exists
- Default behavior is no policy (model runs unweighted)
- When a policy is active, it adjusts a priority score and can change the tier
- Policy has a version string recorded in reports
- Simple role simulation: Investigator / PolicyAdmin / Approver
- Policy changes require POLICY_ADMIN propose → APPROVER approve/reject cycle
  before becoming active. Only an APPROVER-approved policy becomes the active policy.
- The raw ML recommendation is always shown alongside the policy-adjusted
  recommendation in both the UI and the report.
- The policy is clearly labelled in all UI surfaces and reports as a
  configurable prototype mechanism, not an official forensic standard or
  scientifically validated weighting scheme.
- Raw ML recommendation is always shown alongside the policy-adjusted result

**Todo List**
1. Create `src/models/policy.py` with:
   a. PolicyRole enum: INVESTIGATOR, POLICY_ADMIN, APPROVER
   b. TriagePolicy dataclass:
      - version (str)       — e.g. "1.0.0"; recorded in every report
      - label (str)         — human-readable name, e.g. "Homicide Focus Policy"
      - description (str)
      - weight_P (float)    — default 1.0
      - weight_D (float)    — default 1.0
      - weight_E (float)    — default 1.0
      - weight_S (float)    — default 1.0
      - status (str)        — "proposed" | "approved" | "rejected"
      - proposed_by (str)
      - approved_by (str | None)
   c. DEFAULT_POLICY = None  — system runs unweighted by default
   d. POLICY_HISTORY: list of TriagePolicy — in-memory store for prototype
   e. apply_policy(result: TriageResult, item: EvidenceItem, policy: TriagePolicy | None)
      -> TriageResult  — returns a new TriageResult with adjusted score and tier
      - If policy is None, return result unchanged
      - Weighted score = (P * weight_P + D * weight_D + E * weight_E + S_inv * weight_S)
        normalised to [0, 1] where S_inv = 1 - normalised(testing_lead_time)
      - Re-derive tier from weighted score using documented thresholds
      - Add policy.version to the returned TriageResult
   f. propose_policy(policy: TriagePolicy, role: PolicyRole) -> TriagePolicy
      - Only POLICY_ADMIN role can propose; raises ValueError otherwise
   g. approve_policy(policy: TriagePolicy, approver_name: str, role: PolicyRole)
      -> TriagePolicy
      - Only APPROVER role can approve; raises ValueError otherwise
   h. get_active_policy() -> TriagePolicy | None
      - Returns the most recently approved policy, or None

**Relevant Context**
- The weighted score computation must use EvidenceItem.probative_value (P),
  perishability (D), exclusionary_power (E), and testing_lead_time (inverted S)
- Tier re-derivation thresholds must be documented and transparent
- The UI must show both the raw tier and the policy-adjusted tier side by side

**Status:** [x] done

---

## Sub-Task 4 — AI/NLP Extraction

**Intent**
Allow the investigator to enter a free-text evidence description and have
an LLM suggest structured feature values. The investigator always reviews
and can correct every extracted value before triage runs.

**Expected Outcomes**
- `extractor.py` exists
- extract_features() returns a dict pre-filling the EvidenceItem form
- If watsonx.ai is unavailable, function returns empty dict (graceful fallback)
- No API key means no extraction — the form simply appears blank
- ai_extracted and investigator_corrected flags are set appropriately

**Todo List**
1. Create `src/extractor.py` with:
   a. A prompt template that asks the LLM to return a JSON object with keys
      matching EvidenceItem fields (evidence_type, probative_value, perishability,
      contamination_risk, specialist_required, testing_lead_time, exclusionary_power,
      evidence_condition, specialist_type, quantity)
   b. extract_features(description: str, context: CaseContext,
                       api_key: str | None, project_id: str | None)
      -> dict
      - Calls watsonx.ai (ibm-watson-machine-learning SDK) if credentials present
      - Falls back to empty dict if credentials missing or call fails
      - Parses the LLM JSON response, validates values against allowed ranges
      - Returns dict with only the fields where a confident value was extracted
   c. build_extraction_prompt(description: str, context: CaseContext) -> str
      - Constructs the structured extraction prompt
      - Lists allowed values for categorical fields
2. Update `src/.env.example`:
   - Remove DATABASE_URL (not used)
   - Remove SLACK_WEBHOOK_URL (not used)
   - Keep WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL
   - Add note: "Optional — leave blank to use manual feature entry only"

**Relevant Context**
- EVIDENCE_TYPES and OFFENCE_TYPES from triage_models.py define allowed values
- ibm-watsonx-ai Python SDK (formerly ibm-watson-machine-learning) — add to requirements.txt
- The form in app.py is always shown; extraction only pre-fills it

**Status:** [ ] pending

---

## Sub-Task 5 — FSL Scheduler

**Intent**
Convert a list of TriageResult objects into a deterministic, transparent
three-batch examination schedule. No ML — rule-based logic only.

**Expected Outcomes**
- `scheduler.py` exists
- build_schedule() assigns every evidence item to Batch 1, 2, or 3
- Batch assignment rules are documented and transparent
- Within each batch, items are sorted by degradation urgency then lead time
- Investigator overrides are respected (override can promote or demote an item)
- Scheduler returns a list of ScheduledItem objects

**Todo List**
1. Create `src/scheduler.py` with:
   a. ScheduledItem dataclass:
      - item (EvidenceItem)
      - triage_result (TriageResult)
      - policy_result (TriageResult | None)   — post-policy result if policy active
      - batch (str)      — "Batch 1: Immediate" | "Batch 2: Secondary" | "Batch 3: Archive"
      - batch_reason (str) — one sentence explaining the batch assignment
      - investigator_decision (str | None)    — "accepted" | "overridden"
      - override_reason (str | None)
      - final_tier (str) — investigator's final tier after potential override
   b. Batch assignment rules (TRANSPARENT AND DOCUMENTED):
      Urgency rule (same as TriageResult.urgency_flag — defined once in triage_models.py,
      consumed here): perishability==3 always urgent; perishability==2 AND
      collection_age_hours in (1..5] also urgent; perishability==1 never urgent via
      collection age; collection_age_hours==0 (unknown) does not trigger urgency.
      - Batch 1 (Immediate):
        * priority_tier == "Critical" OR
        * urgency_flag == True (see urgency rule above)
      - Batch 2 (Secondary):
        * priority_tier == "High" AND urgency_flag == False OR
        * priority_tier == "Standard" AND specialist_required == 1
      - Batch 3 (Archive):
        * priority_tier == "Low" OR
        * priority_tier == "Standard" AND specialist_required == 0
   c. Within-batch sort:
      - Primary: perishability descending
      - Secondary: testing_lead_time ascending
   d. build_schedule(items: list[EvidenceItem],
                     results: list[TriageResult],
                     overrides: dict[str, tuple[str, str]],   — item_id -> (tier, reason)
                     policy: TriagePolicy | None)
      -> list[ScheduledItem]

**Relevant Context**
- urgency_flag is computed in predict_priority(), available in TriageResult
- override decisions come from the human review step in app.py
- The batch_reason string is shown in the report and the UI

**Status:** [x] done

---

## Sub-Task 6 — Report Generation

**Intent**
Produce a complete, human-readable Markdown report that an investigator
can review and download.

**Expected Outcomes**
- `report.py` exists
- generate_report() returns a Markdown string
- Report covers all required sections (see list below)
- Report clearly labels every AI/ML recommendation as a recommendation
- Report records policy version if active
- Report is downloadable from Streamlit as a .md file

**Todo List**
1. Create `src/report.py` with:
   a. ReportData dataclass:
      - context (CaseContext)
      - items (list[EvidenceItem])
      - results (list[TriageResult])
      - schedule (list[ScheduledItem])
      - policy (TriagePolicy | None)
      - generated_at (str)      — ISO timestamp
      - model_used (str)
      - cv_accuracy (float)
   b. generate_report(data: ReportData) -> str
      Sections to include:
      1. Header + disclaimer (AI recommendations; investigator is final decision-maker)
      2. Case / FIR summary (FIR number, offence type, narrative)
      3. Evidence inventory (table: item ID, label, type, collection age, condition)
      4. Evidence classification (table: label, PDES values, priority tier, urgency)
      5. Priority explanations (per item: explanation string + decision path if DT)
      6. FSL examination schedule (batches, sorted items, batch reasons)
      7. Investigator review decisions (table: item, AI tier, final tier, override reason)
      8. Model and policy information (model name, cv_accuracy, policy version or "No active policy")
      9. Disclaimer footer (synthetic data warning for prototype)
   c. Helper: _evidence_table(items) -> str
   d. Helper: _classification_table(results) -> str
   e. Helper: _schedule_section(schedule) -> str

**Relevant Context**
- Streamlit can render Markdown directly with st.markdown()
- st.download_button() accepts the string as file content
- The disclaimer must appear at the top and the bottom of the report

**Status:** [x] done

---

## Sub-Task 7 — Streamlit Application

**Intent**
Build the full Streamlit investigator platform that orchestrates all modules
through an 8-step workflow. The UI layer only handles presentation and
user interaction — it does not contain ML or business logic.

**Expected Outcomes**
- `app.py` exists and runs with: `streamlit run src/app.py`
- 8-step workflow is fully navigable
- Model can be selected and switched without restarting
- Policy panel is hidden by default; visible only to authorised role
- Every AI/ML result is labelled as a recommendation
- Human override captures reason and records it
- FSL schedule is displayed as sorted batches
- Report can be downloaded as a Markdown file

**Todo List**
1. Create `src/app.py` with the following sections:

   a. Session state initialisation:
      - case_context, evidence_items, training_result, triage_results,
        scheduled_items, overrides, active_policy, role

   b. Sidebar:
      - EvidencePro title and disclaimer banner
      - Role selector (Investigator / Policy Admin / Approver) — prototype simulation
      - Model selector (Decision Tree / Random Forest / Gradient Boosting)
      - Active policy display (version or "No active policy")
      - [Train / Retrain Model] button

   c. Step 1 — Case Context:
      - FIR number text input
      - Offence type selectbox (OFFENCE_TYPES)
      - Case narrative text area

   d. Step 2 — Evidence Input:
      - Add Evidence form:
        * Evidence description text area
        * [Extract with AI] button → calls extractor.extract_features()
        * Review/correct form with all EvidenceItem fields pre-filled
        * ai_extracted badges on fields filled by AI
        * [Add Evidence Item] button adds to session state
      - Evidence items list (expandable per item; edit/remove buttons)

   e. Step 3 — AI Extraction (embedded in Step 2 form — not a separate page)

   f. Step 4 — Model Configuration (sidebar handles this)
      - Policy admin panel (shown only for POLICY_ADMIN and APPROVER roles):
        * Policy history list
        * Propose new policy form (POLICY_ADMIN only)
        * Approve / reject pending proposals (APPROVER only)

   g. Step 5 — Triage Results:
      - [Run Triage] button triggers predict_priority() for all items
      - Results table: item, priority tier, urgency flag, specialist, AI recommendation badge
      - Expandable per-item section: explanation + decision path (DT) or top features (RF/GB)
      - Raw ML result and policy-adjusted result shown side by side when policy is active

   h. Step 6 — Human Review:
      - Per-item: [Accept] button or [Override] with reason text input
      - Override reason is mandatory before saving
      - Summary of decisions shown

   i. Step 7 — FSL Schedule:
      - build_schedule() called with items, results, overrides, active_policy
      - Display three batch sections with sorted item cards
      - Each card shows: label, final tier, urgency, specialist, batch reason

   j. Step 8 — Report:
      - generate_report() called, rendered with st.markdown()
      - [Download Report] button using st.download_button()

2. Update `requirements.txt`:
   - Add: streamlit, python-dotenv
   - Add: ibm-watsonx-ai (optional — extraction degrades gracefully without it)
   - Pin versions appropriate for Python 3.11+

**Relevant Context**
- st.session_state persists across reruns
- Use st.tabs() or st.expander() to organise the long workflow
- Role selector is a prototype simulation — no real authentication
- All training should happen once per model selection, cached in session state

**Status:** [ ] pending

---

## Sub-Task 8 — CLI Adapter and Baseline Migration

**Intent**
Verify the frozen baseline still runs, provide a side-by-side comparison function
so the new pipeline can be validated against the old one, and then retire
evidence_triage.py to a reference-only artifact. triage_models.py is the single
authoritative ML implementation — no permanent duplicate logic is maintained.

**Expected Outcomes**
- `python run_triage.py` still produces the original baseline output unchanged
- After Phase 1 (triage_models.py complete), run_triage.py gains an optional
  `--compare` mode that runs the same TEST_ITEMS through both the old and new
  pipelines and prints results side by side for verification
- Once the new pipeline is verified, evidence_triage.py is documented as
  "frozen baseline / retired reference" — it is NOT imported by app.py,
  scheduler.py, report.py, or any other new module
- triage_models.py is the only ML implementation used by the application going forward

**Todo List**
1. Add a comment block to the top of run_triage.py (without modifying its logic)
   noting: "Legacy CLI harness. Uses the frozen evidence_triage.py baseline.
   For the full EvidencePro pipeline see models/triage_models.py."
2. Confirm `python run_triage.py` still runs without errors after all other
   new files are created (no import side effects from new modules)
3. After Sub-Task 2 is complete, add a comparison function to run_triage.py:
   - Import TriageResult from models.triage_models
   - Run the same TEST_ITEMS through get_trained_model + predict_priority
   - Print old output and new TriageResult side by side for each item
   - Gate this behind `if __name__ == "__main__" and "--compare" in sys.argv`
     so normal `python run_triage.py` is unchanged
4. Once the comparison confirms the new pipeline is consistent, add a module-level
   note to evidence_triage.py's existing docstring section in run_triage.py
   (not in evidence_triage.py itself — that file is never touched) stating
   that triage_models.py is now the authoritative implementation

**Relevant Context**
- evidence_triage.py imports: numpy, pandas, sklearn only — no circular imports risk
- run_triage.py imports only from evidence_triage — safe throughout
- The comparison step is the validation gate before evidence_triage.py is retired

**Status:** [ ] pending

---

## Sub-Task 9 — Tests

**Intent**
Write focused tests for the ML module, scheduler, and report generator
that can run without API keys.

**Expected Outcomes**
- All tests pass with: `pytest src/tests/ -v`
- No test requires an LLM API key
- ML tests use a small fixture dataset, not the full synthetic dataset
- Coverage includes: prediction, explanation, scheduling, report structure

**Todo List**
1. Create `src/tests/__init__.py`
2. Create `src/tests/test_triage_models.py`:
   - test that get_trained_model returns TrainingResult for all three model names
   - test that predict_priority returns a TriageResult with all expected fields
   - test that explain_decision returns a non-empty string for all three models
   - test that validate_item catches missing/invalid fields
3. Create `src/tests/test_scheduler.py`:
   - test Critical item goes to Batch 1
   - test Low item goes to Batch 3
   - test override moves an item to the correct batch
4. Create `src/tests/test_report.py`:
   - test that generate_report returns a non-empty string containing required sections
   - test disclaimer text is present

**Status:** [ ] pending

---

## Sub-Task 10 — Documentation and Submission Metadata

**Intent**
Complete all unfilled template files so the project passes the GitHub Actions
validation workflow and is ready for submission.

**Expected Outcomes**
- submission.yaml has all required fields filled
- README.md describes the actual EvidencePro project
- docs/architecture.md describes the actual system
- docs/setup-guide.md describes how to run `streamlit run src/app.py`
- docs/problem-statement.md and docs/solution-overview.md are complete

**Todo List**
1. Fill in submission.yaml (team, title, problem_statement, solution_summary, key_features, tech_stack)
2. Replace README.md placeholders with EvidencePro content
3. Write docs/problem-statement.md (forensic triage problem, target users)
4. Write docs/solution-overview.md (what EvidencePro does, PDES framework, IBM technologies used)
5. Write docs/architecture.md (actual system architecture with module descriptions)
6. Write docs/setup-guide.md (pip install, .env, streamlit run command)

**Status:** [ ] pending

---

## Overall System Architecture

```
CaseContext + Evidence descriptions (investigator input)
    |
    v
extractor.py  --  AI/NLP extraction (watsonx.ai, optional)
    |
    v
EvidenceItem dataclass  --  structured features, reviewed and corrected by investigator
    |
    v
models/triage_models.py  --  TriageEncoders + model factory (DT / RF / GB)
    |
    v
TriageResult  --  priority_tier, priority_score, urgency_flag, explanation, decision_path
    |
    v
models/policy.py  --  optional PDES weighting  (TriageResult + TriagePolicy -> adjusted TriageResult)
    |
    v
Human review  --  investigator accepts or overrides, records reason  (app.py session state)
    |
    v
scheduler.py  --  deterministic batch assignment -> list[ScheduledItem]
    |
    v
report.py  --  Markdown report assembly
    |
    v
app.py  --  Streamlit presentation layer
```

---

## ML Architecture

The core principle: one preprocessor, one encoding scheme, three interchangeable estimators.

```
data/synthetic_dataset.py
    build_dataset() -> DataFrame (180+ rows, 9 features, 4 labels)
         |
models/triage_models.py
    TriageEncoders.encode_features(EvidenceItem) -> np.ndarray [1 x 9]
         |
    get_trained_model(model_name) -> TrainingResult
         uses StratifiedKFold CV for accuracy; trains final model on full data
         |
    predict_priority(EvidenceItem, TrainingResult) -> TriageResult
         |
    Decision Tree: explain via decision_path (port from evidence_triage.py)
    Random Forest: explain via feature_importances_ top 3
    Gradient Boosting: explain via feature_importances_ top 3
```

### ML Features (what enters the model)

| Feature | Type | Dimensions |
|---|---|---|
| evidence_type | categorical | — |
| offence_type | categorical | — |
| probative_value | int 1-3 | P |
| perishability | int 1-3 | D |
| exclusionary_power | int 1-3 | E |
| contamination_risk | int 1-3 | secondary |
| specialist_required | binary 0/1 | secondary |
| testing_lead_time | int days | S (inverse) |

### NOT ML features (used only for scheduling, urgency, and reporting)

- collection_age_hours — drives urgency_flag in predict_priority
- specialist_type — free text, used only in report
- quantity — descriptive only
- evidence_condition — reporting only

---

## Policy Architecture

```
DEFAULT_POLICY = None  ->  model runs without weighting

POLICY_HISTORY = []  ->  in-memory store (prototype)

Propose (POLICY_ADMIN) -> status: "proposed"
Approve (APPROVER)     -> status: "approved" -> becomes get_active_policy()

apply_policy(TriageResult, EvidenceItem, TriagePolicy) -> adjusted TriageResult
  weighted_score = (P * wP + D * wD + E * wE + S_inv * wS) / (wP + wD + wE + wS)
  tier = re-derive from thresholds:
    >= 0.75 -> Critical
    >= 0.50 -> High
    >= 0.25 -> Standard
    else    -> Low

Report records: policy.version or "No active policy"
```

---

## FSL Scheduling Architecture

```
Urgency rule (defined once in triage_models.py, used by both TriageResult and scheduler):
  perishability == 3                                        → always urgent
  perishability == 2 AND collection_age_hours in (0, 6)    → urgent
  perishability == 1                                        → never urgent via collection age
  collection_age_hours == 0 (unknown)                      → urgency NOT assumed

Batch 1: Immediate
  IF priority_tier == "Critical" OR urgency_flag == True
  Sort by: perishability DESC, testing_lead_time ASC

Batch 2: Secondary
  IF priority_tier == "High" AND NOT urgent
  IF priority_tier == "Standard" AND specialist_required == 1
  Sort by: perishability DESC, testing_lead_time ASC

Batch 3: Archive
  IF priority_tier == "Low"
  IF priority_tier == "Standard" AND specialist_required == 0
  Sort by: testing_lead_time ASC

Override: investigator-specified final_tier overrides batch assignment
```

---

## Report Architecture

```
1. Header + AI disclaimer (must appear at top)
2. Case / FIR summary
3. Evidence inventory table
4. Evidence classification table (PDES values + tier + urgency)
5. Priority explanations (per item)
6. FSL examination schedule (three batches)
7. Investigator review decisions
8. Model and policy metadata
9. Synthetic data disclaimer (prototype footer)
```

---

## Dependencies

```
# Core ML (already in requirements.txt)
scikit-learn
pandas
numpy

# UI
streamlit

# Environment
python-dotenv

# AI extraction (optional — extraction degrades gracefully without it)
ibm-watsonx-ai

# Testing
pytest
```

No deep learning, no SHAP, no vector databases, no microservices.

---

## Implementation Phases

### Phase 1 — Foundation (Sub-Tasks 1-2)
evidence_item.py + synthetic_dataset.py + triage_models.py.
Goal: all three models train and predict from the same interface.
Milestone: triage_models works from a Python script; no UI yet.

### Phase 2 — Policy + Scheduling + Report (Sub-Tasks 3, 5, 6)
policy.py + scheduler.py + report.py.
Goal: end-to-end pipeline without UI. Test with Python scripts.
Milestone: given a list of EvidenceItem dicts, produce a Markdown report.

### Phase 3 — Streamlit UI (Sub-Task 7)
app.py connecting all modules.
Goal: complete 8-step investigator workflow in browser.
Milestone: investigator can run full triage from browser and download report.

### Phase 4 — AI Extraction (Sub-Task 4)
extractor.py.
Goal: free-text input populates form fields.
Milestone: entering "blood swab from victim" pre-fills biological_dna fields.

### Phase 5 — CLI Adapter + Tests + Docs (Sub-Tasks 8, 9, 10)
Goal: python run_triage.py still works; pytest passes; submission metadata filled.
Milestone: GitHub Actions validation passes.

---

## Risks and Issues

| Risk | Details | Mitigation |
|---|---|---|
| 48-row dataset gives high variance CV | Even 5-fold CV on 48 rows gives ~10 rows per fold — noisy estimates | Expand to 180+ rows before presenting CV results; document the limitation |
| DT train=test accuracy is misleading | build_model() returns clf.score(X, y) — inflated metric | triage_models.py uses cross_val_score; old value is never surfaced in the new UI |
| explain_decision is DT-only | clf.tree_ and decision_path() do not exist on RF/GB | feature_importances_ fallback covers RF/GB; documented clearly |
| watsonx.ai API key may not be available | Extraction depends on credentials | Graceful fallback: form appears blank, investigator enters manually |
| Policy weights are not standards | Prototype weights could be mistaken for official thresholds | Prominent disclaimer in UI and report; policy labeled "configurable prototype" |
| collection_age_hours requires real-time input | Investigator must enter time since collection | Default = 0 (unknown/not entered). urgency_flag does NOT fire on zero. Low-perishability items never trigger urgency via collection age. The 6-hour threshold for medium-perishability items is a documented, transparent rule — not a blanket automatic trigger. |
| Streamlit session state lost on page refresh | Multi-step workflow state is in-memory only | Document as a known prototype limitation in submission |
| evidence_triage.py uses Python 3.10+ tuple return type hint | tuple[...] syntax needs Python 3.10+ | requirements.txt should note Python 3.10+ minimum |
