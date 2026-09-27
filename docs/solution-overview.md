# EvidencePro — Solution Overview

## What We Built

EvidencePro is a Streamlit-based AI-assisted forensic evidence triage platform
for police investigators. It takes a list of crime-scene evidence items, runs
them through one of three ML models, explains each recommendation, lets the
investigator review and override every decision, and produces a deterministic
FSL examination schedule and a downloadable Markdown report — all within a
single browser session.

All AI and ML outputs are **recommendations only**. The investigator is the
final decision-maker at every step.

## How It Works

1. **Case context entry (Tab 1):** The investigator enters the FIR / case
   number, selects the offence type, and optionally adds a case narrative.
   The offence type becomes a feature in the ML model for every evidence item.

2. **Evidence input with optional AI extraction (Tab 2):** The investigator
   enters a free-text description of each evidence item. Clicking
   "Extract with AI" calls IBM watsonx.ai (`granite-13b-instruct-v2`) to
   extract structured field values (evidence type, perishability, specialist
   requirement, etc.) from the description. Without an API key, a heuristic
   keyword-matching fallback extracts partial values. The investigator reviews
   every suggested value in a pre-filled form — fields populated by AI are
   marked with a 🤖 badge — and explicitly confirms the item before it
   enters the pipeline.

3. **ML triage (Tab 3):** The trained model (Decision Tree, Random Forest, or
   Gradient Boosting — selected in the sidebar) runs `predict_priority()` on
   every evidence item. Each item receives a priority tier (Critical / High /
   Standard / Low), a confidence score, and an urgency flag. The urgency flag
   fires independently of the ML tier: perishability 3 is always urgent;
   perishability 2 is urgent if the evidence is less than 6 hours old.

4. **Explainability (Tab 4):** Each item's recommendation is explained.
   Decision Tree shows the actual per-item rule path (node by node). Random
   Forest and Gradient Boosting show the item's own feature values alongside
   the model's global feature importances — with a prominent disclaimer that
   global importances are a property of the model across all training data
   and are NOT a causal explanation of this specific item.

5. **Human review (Tab 5):** The investigator accepts or overrides each AI
   recommendation. Overrides require a mandatory text reason. Accepted items
   proceed with the AI tier; overridden items proceed with the investigator's
   chosen tier. Either way, the decision is recorded.

6. **FSL schedule (Tab 6):** `build_schedule()` assigns every item to one of
   three batches using deterministic, documented rules. The batch assignment
   rules are transparent and shown in the UI. Items within each batch are
   sorted by degradation risk (highest first), then by testing lead time
   (shortest first).

7. **Report (Tab 7):** `generate_report()` produces a nine-section Markdown
   report covering the case, evidence inventory, classification table, per-item
   explanations, FSL schedule, investigator decisions, model metadata, and
   disclaimers. The report is rendered in the browser and available as a
   downloadable `.md` file.

8. **Policy Admin (Tab 8):** Users with the Policy Admin role can propose
   PDES-dimension weight adjustments. An Approver must review and approve the
   proposal before it becomes active. The raw ML recommendation is always
   shown alongside the policy-adjusted recommendation so both can be compared.

## Architecture Diagram

See [`architecture.md`](architecture.md) for the detailed component diagram.

```
[Free-text description]
    |
    v
extractor.py  --  watsonx.ai or heuristic extraction
    |
    v
[Investigator reviews/corrects form]  →  EvidenceItem dataclass
    |
    v
models/triage_models.py  --  DT / RF / GB  →  TriageResult
    |
    v
models/policy.py  --  optional PDES weighting  →  adjusted TriageResult
    |
    v
[Investigator accepts / overrides each item]
    |
    v
scheduler.py  --  deterministic batch assignment  →  ScheduledItem list
    |
    v
report.py  --  Markdown report generation
    |
    v
app.py  --  Streamlit presentation layer (download + display)
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Human-in-the-loop at every step | Forensic decisions require investigator accountability. AI provides a recommendation; the investigator is the final decision-maker. |
| Separate urgency flag from ML tier | Urgency is a physical property of evidence degradation — it should not be computed by the ML model. `compute_urgency_flag()` is defined once in `triage_models.py` and used consistently. |
| Decision Tree path-walk for DT explainability | The exact rule path for a specific item is more meaningful to an investigator than a global importance score. |
| Global importances for RF/GB, clearly labelled | SHAP was explicitly excluded. RF/GB global importances are prominently labelled as NOT per-item causation, with the item's own feature values shown separately for comparison. |
| Policy governance: propose → approve cycle | Weights that affect evidence prioritisation should not be changeable by any individual investigator in real time. The two-role approval cycle prevents unilateral changes. |
| Graceful extraction fallback | The application must work without an IBM Cloud API key. Heuristic keyword extraction and fully manual entry are supported with no degradation to the core workflow. |

## IBM Technologies Used

- **IBM watsonx.ai (`ibm/granite-13b-instruct-v2`):** Used in `extractor.py`
  via the `ibm-watsonx-ai` Python SDK to extract structured evidence feature
  values from a free-text investigator description. The LLM is given a
  carefully constructed prompt that lists allowed field values and explicitly
  instructs the model not to assign a priority tier or make forensic decisions.
  The response is parsed as JSON and each field is validated before it is shown
  to the investigator. The system degrades gracefully to heuristic extraction
  if the SDK or credentials are absent.

- **IBM Bob:** Used as the AI coding assistant throughout the development of
  EvidencePro — for code generation, architecture review, test authoring, and
  documentation.
