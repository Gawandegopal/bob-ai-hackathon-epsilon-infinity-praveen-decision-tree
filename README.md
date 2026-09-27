# EvidencePro — AI-Assisted Forensic Evidence Triage

> A Streamlit platform that helps police investigators prioritise, explain, and
> schedule crime-scene evidence submissions to a Forensic Science Laboratory (FSL).
> Built for the IBM Bob AI Hackathon — Problem 3: AI-Based Crime Scene Evidence Prioritisation.

---

## Team

| Field | Value |
|---|---|
| **Team Name** | Epsilon Infinity |
| **Track** | AI |
| **Team Lead** | Praveen |

---

## Problem Statement

Forensic investigators must decide which evidence items to submit to an FSL
first — without any AI support. These decisions are informal, inconsistent, and
prone to overlooking time-sensitive degradable evidence. Delayed processing of
perishable evidence (biological samples, gunshot residue, toxicology) can
compromise prosecutions and result in irreversible evidence loss.

---

## Solution

EvidencePro is a Streamlit-based AI-assisted triage platform that:

1. Accepts free-text evidence descriptions and uses IBM watsonx.ai (or
   heuristic keyword matching) to extract structured feature values
2. Runs three ML models (Decision Tree, Random Forest, Gradient Boosting)
   across four forensic PDES dimensions to recommend a priority tier
3. Flags time-sensitive evidence as urgent based on perishability and
   collection age
4. Lets investigators accept or override every AI recommendation with a
   mandatory recorded reason
5. Produces a deterministic, three-batch FSL examination schedule
6. Generates a downloadable Markdown report with full disclaimers and
   investigator decision audit trail

**All AI/ML outputs are recommendations only. The investigator is the final
decision-maker.**

---

## Key Features

- **Three-model ML triage** — Decision Tree, Random Forest, and Gradient
  Boosting with cross-validated accuracy and per-item explanations
- **IBM watsonx.ai NLP extraction** — free-text description auto-populates
  form fields via `granite-13b-instruct-v2`; graceful heuristic fallback
  without an API key
- **Deterministic FSL batch scheduler** — Batch 1: Immediate, Batch 2:
  Secondary, Batch 3: Archive — with transparent, documented rules
- **Human-in-the-loop review** — mandatory reason required for every override;
  full audit trail in the report
- **Policy governance layer** — Policy Admin proposes PDES weight adjustments;
  Approver must approve before activation; raw ML and policy-adjusted results
  always shown side-by-side

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **UI Framework** | Streamlit |
| **ML** | scikit-learn (Decision Tree, Random Forest, Gradient Boosting) |
| **Data** | pandas, numpy |
| **IBM Technologies** | IBM watsonx.ai (`granite-13b-instruct-v2`), IBM Bob |
| **Other** | python-dotenv, pytest |

---

## Repository Structure

```
src/
  app.py                    Streamlit UI (presentation/orchestration only)
  extractor.py              AI/NLP feature extraction (watsonx.ai + heuristic)
  scheduler.py              Deterministic FSL batch scheduler
  report.py                 Markdown report generator
  evidence_triage.py        FROZEN baseline (never modified)
  run_triage.py             FROZEN CLI harness (never modified)
  models/
    evidence_item.py        EvidenceItem + CaseContext dataclasses
    triage_models.py        ML module: DT / RF / GB training and prediction
    policy.py               PDES policy layer + governance workflow
  data/
    synthetic_dataset.py    148-row synthetic training dataset
  tests/
    test_triage_models.py   Unit tests: ML module (33 tests)
    test_scheduler.py       Unit tests: FSL scheduler (14 tests)
    test_report.py          Unit tests: report generator (14 tests) + 2 tests

docs/                       Architecture, setup guide, problem statement, solution overview
submission.yaml             Hackathon submission metadata
requirements.txt            pip dependencies
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/ibm-build-lab/bob-ai-hackathon-epsilon-infinity.git
cd bob-ai-hackathon-epsilon-infinity

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Configure watsonx.ai credentials
cp src/.env.example src/.env
# Edit src/.env — leave blank to use heuristic extraction

# 4. Launch the application
streamlit run src/app.py
```

The application opens at `http://localhost:8501`.

See [`docs/setup-guide.md`](docs/setup-guide.md) for full instructions.

---

## Demo

| Artifact | Link |
|---|---|
| Demo Video | [See demo/demo-video-link.txt](demo/demo-video-link.txt) |
| Live Demo | [See demo/live-demo-url.txt](demo/live-demo-url.txt) |
| Screenshots | [See demo/screenshots/](demo/screenshots/) |
| Presentation | [See presentation/](presentation/) |

---

## Running Tests

```bash
pip install pytest
python -m pytest src/tests/ -v
# Expected: 63 tests passed
```

---

## Known Limitations

- ML models are trained on a **148-row synthetic dataset** — accuracy figures
  are demonstrative only and do not represent real-world forensic performance
- Session state is **lost on page refresh** — no persistent database in this
  prototype
- Role selection in Policy Admin tab is a **simulation only** — no real
  authentication
- watsonx.ai extraction requires an IBM Cloud API key and `ibm-watsonx-ai`
  package; without these, heuristic extraction or manual entry is used
- PDES policy weights are a **configurable prototype mechanism**, not
  peer-reviewed forensic standards

---

## What We're Most Proud Of

The **human-in-the-loop design** and **explainability architecture**:

Every AI recommendation is labelled as a recommendation. The investigator
explicitly accepts or overrides each one with a mandatory recorded reason.
Decision Tree predictions show the exact per-item node-by-node rule path.
Random Forest and Gradient Boosting predictions show the item's own feature
values alongside global model importances — with a prominent disclaimer that
these are NOT per-item causal explanations. This transparency design directly
addresses the trust and accountability requirements for forensic applications.

---

> ⚠️ **IMPORTANT DISCLAIMERS**
>
> All AI/ML outputs in EvidencePro are **recommendations only**.
> The investigator is the **final decision-maker** for all triage and
> evidence submission decisions.
>
> Accuracy figures are from a **synthetic/demonstration dataset** and do not
> represent real-world forensic performance.
>
> Policy weights are a **CONFIGURABLE PROTOTYPE MECHANISM** and are NOT
> official forensic standards.
