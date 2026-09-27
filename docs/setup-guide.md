# EvidencePro — Setup Guide

> **This file is read by the automated evaluation pipeline. Be precise and complete.**

## Prerequisites

Before you begin, ensure you have the following installed:

- [x] **Python 3.10 or later** (tested on Python 3.14)
- [x] **pip** (bundled with Python)

No Docker, Node.js, or database is required.

## Clone the Repository

```bash
git clone https://github.com/ibm-build-lab/bob-ai-hackathon-epsilon-infinity.git
cd bob-ai-hackathon-epsilon-infinity
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `scikit-learn`, `pandas`, `numpy`, `streamlit>=1.32.0`,
`python-dotenv`.

### Optional: IBM watsonx.ai extraction

If you have an IBM Cloud API key and want to enable AI-powered evidence
feature extraction, install the optional SDK:

```bash
pip install ibm-watsonx-ai
```

The application runs fully without this package. Without it, evidence fields
are either pre-filled using heuristic keyword matching or entered manually.

## Environment Variables (Optional)

AI extraction via watsonx.ai requires credentials. All other features work
without any environment configuration.

```bash
cp src/.env.example src/.env
```

Edit `src/.env` and fill in your IBM Cloud credentials:

```
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

| Variable | Description | Required |
|---|---|---|
| `WATSONX_API_KEY` | IBM Cloud API key | No — enables watsonx.ai extraction |
| `WATSONX_PROJECT_ID` | watsonx.ai project ID | No — enables watsonx.ai extraction |
| `WATSONX_URL` | Inference endpoint URL | No — defaults to `us-south` |

## Run the Application

```bash
streamlit run src/app.py
```

The application will open automatically in your default browser at:
`http://localhost:8501`

## Typical Workflow

1. **Train a model** — click "Train / Retrain Model" in the sidebar
2. **Tab 1** — enter FIR number and offence type
3. **Tab 2** — add evidence items (optionally use "Extract with AI")
4. **Tab 3** — click "Run Triage"
5. **Tab 4** — review explanations
6. **Tab 5** — accept or override each recommendation
7. **Tab 6** — click "Build / Rebuild FSL Schedule"
8. **Tab 7** — click "Generate Report" and download the `.md` file

## Run the Tests

```bash
pip install pytest
python -m pytest src/tests/ -v
```

Expected result: **63 tests pass**.

## Run Validation Scripts (Optional)

```bash
python src/phase1_demo.py   # Phase 1: data model + dataset
python src/phase2_demo.py   # Phase 2: policy + scheduler + report
python src/phase3_smoke.py  # Phase 3: full pipeline smoke test
python src/phase4_demo.py   # Phase 4: extractor validation
```

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| App shows "Heuristic extraction" instead of watsonx.ai | Expected if `WATSONX_API_KEY` is not set. All features work normally. |
| Streamlit opens but shows an error on first load | Ensure you are running from the repo root: `streamlit run src/app.py` |
| `python -m pytest` fails with "No module named pytest" | Run `pip install pytest` first |
