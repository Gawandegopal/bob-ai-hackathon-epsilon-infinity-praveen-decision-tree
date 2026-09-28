"""
app.py
======
EvidencePro — Streamlit Investigator Platform
----------------------------------------------
Launch with:
    streamlit run src/app.py

IMPORTANT NOTICES
-----------------
• All AI/ML outputs in this application are RECOMMENDATIONS only.
  The investigator is the final decision-maker for all forensic triage
  and evidence submission decisions.

• This prototype is trained on SYNTHETIC / DEMONSTRATION DATA. Accuracy
  figures do not represent real-world forensic performance.

• Any policy weights are a CONFIGURABLE PROTOTYPE MECHANISM and are NOT
  official forensic standards.

Architecture
------------
This file is the presentation and orchestration layer only.
All ML, policy, scheduling, and report logic lives in the imported modules:

  document_processor.py    → document upload + Docling/fallback OCR
  extractor.py             → AI/NLP extraction (watsonx.ai / heuristic)
  models/triage_models.py  → ML training, prediction, explanation
  models/evidence_item.py  → EvidenceItem, CaseContext dataclasses
  models/policy.py         → TriagePolicy, governance, apply_policy
  scheduler.py             → build_schedule, ScheduledItem
  report.py                → generate_report, generate_pdf_report, ReportData
  feedback.py              → override monitoring, candidate retraining

No ML logic, policy formulas, or batch assignment rules are defined here.
"""

import sys
import os
from datetime import datetime, timezone

import streamlit as st

# Ensure src/ is on the path when launched from project root
sys.path.insert(0, os.path.dirname(__file__))

from extractor import extract_features, extraction_mode_label, is_watsonx_available

from models.evidence_item import (
    EvidenceItem, CaseContext, EVIDENCE_TYPES, OFFENCE_TYPES, PRIORITY_LABELS
)
from models.triage_models import (
    get_trained_model, predict_priority, validate_item, DATA_DISCLAIMER,
    TrainingResult,
)
from models.policy import (
    TriagePolicy, PolicyRole,
    POLICY_HISTORY,
    propose_policy, approve_policy, reject_policy, get_active_policy,
    apply_policy, policy_summary,
)
from scheduler import build_schedule, schedule_summary, BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE

from report import ReportData, generate_report, generate_pdf_report, generate_pdf_from_md, REPORTLAB_AVAILABLE

from document_processor import (
    process_uploaded_document,
    is_document_processing_available,
    processing_capability_summary,
    explain_document_processing,
    DOCLING_AVAILABLE,
    SUPPORTED_EXTENSIONS,
)

from feedback import (
    override_monitoring_report,
    request_candidate_retrain,
    evaluate_candidate,
    approve_candidate,
    reject_candidate,
    activate_candidate,
    get_active_candidate,
    candidate_status_summary,
    FEEDBACK_LOG,
    CANDIDATE_LOG,
    OVERRIDE_THRESHOLD_DEFAULT,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EvidencePro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "case_context":       None,        # CaseContext
        "evidence_items":     [],          # list[EvidenceItem]
        "training_result":    None,        # TrainingResult
        "triage_results":     [],          # list[TriageResult]
        "overrides":          {},          # dict[item_id -> (tier, reason)]
        "schedule":           [],          # list[ScheduledItem]
        "role":               "investigator",
        "model_name":         "decision_tree",
        "item_counter":       0,           # used to generate unique item IDs
        "triage_run":         False,       # True once Run Triage has been clicked
        "schedule_built":     False,
        "_extraction":        {},          # last extract_features() result (pre-fill cache)
        "_extract_desc":      "",          # description that was last extracted
        "_doc_result":        None,        # last DocumentResult from document_processor
        "_report_md":         None,        # generated Markdown report
        "_report_pdf":        None,        # generated PDF bytes
        # Workflow step (1-5) for guided mode
        "workflow_step":      1,
        # Override threshold
        "override_threshold": OVERRIDE_THRESHOLD_DEFAULT,
        # Session ID for feedback tracking
        "session_id":         datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "decision_tree":      "Decision Tree",
    "random_forest":      "Random Forest",
    "gradient_boosting":  "Gradient Boosting",
}

# Plain-language model descriptions for investigators (Feature 3)
MODEL_EXPLANATIONS = {
    "decision_tree": (
        "Uses a sequence of interpretable evidence characteristics to reach a priority "
        "recommendation. Each step tests one feature (e.g. perishability, probative value) "
        "and follows a clear branch. The exact decision path for each item can be shown — "
        "making this the most transparent model."
    ),
    "random_forest": (
        "Combines predictions from multiple decision trees to produce a more robust "
        "recommendation. It is less likely to overfit than a single tree, but the combined "
        "result cannot be traced through a single readable path. Global feature importances "
        "show which characteristics the model generally relies on most."
    ),
    "gradient_boosting": (
        "Builds an ensemble sequentially, with later trees focusing on correcting errors "
        "from earlier trees. This tends to achieve higher accuracy on the training data "
        "distribution, but is less directly interpretable. Feature importances are available "
        "at the model level."
    ),
}

ROLE_DISPLAY = {
    "investigator": "Investigator",
    "policy_admin": "Policy Admin",
    "approver":     "Approver",
}

PRIORITY_COLOURS = {
    "Critical": "🔴",
    "High":     "🟠",
    "Standard": "🟡",
    "Low":      "🟢",
}

ORDINAL_LABELS = {1: "Low (1)", 2: "Medium (2)", 3: "High (3)"}
BINARY_LABELS  = {0: "No", 1: "Yes"}

def _tier_badge(tier: str) -> str:
    return f"{PRIORITY_COLOURS.get(tier, '')} **{tier}**"

def _active_policy() -> TriagePolicy | None:
    return get_active_policy()

def _role_enum() -> PolicyRole:
    r = st.session_state.role
    return {
        "investigator": PolicyRole.INVESTIGATOR,
        "policy_admin": PolicyRole.POLICY_ADMIN,
        "approver":     PolicyRole.APPROVER,
    }[r]

def _next_item_id() -> str:
    st.session_state.item_counter += 1
    return f"E{st.session_state.item_counter:03d}"

def _retrain():
    with st.spinner(f"Training {MODEL_DISPLAY[st.session_state.model_name]}…"):
        st.session_state.training_result = get_trained_model(st.session_state.model_name)
    st.session_state.triage_run     = False
    st.session_state.triage_results = []
    st.session_state.schedule       = []
    st.session_state.schedule_built = False

def _reset_triage():
    """Clear triage/schedule/report state when items change."""
    st.session_state.triage_run     = False
    st.session_state.triage_results = []
    st.session_state.schedule       = []
    st.session_state.schedule_built = False
    st.session_state._report_md     = None
    st.session_state._report_pdf    = None

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 EvidencePro")
    st.caption("AI-Assisted Forensic Evidence Triage · Prototype")

    st.warning(
        "⚠️ **AI Recommendations Only**\n\n"
        "All outputs are AI-assisted recommendations. "
        "The investigator is the **final decision-maker**.",
        icon=None,
    )

    st.divider()

    # Role selector (prototype simulation)
    st.subheader("Role")
    st.selectbox(
        "Session role (prototype simulation — no real authentication)",
        options=list(ROLE_DISPLAY.keys()),
        format_func=lambda k: ROLE_DISPLAY[k],
        key="role",
    )

    st.divider()

    # Model selector + training
    st.subheader("ML Model")
    st.selectbox(
        "Model",
        options=list(MODEL_DISPLAY.keys()),
        format_func=lambda k: MODEL_DISPLAY[k],
        key="model_name",
    )

    if st.button("Train / Retrain Model", use_container_width=True):
        _retrain()
        st.success(f"Model trained: {MODEL_DISPLAY[st.session_state.model_name]}")

    if st.session_state.training_result is not None:
        tr = st.session_state.training_result
        st.caption(
            f"Active: **{MODEL_DISPLAY[tr.model_name]}** · "
            f"CV accuracy: {tr.cv_accuracy * 100:.1f}%"
        )
        st.caption(f"⚠️ {DATA_DISCLAIMER}")

        with st.expander("ℹ️ How this model works"):
            st.caption(MODEL_EXPLANATIONS.get(tr.model_name, ""))
            st.caption(
                "**What is cross-validation?** The dataset was split into 5 equal parts. "
                "The model was trained on 4 parts and tested on the remaining 1, five times. "
                "The reported accuracy is the average across all 5 test sets. "
                "This provides a more reliable estimate than a single train/test split."
            )
            st.caption(
                "**Synthetic data caveat:** "
                "Model evaluation is based on 5-fold cross-validation using the synthetic "
                "demonstration dataset. These results demonstrate prototype behavior and "
                "must not be interpreted as validated real-world forensic accuracy."
            )
            st.caption(
                f"Decision Tree: ~79.7% | Random Forest: ~83.9% | Gradient Boosting: ~84.5% "
                f"(on synthetic demo dataset)"
            )
    else:
        st.info("No model trained yet. Click **Train / Retrain Model** above.")

    st.divider()

    # Active policy display
    st.subheader("Active Policy")
    active_pol = _active_policy()
    if active_pol:
        st.success(
            f"**{active_pol.label}** · v{active_pol.version}\n\n"
            f"P={active_pol.weight_P} · D={active_pol.weight_D} · "
            f"E={active_pol.weight_E} · S={active_pol.weight_S}"
        )
        st.caption(
            "⚠️ Policy weights are a CONFIGURABLE PROTOTYPE MECHANISM, "
            "not official forensic standards."
        )
    else:
        st.info("No active policy — unweighted ML in use.")

    st.divider()

    # Document processing capability
    st.subheader("Document Processing")
    doc_cap = processing_capability_summary()
    if DOCLING_AVAILABLE:
        st.success(f"✅ {doc_cap}")
    elif is_document_processing_available():
        st.info(f"📄 {doc_cap}")
    else:
        st.warning("⚠️ No document processing libraries available. Manual entry only.")

    # Extraction mode
    st.caption(f"AI extraction: **{extraction_mode_label()}**")

    st.divider()

    # Override monitoring summary (Feature 5)
    items_count = len(st.session_state.evidence_items)
    overrides_count = len(st.session_state.overrides)
    if items_count > 0:
        st.subheader("Override Monitor")
        mon = override_monitoring_report(
            st.session_state.overrides,
            items_count,
            st.session_state.override_threshold,
        )
        if mon["retraining_recommended"]:
            st.error(
                f"🔄 **Retraining Recommended**\n\n"
                f"{mon['message']}"
            )
        else:
            st.info(f"Override rate: {mon['override_pct']:.1f}%\n\n{mon['message']}")

# ---------------------------------------------------------------------------
# MAIN CONTENT — TABS
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "1 · Case & Upload",
    "2 · Evidence",
    "3 · Run Triage",
    "4 · Explanations",
    "5 · Review",
    "6 · Schedule",
    "7 · Report",
    "8 · Policy Admin",
    "9 · Model Feedback",
])

# ===========================================================================
# TAB 1 — CASE CONTEXT + DOCUMENT UPLOAD
# ===========================================================================
with tab1:
    st.header("Step 1 — Case Details and Document Upload")
    st.caption(
        "Start by uploading a case document (PDF, image, or DOCX) to automatically "
        "extract case information, or enter details manually."
    )

    # ---- DOCUMENT UPLOAD PANEL (Feature 1) ----
    st.subheader("📎 Option A — Upload Case Document")

    if not is_document_processing_available():
        st.warning(
            "⚠️ No document processing libraries are installed. "
            "To enable document upload, install `docling` (recommended) or "
            "`pymupdf` / `pdfplumber` for PDFs, `python-docx` for DOCX files. "
            "Please use **Option B — Manual Entry** below."
        )
    else:
        st.caption(
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))} · "
            f"Processing: {processing_capability_summary()}"
        )
        if not DOCLING_AVAILABLE:
            st.info(
                "💡 **Tip:** Install `docling` for full OCR support on scanned PDFs and images. "
                "Current fallback can only extract text from PDFs with a text layer and DOCX files."
            )

        uploaded_file = st.file_uploader(
            "Upload forensic requisition or case document",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            help="Upload a PDF, image, or DOCX containing case/evidence information.",
        )

        if uploaded_file is not None:
            with st.spinner("Processing document…"):
                doc_result = process_uploaded_document(
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name,
                )
            st.session_state._doc_result = doc_result

            if doc_result.success:
                st.success(
                    f"✅ Document processed: **{uploaded_file.name}** "
                    f"({doc_result.pages} page(s), method: `{doc_result.method}`)"
                )
                if doc_result.warning:
                    st.caption(f"ℹ️ {doc_result.warning}")

                with st.expander("📄 View extracted document text", expanded=False):
                    st.text_area(
                        "Extracted text (AI extraction suggestion — review and confirm all values)",
                        value=doc_result.text,
                        height=250,
                        disabled=True,
                        label_visibility="visible",
                    )
                    st.info(
                        "🤖 The text above was extracted from your document. "
                        "Use it as a reference when completing the **Case Context** form below "
                        "and the **Evidence Items** form in Step 2."
                    )
            else:
                st.warning(
                    f"⚠️ Could not extract text from **{uploaded_file.name}**. "
                    f"{doc_result.warning}"
                )
                st.info("Please complete the case details manually using the form below.")

    st.divider()
    st.subheader("📝 Option B — Enter Case Details Manually")

    # Pre-fill narrative from document text if available
    _doc_narrative = ""
    if st.session_state._doc_result and st.session_state._doc_result.success:
        _doc_narrative = (
            f"[Extracted from document: {st.session_state._doc_result.filename}]\n\n"
            + st.session_state._doc_result.text[:500]
        )

    with st.form("case_context_form"):
        fir_number = st.text_input("FIR / Case Number *", placeholder="e.g. FIR-2024-001")
        offence_type = st.selectbox("Offence Type *", options=OFFENCE_TYPES)
        narrative = st.text_area(
            "Case Narrative (optional)",
            value=_doc_narrative if not st.session_state.case_context else (
                st.session_state.case_context.narrative
            ),
            placeholder="Brief description of the incident…",
            height=120,
        )
        submitted = st.form_submit_button("Save Case Context ▶", use_container_width=True)

    if submitted:
        if not fir_number.strip():
            st.error("FIR / Case Number is required.")
        else:
            st.session_state.case_context = CaseContext(
                fir_number=fir_number.strip(),
                offence_type=offence_type,
                narrative=narrative.strip(),
            )
            st.success(f"✅ Case context saved: **{fir_number}** · {offence_type}")

    if st.session_state.case_context:
        ctx = st.session_state.case_context
        st.info(
            f"**Current case:** {ctx.fir_number} · {ctx.offence_type}"
            + (f"\n\n_{ctx.narrative[:200]}_" if ctx.narrative else "")
        )
        st.caption("➡️ Proceed to **Step 2 — Evidence** to add evidence items.")

# ===========================================================================
# TAB 2 — EVIDENCE ITEMS
# ===========================================================================
with tab2:
    st.header("Step 2 — Evidence Items")
    st.caption(
        "Add each piece of evidence. Use **Extract with AI** to pre-fill fields from a "
        "description, or fill in the form manually. All AI suggestions must be reviewed "
        "and confirmed before adding."
    )

    if not st.session_state.case_context:
        st.warning("⬅️ Please complete **Step 1 — Case Details** first.")
    else:
        ctx = st.session_state.case_context

        # ---- EXTRACTION MODE BANNER ----
        ext_mode = extraction_mode_label()
        if is_watsonx_available():
            st.success(f"🤖 AI extraction: **{ext_mode}**")
        else:
            st.info(f"ℹ️ AI extraction: **{ext_mode}** — watsonx.ai not configured")

        # ---- ADD EVIDENCE FORM ----
        with st.expander("➕ Add a new evidence item", expanded=True):

            # ------------------------------------------------------------------
            # AI EXTRACTION PANEL (outside the form — needs its own submit)
            # ------------------------------------------------------------------
            st.markdown("#### AI-Assisted Feature Extraction *(optional)*")
            st.caption(
                "Enter a description and click **Extract with AI** to pre-fill the form. "
                "All extracted values are suggestions — review every field before adding."
            )

            ex = st.session_state._extraction  # current extraction result (may be empty)

            # Pre-fill description from document if available
            _doc_text_hint = ""
            if st.session_state._doc_result and st.session_state._doc_result.success:
                _doc_text_hint = st.session_state._doc_result.text[:200]

            desc_input = st.text_area(
                "Evidence description (free text)",
                value=st.session_state._extract_desc or _doc_text_hint,
                placeholder=(
                    "e.g. Blood swab collected from victim's clothing at the scene, "
                    "stored in sealed forensic bag."
                ),
                height=90,
                key="_desc_textarea",
            )

            if st.button(
                "🔍 Extract with AI",
                help="Analyse the description and suggest field values. Correct any value before adding.",
                use_container_width=False,
            ):
                if not desc_input.strip():
                    st.warning("Enter a description before extracting.")
                else:
                    with st.spinner("Extracting features…"):
                        extracted = extract_features(
                            description=desc_input,
                            context=ctx,
                        )
                    st.session_state._extraction   = extracted
                    st.session_state._extract_desc = desc_input
                    ex = extracted
                    source = extracted.get("_source", "none")
                    if source == "watsonx.ai":
                        st.success("✅ Extracted via IBM watsonx.ai. Review all values below.")
                    elif source == "heuristic":
                        st.info("ℹ️ Heuristic extraction (no API key). Some fields pre-filled from keywords. Review carefully.")
                    else:
                        st.warning("No features could be extracted. Please fill in the form manually.")

            def _ai_badge(field: str) -> str:
                if field in ex and "_source" in ex and ex.get("_source") != "none":
                    return " 🤖"
                return ""

            st.divider()

            # ------------------------------------------------------------------
            # EVIDENCE FORM
            # ------------------------------------------------------------------
            with st.form("add_evidence_form", clear_on_submit=True):
                st.markdown("**Item details**")
                label = st.text_input(
                    "Short label / description *",
                    value=desc_input if ex else "",
                    placeholder="e.g. Blood swab from victim's clothing",
                )

                col1, col2 = st.columns(2)
                with col1:
                    _et_default = ex.get("evidence_type", EVIDENCE_TYPES[0])
                    _et_index   = EVIDENCE_TYPES.index(_et_default) if _et_default in EVIDENCE_TYPES else 0
                    evidence_type = st.selectbox(
                        f"Evidence Type{_ai_badge('evidence_type')}",
                        EVIDENCE_TYPES,
                        index=_et_index,
                    )
                with col2:
                    st.text_input("Offence Type (from case context)", value=ctx.offence_type, disabled=True)

                st.markdown("**PDES dimensions** — used by the ML model")
                st.caption(
                    "P = Probative Value · D = Degradation/Perishability Risk · "
                    "E = Exclusionary Power · S = Processing Speed (via lead time)"
                )

                col_p, col_d, col_e = st.columns(3)
                with col_p:
                    _pv_default = int(ex.get("probative_value", 2)) - 1
                    probative_value = st.selectbox(
                        f"Probative Value (P){_ai_badge('probative_value')}",
                        options=[1, 2, 3],
                        format_func=lambda v: ORDINAL_LABELS[v],
                        index=max(0, min(2, _pv_default)),
                        help="How strongly does this evidence tend to prove or disprove a fact?",
                    )
                with col_d:
                    _per_default = int(ex.get("perishability", 1)) - 1
                    perishability = st.selectbox(
                        f"Degradation Risk (D){_ai_badge('perishability')}",
                        options=[1, 2, 3],
                        format_func=lambda v: {1: "Stable (1)", 2: "Degrades over days (2)", 3: "Degrades in hours (3)"}[v],
                        index=max(0, min(2, _per_default)),
                        help="How quickly does this evidence degrade without processing?",
                    )
                with col_e:
                    _ep_default = int(ex.get("exclusionary_power", 2)) - 1
                    exclusionary_power = st.selectbox(
                        f"Exclusionary Power (E){_ai_badge('exclusionary_power')}",
                        options=[1, 2, 3],
                        format_func=lambda v: ORDINAL_LABELS[v],
                        index=max(0, min(2, _ep_default)),
                        help="How effectively can this evidence exclude suspects?",
                    )

                st.markdown("**Secondary features**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    _cr_default = int(ex.get("contamination_risk", 1)) - 1
                    contamination_risk = st.selectbox(
                        f"Contamination Risk{_ai_badge('contamination_risk')}",
                        options=[1, 2, 3],
                        format_func=lambda v: ORDINAL_LABELS[v],
                        index=max(0, min(2, _cr_default)),
                    )
                with col_b:
                    _sr_default = int(ex.get("specialist_required", 0))
                    specialist_required = st.selectbox(
                        f"Specialist Required{_ai_badge('specialist_required')}",
                        options=[0, 1],
                        format_func=lambda v: BINARY_LABELS[v],
                        index=max(0, min(1, _sr_default)),
                    )
                with col_c:
                    _lt_default = int(ex.get("testing_lead_time", 7))
                    testing_lead_time = st.number_input(
                        f"Testing Lead Time (days){_ai_badge('testing_lead_time')}",
                        min_value=1, max_value=60,
                        value=max(1, min(60, _lt_default)),
                        help="Approximate laboratory turnaround in days.",
                    )

                st.markdown("**Operational details** *(not used in ML model)*")
                col_op1, col_op2, col_op3 = st.columns(3)
                with col_op1:
                    collection_age_hours = st.number_input(
                        "Collection Age (hours)",
                        min_value=0, max_value=720, value=0,
                        help="Hours since evidence was collected. 0 = unknown.",
                    )
                with col_op2:
                    _ec_default = int(ex.get("evidence_condition", 2)) - 1
                    evidence_condition = st.selectbox(
                        f"Evidence Condition{_ai_badge('evidence_condition')}",
                        options=[1, 2, 3],
                        format_func=lambda v: {1: "Poor (1)", 2: "Fair (2)", 3: "Good (3)"}[v],
                        index=max(0, min(2, _ec_default)),
                    )
                with col_op3:
                    specialist_type = st.text_input(
                        f"Specialist Type (optional){_ai_badge('specialist_type')}",
                        value=ex.get("specialist_type", ""),
                        placeholder="e.g. DNA analyst",
                    )

                if ex and any(k not in ("_source", "_partial") for k in ex):
                    st.info(
                        f"🤖 Fields marked with **🤖** were suggested by AI extraction "
                        f"({ex.get('_source', 'unknown')} mode). "
                        "Review and correct them before adding the item. "
                        "**All AI suggestions require investigator confirmation.**"
                    )

                add_submitted = st.form_submit_button("✅ Add Evidence Item", use_container_width=True)

            if add_submitted:
                if not label.strip():
                    st.error("Label / description is required.")
                else:
                    _ai_extracted = bool(ex and any(k not in ("_source", "_partial") for k in ex))
                    new_item = EvidenceItem(
                        item_id=_next_item_id(),
                        label=label.strip(),
                        evidence_type=evidence_type,
                        offence_type=ctx.offence_type,
                        probative_value=probative_value,
                        perishability=perishability,
                        exclusionary_power=exclusionary_power,
                        contamination_risk=contamination_risk,
                        specialist_required=specialist_required,
                        testing_lead_time=int(testing_lead_time),
                        collection_age_hours=int(collection_age_hours),
                        evidence_condition=evidence_condition,
                        specialist_type=specialist_type.strip(),
                        ai_extracted=_ai_extracted,
                    )
                    errors = validate_item(new_item)
                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        st.session_state.evidence_items.append(new_item)
                        st.session_state._extraction   = {}
                        st.session_state._extract_desc = ""
                        _reset_triage()
                        ai_note = " *(AI-assisted)*" if _ai_extracted else ""
                        st.success(f"✅ Item added: **{new_item.item_id}** — {new_item.label}{ai_note}")

        # ---- EVIDENCE LIST ----
        items = st.session_state.evidence_items
        if not items:
            st.info("No evidence items yet. Use the form above to add items.")
        else:
            st.subheader(f"Evidence items ({len(items)})")
            remove_id = None
            for item in items:
                with st.expander(f"**{item.item_id}** · {item.label}", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Evidence type", item.evidence_type.replace("_", " "))
                    c2.metric("Probative (P)", item.probative_value)
                    c3.metric("Degradation (D)", item.perishability)
                    c4.metric("Exclusionary (E)", item.exclusionary_power)
                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Lead time (days)", item.testing_lead_time)
                    c6.metric("Contamination risk", item.contamination_risk)
                    c7.metric("Specialist req.", "Yes" if item.specialist_required else "No")
                    c8.metric("Collection age (h)", item.collection_age_hours if item.collection_age_hours else "Unknown")
                    if item.ai_extracted:
                        st.caption("🤖 This item was added using AI-assisted extraction.")
                    if st.button(f"Remove {item.item_id}", key=f"remove_{item.item_id}"):
                        remove_id = item.item_id
            if remove_id:
                st.session_state.evidence_items = [
                    i for i in st.session_state.evidence_items if i.item_id != remove_id
                ]
                _reset_triage()
                st.rerun()

            st.caption(f"➡️ When all items are added, proceed to **Step 3 — Run Triage**.")

# ===========================================================================
# TAB 3 — RUN TRIAGE
# ===========================================================================
with tab3:
    st.header("Step 3 — Run Triage")
    st.caption(
        "The ML model classifies each evidence item and assigns a priority tier. "
        "All results are **AI-assisted recommendations** — the investigator reviews "
        "and makes the final decision in Step 5."
    )

    items = st.session_state.evidence_items
    tr    = st.session_state.training_result

    col_run1, col_run2 = st.columns([2, 1])
    with col_run1:
        if not st.session_state.case_context:
            st.warning("Complete Step 1 (Case Details) first.")
        elif not items:
            st.warning("Add at least one evidence item in Step 2.")
        elif tr is None:
            st.warning("Train a model using the sidebar first.")
        else:
            if st.button(
                f"▶ Run Triage ({len(items)} item{'s' if len(items) != 1 else ''})",
                use_container_width=True,
                type="primary",
            ):
                active_pol = _active_policy()
                with st.spinner("Running triage…"):
                    results = [predict_priority(item, tr) for item in items]
                st.session_state.triage_results  = results
                st.session_state.triage_run      = True
                st.session_state.schedule_built  = False
                st.session_state.schedule        = []
                st.success(f"✅ Triage complete. {len(results)} item(s) classified.")

    with col_run2:
        if tr:
            st.info(
                f"Model: **{MODEL_DISPLAY[tr.model_name]}**\n\n"
                f"CV accuracy: {tr.cv_accuracy*100:.1f}%\n\n"
                f"⚠️ Synthetic data only"
            )

    # --- Model explanation (Feature 3) ---
    if tr:
        with st.expander("ℹ️ How does this model work?", expanded=False):
            st.markdown(f"**{MODEL_DISPLAY[tr.model_name]}**")
            st.caption(MODEL_EXPLANATIONS.get(tr.model_name, ""))
            st.divider()
            st.markdown("**Model Evaluation Transparency**")
            st.caption(
                "Model evaluation is based on 5-fold cross-validation using the synthetic "
                "demonstration dataset. These results demonstrate prototype behavior and "
                "must not be interpreted as validated real-world forensic accuracy."
            )
            cv_pct = tr.cv_accuracy * 100
            st.caption(
                f"Current model ({MODEL_DISPLAY[tr.model_name]}) CV accuracy: **{cv_pct:.1f}%**\n\n"
                "Reference: Decision Tree ~79.7% · Random Forest ~83.9% · Gradient Boosting ~84.5%\n\n"
                "These figures are on the same 180-row synthetic dataset. They are not comparable "
                "to accuracy on real-world casework."
            )

    if st.session_state.triage_run and st.session_state.triage_results:
        active_pol = _active_policy()
        results    = st.session_state.triage_results

        st.divider()
        st.subheader("Triage Results")
        st.caption(
            "🔴 Critical · 🟠 High · 🟡 Standard · 🟢 Low\n\n"
            "⚠️ = Urgent (perishable or recently collected degrading evidence)"
        )

        import pandas as pd
        rows = []
        for item, result in zip(items, results):
            pol_result = apply_policy(result, item, active_pol) if active_pol else None
            urgency = "⚠️ YES" if result.urgency_flag else "No"
            raw_tier = f"{PRIORITY_COLOURS.get(result.priority_tier, '')} {result.priority_tier}"
            pol_tier = (
                f"{PRIORITY_COLOURS.get(pol_result.priority_tier, '')} {pol_result.priority_tier}"
                if pol_result and pol_result.priority_tier != result.priority_tier
                else "—"
            )
            rows.append({
                "ID": item.item_id,
                "Label": item.label,
                "Raw ML Tier": raw_tier,
                "Policy Tier": pol_tier,
                "Urgency": urgency,
                "Score": f"{result.priority_score:.3f}",
                "Specialist": "Yes" if item.specialist_required else "No",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if active_pol:
            st.info(
                f"Active policy: **{active_pol.label}** v{active_pol.version} · "
                "Policy Tier column shows adjusted tier where it differs from raw ML tier.\n\n"
                "⚠️ Policy weights are a CONFIGURABLE PROTOTYPE MECHANISM, not official forensic standards."
            )
        else:
            st.caption("No active policy — raw ML tiers shown.")

        st.caption("➡️ Review explanations in **Step 4**, then confirm or override in **Step 5**.")

# ===========================================================================
# TAB 4 — EXPLANATIONS
# ===========================================================================
with tab4:
    st.header("Step 4 — Explanations")
    st.caption(
        "Understand why each item received its recommendation. "
        "**Decision Tree** shows the exact rule path followed. "
        "**Random Forest / Gradient Boosting** show feature values and global model importances."
    )

    if not st.session_state.triage_run or not st.session_state.triage_results:
        st.info("Run triage in Step 3 first.")
    else:
        items   = st.session_state.evidence_items
        results = st.session_state.triage_results
        active_pol = _active_policy()
        tr = st.session_state.training_result

        # Model explanation banner (Feature 3)
        if tr:
            st.info(
                f"**{MODEL_DISPLAY[tr.model_name]}:** "
                + MODEL_EXPLANATIONS.get(tr.model_name, "")
            )

        for item, result in zip(items, results):
            pol_result = apply_policy(result, item, active_pol) if active_pol else None
            urgent_str = " · ⚠️ **URGENT**" if result.urgency_flag else ""

            with st.expander(
                f"**{item.item_id}** · {item.label} — {PRIORITY_COLOURS.get(result.priority_tier,'')} {result.priority_tier}{urgent_str}",
                expanded=False,
            ):
                col_raw, col_pol = st.columns(2) if pol_result else (st.container(), None)

                with col_raw:
                    st.markdown(f"##### Raw ML Recommendation")
                    st.markdown(f"**Priority tier:** {_tier_badge(result.priority_tier)}")
                    st.markdown(f"**Confidence score:** {result.priority_score:.3f}")
                    if result.urgency_flag:
                        st.error("⚠️ URGENT — this evidence requires immediate attention.")
                    st.markdown("**PDES feature values for this item:**")
                    st.markdown(
                        f"- P (Probative Value): **{result.item_feature_values.get('probative_value', '?')}**\n"
                        f"- D (Degradation Risk): **{result.item_feature_values.get('perishability', '?')}**\n"
                        f"- E (Exclusionary Power): **{result.item_feature_values.get('exclusionary_power', '?')}**\n"
                        f"- S (Lead time): **{result.item_feature_values.get('testing_lead_time', '?')} days**"
                    )
                    if result.decision_path:
                        st.markdown("**Decision path (actual rules for this item):**")
                        st.code(result.decision_path, language=None)
                    elif result.model_importances:
                        import pandas as pd
                        st.markdown("**This item's feature values:**")
                        fv_data = [
                            {"Feature": k, "Value": str(v)}
                            for k, v in result.item_feature_values.items()
                        ]
                        st.dataframe(pd.DataFrame(fv_data), use_container_width=True, hide_index=True)

                        st.markdown("**Global model feature importance** *(not per-item causation)*:")
                        st.caption(
                            "These importances describe which features the trained model generally "
                            "relies on most. They are NOT a causal explanation of this specific item."
                        )
                        fi_data = [
                            {"Rank": e["rank"], "Feature": e["feature"], "Importance": f"{e['importance']:.4f}"}
                            for e in result.model_importances
                        ]
                        st.dataframe(pd.DataFrame(fi_data), use_container_width=True, hide_index=True)

                if pol_result and col_pol:
                    with col_pol:
                        st.markdown(f"##### Policy-Adjusted Recommendation")
                        st.caption(
                            f"Active policy: **{active_pol.label}** v{active_pol.version}\n\n"
                            "⚠️ CONFIGURABLE PROTOTYPE MECHANISM — not an official forensic standard."
                        )
                        changed = pol_result.priority_tier != result.priority_tier
                        if changed:
                            st.warning(
                                f"Policy changes tier: **{result.priority_tier}** → "
                                f"**{pol_result.priority_tier}**"
                            )
                        else:
                            st.info(f"Policy-adjusted tier: **{pol_result.priority_tier}** (unchanged)")
                        st.markdown(f"**Policy-adjusted score:** {pol_result.priority_score:.4f}")

                st.caption(
                    "⚠️ This is an AI-assisted recommendation. "
                    "The investigator is the final decision-maker."
                )

# ===========================================================================
# TAB 5 — HUMAN REVIEW
# ===========================================================================
with tab5:
    st.header("Step 5 — Human Review")
    st.caption(
        "Review each AI recommendation. **Accept** it or **override** it with a reason. "
        "Your decision — not the AI's — determines the final tier used for scheduling. "
        "Individual overrides do **not** change the system-wide policy."
    )

    if not st.session_state.triage_run or not st.session_state.triage_results:
        st.info("Run triage in Step 3 first.")
    else:
        items      = st.session_state.evidence_items
        results    = st.session_state.triage_results
        active_pol = _active_policy()
        overrides  = st.session_state.overrides

        for item, result in zip(items, results):
            pol_result   = apply_policy(result, item, active_pol) if active_pol else None
            display_tier = pol_result.priority_tier if pol_result else result.priority_tier
            is_overridden = item.item_id in overrides
            current_override = overrides.get(item.item_id, (None, None))

            with st.expander(
                f"**{item.item_id}** · {item.label} "
                f"— AI: {PRIORITY_COLOURS.get(result.priority_tier,'')} {result.priority_tier}"
                + (" · ✏️ Overridden" if is_overridden else " · ✅ Accepted"),
                expanded=not is_overridden,
            ):
                col_info, col_action = st.columns([2, 3])
                with col_info:
                    st.markdown(f"**Raw ML:** {_tier_badge(result.priority_tier)} (score: {result.priority_score:.3f})")
                    if pol_result:
                        st.markdown(f"**Policy-adjusted:** {_tier_badge(pol_result.priority_tier)}")
                    if result.urgency_flag:
                        st.warning("⚠️ Urgency flag is set.")
                    if is_overridden:
                        ov_tier, ov_reason = current_override
                        st.success(f"✏️ **Your decision:** {_tier_badge(ov_tier)}\n\n**Reason:** {ov_reason}")

                with col_action:
                    if is_overridden:
                        if st.button(
                            f"Revert to AI recommendation ({display_tier})",
                            key=f"revert_{item.item_id}",
                        ):
                            del st.session_state.overrides[item.item_id]
                            st.session_state.schedule_built = False
                            st.rerun()
                    else:
                        st.markdown("**Override this recommendation?**")
                        with st.form(key=f"override_form_{item.item_id}"):
                            new_tier = st.selectbox(
                                "Change tier to",
                                options=PRIORITY_LABELS,
                                index=PRIORITY_LABELS.index(display_tier) if display_tier in PRIORITY_LABELS else 0,
                                key=f"tier_sel_{item.item_id}",
                            )
                            reason = st.text_input(
                                "Override reason (required)",
                                placeholder="e.g. Physical connection to suspect confirmed on-scene",
                                key=f"reason_{item.item_id}",
                            )
                            if st.form_submit_button("Save override"):
                                if not reason.strip():
                                    st.error("An override reason is required.")
                                elif new_tier == display_tier:
                                    st.warning("Selected tier is the same as the recommendation — no override saved.")
                                else:
                                    st.session_state.overrides[item.item_id] = (new_tier, reason.strip())
                                    st.session_state.schedule_built = False
                                    st.rerun()

        st.divider()
        st.subheader("Review Summary")
        ov_count  = len(overrides)
        acc_count = len(items) - ov_count
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Items reviewed", len(items))
        col_s2.metric("Accepted", acc_count)
        col_s3.metric("Overridden", ov_count)

        # Override monitoring (Feature 5)
        if len(items) > 0:
            mon = override_monitoring_report(
                overrides, len(items), st.session_state.override_threshold
            )
            if mon["retraining_recommended"]:
                st.error(f"🔄 **{mon['message']}**")
                st.caption(
                    "An override does NOT automatically mean the model was wrong. "
                    "See **Step 9 — Model Feedback** for the candidate retraining workflow."
                )
            elif ov_count > 0:
                st.info(mon["message"])

        st.caption("➡️ Proceed to **Step 6 — Schedule** to build the FSL examination schedule.")

# ===========================================================================
# TAB 6 — FSL SCHEDULE
# ===========================================================================
with tab6:
    st.header("Step 6 — FSL Examination Schedule")
    st.caption(
        "The schedule assigns each evidence item to an examination batch using "
        "**deterministic, rule-based logic** — not another ML model. "
        "Batch assignment is based on final tier, urgency, and specialist requirement."
    )

    if not st.session_state.triage_run or not st.session_state.triage_results:
        st.info("Run triage in Step 3 first.")
    else:
        items   = st.session_state.evidence_items
        results = st.session_state.triage_results

        if st.button("Build / Rebuild FSL Schedule", use_container_width=True, type="primary"):
            active_pol = _active_policy()
            overrides  = st.session_state.overrides
            with st.spinner("Building schedule…"):
                schedule = build_schedule(items, results, overrides, active_pol)
            st.session_state.schedule       = schedule
            st.session_state.schedule_built = True
            st.success(f"✅ Schedule built — {len(schedule)} item(s) assigned to batches.")

        if st.session_state.schedule_built and st.session_state.schedule:
            schedule = st.session_state.schedule
            groups   = schedule_summary(schedule)

            batch_info = {
                BATCH_IMMEDIATE: ("🔴", "Critical priority or urgent evidence — immediate laboratory examination"),
                BATCH_SECONDARY: ("🟠", "High priority (non-urgent) or Standard with specialist requirement"),
                BATCH_ARCHIVE:   ("🟢", "Low priority or Standard with no specialist requirement — routine processing"),
            }

            for batch_label in [BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE]:
                icon, desc = batch_info[batch_label]
                batch_items = groups[batch_label]
                st.subheader(f"{icon} {batch_label}")
                st.caption(desc)

                if not batch_items:
                    st.info("No items in this batch.")
                else:
                    for rank, si in enumerate(batch_items, 1):
                        urgency_tag = " · ⚠️ URGENT" if si.triage_result.urgency_flag else ""
                        ov_tag      = " · ✏️ Overridden" if si.investigator_decision == "overridden" else ""
                        with st.container():
                            col_n, col_body = st.columns([1, 9])
                            col_n.markdown(f"**#{rank}**")
                            with col_body:
                                st.markdown(
                                    f"**{si.item.item_id}** · {si.item.label}{urgency_tag}{ov_tag}\n\n"
                                    f"Final tier: {_tier_badge(si.final_tier)} · "
                                    f"Lead time: {si.item.testing_lead_time}d · "
                                    f"Specialist: {'Yes' if si.item.specialist_required else 'No'}"
                                )
                                st.caption(f"Reason: {si.batch_reason}")
                                if si.override_reason:
                                    st.caption(f"Override reason: {si.override_reason}")
                        st.divider()

            st.caption(
                "Items within each batch are sorted by degradation risk (highest first) "
                "then by testing lead time (shortest first)."
            )
            st.caption("➡️ Proceed to **Step 7 — Report** to generate the final PDF report.")

# ===========================================================================
# TAB 7 — REPORT
# ===========================================================================
with tab7:
    st.header("Step 7 — Report")
    st.caption(
        "Generate a professional forensic triage report. "
        "The PDF report contains natural-language sections describing the actual case findings."
    )

    can_report = (
        st.session_state.case_context is not None
        and st.session_state.triage_run
        and len(st.session_state.triage_results) > 0
        and st.session_state.schedule_built
        and len(st.session_state.schedule) > 0
    )

    if not can_report:
        missing = []
        if not st.session_state.case_context:
            missing.append("Case context (Step 1)")
        if not st.session_state.triage_run:
            missing.append("Triage results (Step 3)")
        if not st.session_state.schedule_built:
            missing.append("FSL schedule (Step 6)")
        st.warning("To generate a report, complete: " + ", ".join(missing))
    else:
        col_gen, col_info = st.columns([2, 1])
        with col_gen:
            if st.button("Generate Report", use_container_width=True, type="primary"):
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                active_pol = _active_policy()
                tr = st.session_state.training_result

                report_data = ReportData(
                    context=st.session_state.case_context,
                    items=st.session_state.evidence_items,
                    results=st.session_state.triage_results,
                    schedule=st.session_state.schedule,
                    policy=active_pol,
                    generated_at=now,
                    model_used=MODEL_DISPLAY.get(tr.model_name, tr.model_name) if tr else "Unknown",
                    cv_accuracy=tr.cv_accuracy if tr else 0.0,
                )

                # Generate Markdown report (always)
                report_md = generate_report(report_data)
                st.session_state["_report_md"] = report_md
                st.session_state["_report_data"] = report_data

                # Generate ReportLab PDF (preferred)
                pdf_bytes = None
                pdf_method = ""
                if REPORTLAB_AVAILABLE:
                    with st.spinner("Generating professional A4 PDF…"):
                        pdf_bytes = generate_pdf_report(report_data)
                    if pdf_bytes:
                        pdf_method = "ReportLab"
                    else:
                        st.warning("ReportLab PDF generation failed. Trying xhtml2pdf fallback…")

                # Fallback: xhtml2pdf from Markdown
                if not pdf_bytes:
                    with st.spinner("Generating PDF (fallback method)…"):
                        pdf_bytes = generate_pdf_from_md(report_md)
                    if pdf_bytes:
                        pdf_method = "xhtml2pdf"
                    else:
                        st.warning(
                            "PDF generation is not available. "
                            "Install `reportlab` (recommended) or `xhtml2pdf`. "
                            "Markdown download is still available."
                        )

                st.session_state["_report_pdf"] = pdf_bytes
                if pdf_bytes:
                    st.success(f"✅ Report generated (Markdown + A4 PDF via {pdf_method}).")
                else:
                    st.success("✅ Report generated (Markdown only — PDF unavailable).")

        with col_info:
            if REPORTLAB_AVAILABLE:
                st.success("📄 ReportLab available — professional A4 PDF supported")
            else:
                st.info("📄 Install `reportlab` for professional A4 PDF generation")

        if "_report_md" in st.session_state and st.session_state["_report_md"]:
            report_md  = st.session_state["_report_md"]
            report_pdf = st.session_state.get("_report_pdf", None)
            fir = st.session_state.case_context.fir_number.replace("/", "-")

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                st.download_button(
                    label="⬇️ Download Report (.md)",
                    data=report_md.encode("utf-8"),
                    file_name=f"EvidencePro_{fir}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with col_dl2:
                if report_pdf:
                    st.download_button(
                        label="📄 Download Report (.pdf)",
                        data=report_pdf,
                        file_name=f"EvidencePro_{fir}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "📄 PDF unavailable — install reportlab",
                        disabled=True,
                        use_container_width=True,
                    )

            st.divider()
            with st.expander("📋 Preview report (Markdown)", expanded=False):
                st.markdown(report_md)

# ===========================================================================
# TAB 8 — POLICY ADMIN
# ===========================================================================
with tab8:
    st.header("Step 8 — Policy Administration")
    st.caption(
        "Policy weights are a **CONFIGURABLE PROTOTYPE MECHANISM** only. "
        "They are NOT official forensic standards, legally validated thresholds, "
        "or scientifically peer-reviewed forensic triage guidelines."
    )

    role = _role_enum()

    # ---- POLICY HISTORY ----
    st.subheader("Policy History")
    if not POLICY_HISTORY:
        st.info("No policies have been proposed yet.")
    else:
        import pandas as pd
        hist_rows = [
            {
                "Version": p.version,
                "Label": p.label,
                "P": p.weight_P, "D": p.weight_D,
                "E": p.weight_E, "S": p.weight_S,
                "Status": p.status,
                "Proposed by": p.proposed_by,
                "Approved by": p.approved_by or "—",
            }
            for p in POLICY_HISTORY
        ]
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    active_pol = _active_policy()
    if active_pol:
        st.success(
            f"**Active policy:** {active_pol.label} · v{active_pol.version} · "
            f"P={active_pol.weight_P} · D={active_pol.weight_D} · "
            f"E={active_pol.weight_E} · S={active_pol.weight_S}"
        )
    else:
        st.info("No active policy — system is running in unweighted ML mode.")

    st.divider()

    # ---- PROPOSE POLICY (POLICY_ADMIN only) ----
    if role == PolicyRole.POLICY_ADMIN:
        st.subheader("Propose a New Policy")
        st.warning(
            "⚠️ Any policy you propose must be reviewed and approved by an Approver "
            "before it becomes active. Your proposal does NOT immediately change the system."
        )
        with st.form("propose_policy_form"):
            p_version     = st.text_input("Version", placeholder="e.g. 1.0.0")
            p_label       = st.text_input("Label", placeholder="e.g. Homicide Focus Policy")
            p_description = st.text_area("Description", height=80)
            c1, c2, c3, c4 = st.columns(4)
            p_wP = c1.number_input("Weight P", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            p_wD = c2.number_input("Weight D", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            p_wE = c3.number_input("Weight E", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            p_wS = c4.number_input("Weight S", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            p_proposer = st.text_input("Your name/ID", placeholder="e.g. admin1")
            if st.form_submit_button("Submit Policy Proposal"):
                if not p_version.strip() or not p_label.strip() or not p_proposer.strip():
                    st.error("Version, Label, and Your name/ID are required.")
                else:
                    try:
                        new_pol = TriagePolicy(
                            version=p_version.strip(),
                            label=p_label.strip(),
                            description=p_description.strip(),
                            weight_P=p_wP, weight_D=p_wD,
                            weight_E=p_wE, weight_S=p_wS,
                            proposed_by=p_proposer.strip(),
                        )
                        propose_policy(new_pol, role)
                        st.success(f"Policy proposed: **{p_label}** v{p_version}. Awaiting Approver review.")
                    except ValueError as e:
                        st.error(str(e))

    # ---- APPROVE / REJECT (APPROVER only) ----
    if role == PolicyRole.APPROVER:
        proposed = [p for p in POLICY_HISTORY if p.status == "proposed"]
        if proposed:
            st.subheader("Pending Policy Proposals")
            for pol in proposed:
                with st.expander(f"**{pol.label}** v{pol.version} — proposed by {pol.proposed_by}"):
                    st.write(f"Weights: P={pol.weight_P} · D={pol.weight_D} · E={pol.weight_E} · S={pol.weight_S}")
                    if pol.description:
                        st.caption(pol.description)
                    col_a, col_r = st.columns(2)
                    approver_name = st.text_input(
                        "Your name/ID",
                        key=f"approver_{pol.version}",
                        placeholder="e.g. approver1",
                    )
                    with col_a:
                        if st.button(f"✅ Approve v{pol.version}", key=f"appr_{pol.version}"):
                            if not approver_name.strip():
                                st.error("Enter your name/ID.")
                            else:
                                try:
                                    approve_policy(pol, approver_name.strip(), role)
                                    st.success(f"Policy approved and now active: {pol.label}")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))
                    with col_r:
                        if st.button(f"❌ Reject v{pol.version}", key=f"rej_{pol.version}"):
                            if not approver_name.strip():
                                st.error("Enter your name/ID.")
                            else:
                                try:
                                    reject_policy(pol, approver_name.strip(), role)
                                    st.warning(f"Policy rejected: {pol.label}")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))
        else:
            st.info("No pending policy proposals to review.")

# ===========================================================================
# TAB 9 — MODEL FEEDBACK + CANDIDATE RETRAINING
# ===========================================================================
with tab9:
    st.header("Step 9 — Model Feedback and Candidate Retraining")
    st.caption(
        "This tab provides a **prototype** feedback monitoring and candidate retraining workflow. "
        "It uses the synthetic/demo dataset and must not be used for real forensic decisions."
    )

    st.warning(
        "⚠️ **PROTOTYPE MECHANISM**\n\n"
        "This is a demonstration of a human-in-the-loop retraining workflow. "
        "Candidate models are trained on the same synthetic dataset used for the active model. "
        "Rejected or unapproved candidates NEVER replace the active model. "
        "Approval by a designated reviewer is required before any candidate becomes active."
    )

    items     = st.session_state.evidence_items
    results   = st.session_state.triage_results
    overrides = st.session_state.overrides
    tr        = st.session_state.training_result

    # ---- OVERRIDE MONITORING ----
    st.subheader("Override Monitoring")

    col_th1, col_th2 = st.columns([3, 1])
    with col_th1:
        st.caption("Configure the override rate threshold for retraining recommendations.")
    with col_th2:
        threshold_pct = st.number_input(
            "Threshold (%)",
            min_value=1, max_value=100,
            value=int(st.session_state.override_threshold * 100),
            step=5,
            key="threshold_input",
        )
        st.session_state.override_threshold = threshold_pct / 100.0

    mon = override_monitoring_report(
        overrides,
        len(items),
        st.session_state.override_threshold,
    )

    c_ov1, c_ov2, c_ov3, c_ov4 = st.columns(4)
    c_ov1.metric("Total Decisions", mon["total_decisions"])
    c_ov2.metric("Overrides", mon["override_count"])
    c_ov3.metric("Override Rate", f"{mon['override_pct']:.1f}%")
    c_ov4.metric("Threshold", f"{threshold_pct}%")

    if mon["retraining_recommended"]:
        st.error(f"🔄 {mon['message']}")
    else:
        st.info(mon["message"])

    st.caption(
        "Note: An override does NOT automatically mean the model was wrong. "
        "Investigators may apply case-specific knowledge not captured in the model's features."
    )

    # ---- FEEDBACK LOG ----
    if FEEDBACK_LOG:
        with st.expander(f"📋 Feedback log ({len(FEEDBACK_LOG)} entries)", expanded=False):
            import pandas as pd
            fb_rows = [
                {
                    "Evidence ID": fb.evidence_id,
                    "Model Rec.": fb.model_recommendation,
                    "Investigator": fb.investigator_decision,
                    "Reason": fb.override_reason[:50] if fb.override_reason else "—",
                    "Model": fb.model_name,
                    "Timestamp": fb.timestamp[:16],
                }
                for fb in FEEDBACK_LOG
            ]
            st.dataframe(pd.DataFrame(fb_rows), use_container_width=True, hide_index=True)

    # ---- CANDIDATE RETRAINING WORKFLOW ----
    st.divider()
    st.subheader("Candidate Retraining Workflow")

    if not tr:
        st.warning("Train a model using the sidebar first.")
    elif not items:
        st.info("Add evidence items and run triage before requesting candidate retraining.")
    else:
        col_req1, col_req2 = st.columns([2, 1])
        with col_req1:
            if st.button(
                "🔄 Request Candidate Retraining",
                help=(
                    "Creates a candidate model from the current override feedback. "
                    "The candidate must be evaluated and approved before it can become active."
                ),
                disabled=(len(overrides) == 0),
            ):
                with st.spinner("Creating candidate model…"):
                    candidate = request_candidate_retrain(
                        model_name=tr.model_name,
                        base_accuracy=tr.cv_accuracy,
                        overrides=overrides,
                        items=items,
                        results=results,
                        session_id=st.session_state.session_id,
                    )
                st.success(
                    f"✅ Candidate {candidate.candidate_id} created. "
                    f"Status: {candidate.status}. Click **Evaluate** to assess it."
                )
                st.rerun()

            if len(overrides) == 0:
                st.caption("Add at least one investigator override to request retraining.")

        with col_req2:
            if CANDIDATE_LOG:
                latest = CANDIDATE_LOG[-1]
                st.info(
                    f"Latest: **{latest.candidate_id}**\n\n"
                    f"Status: {latest.status}\n\n"
                    f"Feedback: {latest.feedback_count} items"
                )

    # ---- CANDIDATE MANAGEMENT ----
    if CANDIDATE_LOG:
        st.divider()
        st.subheader("Candidate Models")

        import pandas as pd
        cand_rows = [
            {
                "ID": c.candidate_id,
                "Model": c.base_model_name,
                "Status": c.status,
                "Feedback": c.feedback_count,
                "Base CV%": f"{c.base_accuracy*100:.1f}%",
                "Cand. CV%": f"{c.cv_accuracy*100:.1f}%" if c.cv_accuracy > 0 else "—",
                "Approved by": c.approved_by or "—",
                "Created": c.created_at[:10],
            }
            for c in CANDIDATE_LOG
        ]
        st.dataframe(pd.DataFrame(cand_rows), use_container_width=True, hide_index=True)

        for candidate in CANDIDATE_LOG:
            with st.expander(
                f"**{candidate.candidate_id}** — {candidate.status.upper()} — {candidate.base_model_name}",
                expanded=(candidate.status == "pending"),
            ):
                st.caption(candidate.notes[:300] if candidate.notes else "")

                if candidate.status == "pending":
                    if st.button(f"🔍 Evaluate {candidate.candidate_id}", key=f"eval_{candidate.candidate_id}"):
                        with st.spinner("Evaluating candidate model…"):
                            evaluate_candidate(candidate)
                        st.success(
                            f"✅ {candidate.candidate_id} evaluated: "
                            f"CV accuracy = {candidate.cv_accuracy*100:.1f}% "
                            f"(base = {candidate.base_accuracy*100:.1f}%)"
                        )
                        st.rerun()

                elif candidate.status == "evaluated":
                    col_cv, col_base = st.columns(2)
                    col_cv.metric(
                        "Candidate CV accuracy",
                        f"{candidate.cv_accuracy*100:.1f}%",
                        delta=f"{(candidate.cv_accuracy - candidate.base_accuracy)*100:+.1f}%",
                    )
                    col_base.metric("Current model CV accuracy", f"{candidate.base_accuracy*100:.1f}%")

                    st.info(
                        "Accuracy comparison is on the same synthetic dataset. "
                        "A higher candidate accuracy does not guarantee better real-world performance. "
                        "Approve only after reviewing the candidate's characteristics."
                    )

                    approver_name = st.text_input(
                        "Approver name/ID",
                        key=f"cand_approver_{candidate.candidate_id}",
                        placeholder="e.g. approver1",
                    )
                    col_appr, col_rej = st.columns(2)
                    with col_appr:
                        if st.button(f"✅ Approve {candidate.candidate_id}", key=f"appr_cand_{candidate.candidate_id}"):
                            if not approver_name.strip():
                                st.error("Enter approver name/ID.")
                            else:
                                try:
                                    approve_candidate(candidate, approver_name.strip())
                                    st.success(f"✅ {candidate.candidate_id} approved.")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))
                    with col_rej:
                        if st.button(f"❌ Reject {candidate.candidate_id}", key=f"rej_cand_{candidate.candidate_id}"):
                            reject_reason = st.session_state.get(f"rej_reason_{candidate.candidate_id}", "")
                            try:
                                reject_candidate(candidate, approver_name.strip() or "reviewer", "Rejected via UI")
                                st.warning(f"⚠️ {candidate.candidate_id} rejected.")
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))

                elif candidate.status == "approved":
                    st.success(
                        f"✅ Approved by {candidate.approved_by}. "
                        "Click **Activate** to make this model active."
                    )
                    active_cand = get_active_candidate()
                    if active_cand and active_cand.candidate_id == candidate.candidate_id:
                        st.success("🟢 This candidate is ACTIVE.")
                    else:
                        if st.button(f"🚀 Activate {candidate.candidate_id}", key=f"activate_{candidate.candidate_id}"):
                            activate_candidate(candidate)
                            if tr and candidate.training_result:
                                st.session_state.training_result = candidate.training_result
                                _reset_triage()
                            st.success(
                                f"✅ {candidate.candidate_id} activated. "
                                "Re-run triage to use the new model."
                            )
                            st.rerun()

                elif candidate.status == "rejected":
                    st.error(f"❌ Rejected by {candidate.approved_by}. This candidate will not be activated.")

        # ---- Active candidate info ----
        active_cand = get_active_candidate()
        if active_cand:
            st.divider()
            st.info(
                f"**Active candidate:** {active_cand.candidate_id} — "
                f"{active_cand.base_model_name} — "
                f"CV: {active_cand.cv_accuracy*100:.1f}%"
            )

    st.divider()
    st.caption(
        "**Prototype disclaimer:** This feedback and candidate retraining mechanism is a "
        "demonstration only. It uses the same synthetic dataset as the active model. "
        "A higher CV accuracy on this synthetic dataset does NOT indicate improved real-world "
        "forensic performance. The investigator always remains the final decision-maker."
    )
