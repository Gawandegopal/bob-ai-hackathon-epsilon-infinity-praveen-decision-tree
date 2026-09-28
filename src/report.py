"""
report.py
=========
EvidencePro — Markdown Report Generator
-----------------------------------------

Produces a complete, human-readable Markdown report for one triage session.

IMPORTANT NOTICES IN EVERY REPORT
-----------------------------------
1. AI/ML outputs are recommendations. The investigator is the final decision-maker.
2. The system was trained on SYNTHETIC / DEMONSTRATION DATA. Accuracy figures
   do not represent real-world forensic performance.
3. Any active policy is a CONFIGURABLE PROTOTYPE MECHANISM — not an official
   forensic standard.

REPORT SECTIONS
---------------
1.  Header + top disclaimer
2.  Case / FIR summary
3.  Evidence inventory
4.  Evidence classification (PDES values, priority tier, urgency)
5.  Priority explanations (per item)
6.  FSL examination schedule (three batches)
7.  Investigator review decisions
8.  Model and policy information
9.  Footer disclaimer (synthetic data warning)

The report is returned as a plain Markdown string. Streamlit renders it with
st.markdown(); it can also be downloaded as a .md file via st.download_button().
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.evidence_item import CaseContext, EvidenceItem
    from models.triage_models import TriageResult
    from models.policy import TriagePolicy
    from scheduler import ScheduledItem

from models.triage_models import DATA_DISCLAIMER
from scheduler import BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE, schedule_summary


# ---------------------------------------------------------------------------
# REPORT DATA CONTAINER
# ---------------------------------------------------------------------------

@dataclass
class ReportData:
    """
    All inputs required to generate a complete triage report.

    Attributes
    ----------
    context      : CaseContext for this session.
    items        : List of EvidenceItem objects.
    results      : List of raw ML TriageResult objects (one per item).
    schedule     : Sorted list of ScheduledItem objects from build_schedule().
    policy       : The active TriagePolicy at the time of the report, or None.
    generated_at : ISO 8601 timestamp string (caller's responsibility to set).
    model_used   : Human-readable model name, e.g. "Decision Tree".
    cv_accuracy  : Cross-validated accuracy from TrainingResult (demonstrative).
    """
    context:      "CaseContext"
    items:        list["EvidenceItem"]
    results:      list["TriageResult"]
    schedule:     list["ScheduledItem"]
    policy:       "TriagePolicy | None"
    generated_at: str
    model_used:   str
    cv_accuracy:  float


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_report(data: ReportData) -> str:
    """
    Generate a complete Markdown triage report.

    Parameters
    ----------
    data : ReportData containing all session information.

    Returns
    -------
    A Markdown string suitable for rendering or download.
    """
    sections = [
        _header_section(data),
        _case_summary_section(data.context),
        _evidence_inventory_section(data.items),
        _evidence_classification_section(data.items, data.results, data.schedule),
        _priority_explanations_section(data.items, data.results, data.schedule),
        _fsl_schedule_section(data.schedule),
        _review_decisions_section(data.schedule),
        _model_policy_section(data),
        _footer_section(),
    ]
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# SECTION BUILDERS
# ---------------------------------------------------------------------------

def _header_section(data: ReportData) -> str:
    return textwrap.dedent(f"""\
        # EvidencePro — Forensic Evidence Triage Report

        **Generated:** {data.generated_at}
        **Case / FIR:** {data.context.fir_number}
        **Offence type:** {data.context.offence_type}

        > **IMPORTANT DISCLAIMER**
        >
        > EvidencePro is an AI-assisted decision-support tool.
        > All recommendations in this report are produced by a machine-learning model
        > trained on synthetic demonstration data.
        >
        > **The investigator is the final decision-maker.**
        > No recommendation in this report constitutes a final forensic, legal,
        > or evidentiary decision.
        >
        > {DATA_DISCLAIMER}
    """).rstrip()


def _case_summary_section(context: "CaseContext") -> str:
    narrative = context.narrative.strip() if context.narrative else "_Not provided_"
    return textwrap.dedent(f"""\
        ## 1. Case / FIR Summary

        | Field | Value |
        |---|---|
        | FIR / Case number | {context.fir_number} |
        | Offence type | {context.offence_type} |

        **Case narrative:**

        {narrative}
    """).rstrip()


def _evidence_inventory_section(items: list["EvidenceItem"]) -> str:
    lines = [
        "## 2. Evidence Inventory",
        "",
        "| # | ID | Label | Evidence Type | Collection Age (h) | Condition |",
        "|---|---|---|---|---|---|",
    ]
    condition_label = {1: "Poor", 2: "Fair", 3: "Good"}
    for i, item in enumerate(items, 1):
        age = str(item.collection_age_hours) if item.collection_age_hours > 0 else "Unknown"
        cond = condition_label.get(item.evidence_condition, str(item.evidence_condition))
        lines.append(
            f"| {i} | {item.item_id} | {item.label} "
            f"| {item.evidence_type} | {age} | {cond} |"
        )
    return "\n".join(lines)


def _evidence_classification_section(
    items:    list["EvidenceItem"],
    results:  list["TriageResult"],
    schedule: list["ScheduledItem"],
) -> str:
    # Build a lookup from item_id to ScheduledItem for final_tier
    sched_lookup = {si.item.item_id: si for si in schedule}

    lines = [
        "## 3. Evidence Classification",
        "",
        "| # | Label | P | D | E | S (days) | Raw ML Tier | "
        "Adj. Tier | Urgency | Specialist |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    speed_label = lambda d: "Fast" if d <= 3 else ("Moderate" if d <= 10 else "Slow")

    for i, (item, result) in enumerate(zip(items, results), 1):
        si = sched_lookup.get(item.item_id)
        final_tier = si.final_tier if si else result.priority_tier
        policy_tier = (si.policy_result.priority_tier
                       if si and si.policy_result else "—")
        urgency = "YES" if result.urgency_flag else "No"
        specialist = "Yes" if item.specialist_required else "No"
        # Show policy-adjusted tier only if different from raw ML tier
        adj_display = policy_tier if policy_tier != result.priority_tier else "—"
        lines.append(
            f"| {i} | {item.label} "
            f"| {item.probative_value} | {item.perishability} "
            f"| {item.exclusionary_power} | {item.testing_lead_time} "
            f"({speed_label(item.testing_lead_time)}) "
            f"| **{result.priority_tier}** | {adj_display} "
            f"| {urgency} | {specialist} |"
        )

    lines.append("")
    lines.append(
        "_P = Probative Value · D = Degradation Risk · E = Exclusionary Power · "
        "S = Processing Speed (testing lead time in days). "
        "Adj. Tier = policy-adjusted tier (shown only when different from raw ML tier)._"
    )
    return "\n".join(lines)


def _priority_explanations_section(
    items:    list["EvidenceItem"],
    results:  list["TriageResult"],
    schedule: list["ScheduledItem"],
) -> str:
    sched_lookup = {si.item.item_id: si for si in schedule}
    lines = ["## 4. Priority Explanations"]

    for item, result in zip(items, results):
        si = sched_lookup.get(item.item_id)
        final_tier = si.final_tier if si else result.priority_tier

        lines.append(f"\n### {item.label}")
        lines.append(f"**AI recommendation:** {result.priority_tier}")
        if si and si.policy_result and si.policy_result.priority_tier != result.priority_tier:
            lines.append(f"**Policy-adjusted recommendation:** {si.policy_result.priority_tier}")
        lines.append(f"**Final tier (after investigator review):** {final_tier}")
        lines.append("")
        lines.append("```")
        lines.append(result.explanation)
        lines.append("```")

        # Decision path (Decision Tree only)
        if result.decision_path:
            lines.append("")
            lines.append("**Decision path:**")
            lines.append("```")
            lines.append(result.decision_path)
            lines.append("```")

        # Policy-adjusted explanation
        if si and si.policy_result and si.policy_result.explanation != result.explanation:
            lines.append("")
            lines.append("**Policy-adjusted explanation:**")
            lines.append("```")
            lines.append(si.policy_result.explanation)
            lines.append("```")

    return "\n".join(lines)


def _fsl_schedule_section(schedule: list["ScheduledItem"]) -> str:
    groups = schedule_summary(schedule)

    batch_meta = {
        BATCH_IMMEDIATE: (
            "Items requiring immediate laboratory examination. "
            "Critical priority or evidence with active urgency flag."
        ),
        BATCH_SECONDARY: (
            "Items for secondary scheduling. "
            "High priority (non-urgent) or Standard priority requiring specialist analysis."
        ),
        BATCH_ARCHIVE: (
            "Items for routine archive processing. "
            "Low priority or Standard priority with no specialist requirement."
        ),
    }

    lines = [
        "## 5. FSL Examination Schedule",
        "",
        "> Items within each batch are sorted by degradation risk (highest first), "
        "then by testing lead time (shortest first).",
        "",
        "> **Scheduling rules are deterministic and rule-based — not produced by an ML model.**",
    ]

    for batch_label in [BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE]:
        batch_items = groups[batch_label]
        lines.append(f"\n### {batch_label}")
        lines.append(f"_{batch_meta[batch_label]}_")
        lines.append(f"**Items in this batch: {len(batch_items)}**")
        lines.append("")

        if not batch_items:
            lines.append("_No items assigned to this batch._")
            continue

        lines.append("| # | Label | Final Tier | Urgency | Specialist | Lead (days) | Reason |")
        lines.append("|---|---|---|---|---|---|---|")

        for rank, si in enumerate(batch_items, 1):
            urgency = "YES" if si.triage_result.urgency_flag else "No"
            specialist = "Yes" if si.item.specialist_required else "No"
            reason_short = si.batch_reason[:80] + "…" if len(si.batch_reason) > 80 else si.batch_reason
            lines.append(
                f"| {rank} | {si.item.label} | **{si.final_tier}** "
                f"| {urgency} | {specialist} | {si.item.testing_lead_time} "
                f"| {reason_short} |"
            )

    return "\n".join(lines)


def _review_decisions_section(schedule: list["ScheduledItem"]) -> str:
    lines = [
        "## 6. Investigator Review Decisions",
        "",
        "| Label | AI Recommendation | Final Decision | Decision | Override Reason |",
        "|---|---|---|---|---|",
    ]

    overrides_present = any(si.investigator_decision == "overridden" for si in schedule)

    for si in schedule:
        ai_tier    = si.triage_result.priority_tier
        final_tier = si.final_tier
        decision   = si.investigator_decision or "—"
        reason     = si.override_reason or "—"
        changed    = " ⚠" if si.investigator_decision == "overridden" else ""
        lines.append(
            f"| {si.item.label} | {ai_tier} | {final_tier}{changed} "
            f"| {decision} | {reason} |"
        )

    if not overrides_present:
        lines.append("")
        lines.append("_No investigator overrides in this session._")

    lines.append("")
    lines.append(
        "_⚠ indicates the investigator's final decision differs from the AI recommendation._"
    )
    return "\n".join(lines)


def _model_policy_section(data: ReportData) -> str:
    from models.policy import policy_summary

    cv_pct = f"{data.cv_accuracy * 100:.1f}%"
    pol_text = policy_summary(data.policy)

    policy_detail = ""
    if data.policy is not None:
        p = data.policy
        policy_detail = textwrap.dedent(f"""
            | Policy field | Value |
            |---|---|
            | Label | {p.label} |
            | Version | {p.version} |
            | Description | {p.description or '—'} |
            | Weight P | {p.weight_P} |
            | Weight D | {p.weight_D} |
            | Weight E | {p.weight_E} |
            | Weight S | {p.weight_S} |
            | Proposed by | {p.proposed_by} |
            | Approved by | {p.approved_by or '—'} |

            > **DISCLAIMER:** These weights are a CONFIGURABLE PROTOTYPE MECHANISM.
            > They are NOT official forensic standards, legally validated thresholds,
            > or scientifically peer-reviewed forensic triage guidelines.
        """).strip()

    return textwrap.dedent(f"""\
        ## 7. Model and Policy Information

        | Field | Value |
        |---|---|
        | Model used | {data.model_used} |
        | Cross-validated accuracy | {cv_pct} (synthetic data — see disclaimer) |
        | Policy status | {pol_text} |
        | Report generated | {data.generated_at} |

        {DATA_DISCLAIMER}

        {policy_detail}
    """).rstrip()


def _footer_section() -> str:
    return textwrap.dedent(f"""\
        ## Important Notices

        **AI Recommendation Disclaimer**
        All priority tiers, urgency flags, and scheduling recommendations in this
        report were produced by an AI-assisted decision-support system. They are
        recommendations only. The investigator is the final decision-maker for
        all forensic triage and evidence submission decisions.

        **Synthetic Data Disclaimer**
        {DATA_DISCLAIMER}
        This system has been trained on synthetic/curated demonstration data only
        and has not been validated against real-world forensic casework. It must
        not be used for actual forensic decisions without appropriate validation
        and authorisation by a qualified forensic authority.

        **Policy Disclaimer**
        Any policy weights shown in this report are a CONFIGURABLE PROTOTYPE
        MECHANISM. They are NOT official forensic standards, legally validated
        thresholds, or scientifically peer-reviewed forensic triage guidelines.

        ---
        _Report generated by EvidencePro (prototype)_
    """).rstrip()


import io
import markdown

# Attempt to import xhtml2pdf safely
try:
    from xhtml2pdf import pisa
    XHTML2PDF_AVAILABLE = True
except ImportError:
    XHTML2PDF_AVAILABLE = False

# Attempt to import ReportLab for the professional A4 PDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# PDF GENERATOR — ReportLab (preferred, professional A4)
# ---------------------------------------------------------------------------

def generate_pdf_report(data: "ReportData") -> bytes | None:
    """
    Generate a professional A4 PDF report using ReportLab.

    The report reads like an investigator-facing narrative document, not a
    dump of Python data structures. Each section contains concise
    natural-language sentences that describe actual case data.

    Returns None if ReportLab is unavailable or generation fails.
    Markdown report always remains available as a fallback.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    try:
        return _build_reportlab_pdf(data)
    except Exception as e:
        print(f"ReportLab PDF generation failed: {e}")
        return None


def _build_reportlab_pdf(data: "ReportData") -> bytes:
    """Build the actual ReportLab PDF document."""
    from models.policy import policy_summary as _policy_summary
    from scheduler import schedule_summary as _schedule_summary, BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE

    buf = io.BytesIO()

    # --- Page setup ---
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=f"EvidencePro Report — {data.context.fir_number}",
        author="EvidencePro (prototype)",
    )

    # --- Styles ---
    styles = getSampleStyleSheet()

    S = {
        "title": ParagraphStyle(
            "title",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#0f2c59"),
            spaceAfter=4,
            spaceBefore=0,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#1a365d"),
            spaceAfter=4,
            spaceBefore=10,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=styles["Heading3"],
            fontSize=10,
            textColor=colors.HexColor("#334155"),
            spaceAfter=3,
            spaceBefore=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#57606a"),
            leading=11,
            spaceAfter=3,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#7c2d12"),
            backColor=colors.HexColor("#fef3c7"),
            leftIndent=6,
            rightIndent=6,
            leading=11,
            spaceAfter=6,
            spaceBefore=4,
        ),
        "code": ParagraphStyle(
            "code",
            parent=styles["Code"],
            fontSize=7.5,
            leading=10,
            fontName="Courier",
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=styles["Normal"],
            fontSize=7.5,
            textColor=colors.HexColor("#57606a"),
            alignment=TA_CENTER,
        ),
    }

    # --- Table style presets ---
    def _hdr_table_style(col_widths=None):
        return TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ])

    def _hr():
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb"), spaceAfter=4)

    # --- Build content ---
    story = []

    # =========================================================
    # TITLE + DISCLAIMER
    # =========================================================
    story.append(Paragraph("EvidencePro — Forensic Evidence Triage Report", S["title"]))
    story.append(_hr())
    story.append(Paragraph(
        f"<b>Generated:</b> {data.generated_at} &nbsp;&nbsp; "
        f"<b>Case / FIR:</b> {data.context.fir_number} &nbsp;&nbsp; "
        f"<b>Offence:</b> {data.context.offence_type}",
        S["body"],
    ))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph(
        "⚠ IMPORTANT DISCLAIMER — "
        "EvidencePro is an AI-assisted decision-support tool. "
        "All recommendations were produced by a machine-learning model trained on "
        "synthetic demonstration data. "
        "The investigator is the final decision-maker. "
        "No recommendation in this report constitutes a final forensic, legal, or evidentiary decision. "
        + DATA_DISCLAIMER,
        S["disclaimer"],
    ))

    # =========================================================
    # 1. CASE SUMMARY
    # =========================================================
    story.append(Paragraph("1. Case Summary", S["h2"]))
    n_items = len(data.items)
    story.append(Paragraph(
        f"{n_items} evidence item{'s' if n_items != 1 else ''} "
        f"{'were' if n_items != 1 else 'was'} reviewed for case <b>{data.context.fir_number}</b>. "
        f"The offence type recorded is <b>{data.context.offence_type}</b>.",
        S["body"],
    ))
    if data.context.narrative:
        story.append(Paragraph(f"<i>Case narrative: {data.context.narrative}</i>", S["caption"]))

    # =========================================================
    # 2. EVIDENCE INVENTORY
    # =========================================================
    story.append(Paragraph("2. Evidence Inventory", S["h2"]))
    story.append(Paragraph(
        f"The following {n_items} evidence item{'s were' if n_items != 1 else ' was'} "
        f"submitted for triage in this session.",
        S["body"],
    ))

    condition_label = {1: "Poor", 2: "Fair", 3: "Good"}
    inv_rows = [["#", "ID", "Description", "Evidence Type", "Age (h)", "Condition"]]
    for i, item in enumerate(data.items, 1):
        age_str = str(item.collection_age_hours) if item.collection_age_hours > 0 else "Unknown"
        inv_rows.append([
            str(i),
            item.item_id,
            item.label[:55] + ("…" if len(item.label) > 55 else ""),
            item.evidence_type.replace("_", " "),
            age_str,
            condition_label.get(item.evidence_condition, str(item.evidence_condition)),
        ])

    col_w = [8 * mm, 14 * mm, 55 * mm, 35 * mm, 14 * mm, 18 * mm]
    inv_t = Table(inv_rows, colWidths=col_w, repeatRows=1)
    inv_t.setStyle(_hdr_table_style())
    story.append(inv_t)

    # =========================================================
    # 3. EVIDENCE CLASSIFICATION
    # =========================================================
    story.append(Paragraph("3. Evidence Classification", S["h2"]))
    story.append(Paragraph(
        "Each evidence item was assessed across four PDES dimensions: "
        "Probative Value (P), Degradation Risk (D), Exclusionary Power (E), and "
        "Processing Speed — the inverse of testing lead time (S). "
        "The ML model combined these dimensions to assign a priority tier.",
        S["body"],
    ))

    sched_lookup = {si.item.item_id: si for si in data.schedule}
    speed_lbl = lambda d: "Fast" if d <= 3 else ("Moderate" if d <= 10 else "Slow")

    cls_rows = [["ID", "P", "D", "E", "S", "ML Tier", "Final Tier", "Urgent"]]
    for item, result in zip(data.items, data.results):
        si = sched_lookup.get(item.item_id)
        final_tier = si.final_tier if si else result.priority_tier
        cls_rows.append([
            item.item_id,
            str(item.probative_value),
            str(item.perishability),
            str(item.exclusionary_power),
            speed_lbl(item.testing_lead_time),
            result.priority_tier,
            final_tier,
            "YES" if result.urgency_flag else "No",
        ])

    cls_t = Table(cls_rows, colWidths=[14*mm, 9*mm, 9*mm, 9*mm, 18*mm, 22*mm, 22*mm, 16*mm], repeatRows=1)
    cls_t.setStyle(_hdr_table_style())
    story.append(cls_t)
    story.append(Paragraph(
        "P=1 Low / 2 Medium / 3 High. D=1 Stable / 2 Degrades in days / 3 Degrades in hours. "
        "E=1 Low / 2 Medium / 3 High exclusionary power.",
        S["caption"],
    ))

    # =========================================================
    # 4. PRIORITY RECOMMENDATIONS
    # =========================================================
    story.append(Paragraph("4. Priority Recommendations", S["h2"]))
    story.append(Paragraph(
        f"The <b>{data.model_used}</b> model was used to classify each evidence item. "
        "The following section describes why each item received its recommendation.",
        S["body"],
    ))

    for item, result in zip(data.items, data.results):
        si = sched_lookup.get(item.item_id)
        final_tier = si.final_tier if si else result.priority_tier

        # Natural-language sentence
        tier_words = {
            "Critical": "classified as Critical, indicating an urgent requirement for immediate examination",
            "High":     "classified as High priority, warranting secondary-batch examination",
            "Standard": "classified as Standard priority for routine processing",
            "Low":      "classified as Low priority for archive/routine processing",
        }
        reason_sentence = (
            f"Evidence <b>{item.item_id}</b> ({item.label}) was "
            + tier_words.get(result.priority_tier, f"classified as {result.priority_tier}")
            + ". "
        )
        if item.perishability == 3:
            reason_sentence += "Its high degradation risk (perishability = 3) means it degrades within hours. "
        elif item.perishability == 2:
            reason_sentence += "Its moderate degradation risk (perishability = 2) means it degrades over days. "
        if item.probative_value == 3:
            reason_sentence += "Its high probative value strongly supports the investigation. "
        if result.urgency_flag:
            reason_sentence += "An urgency flag was set due to perishability. "
        if si and si.investigator_decision == "overridden":
            reason_sentence += (
                f"The investigator overrode the recommendation, changing the priority "
                f"from {result.priority_tier} to <b>{final_tier}</b>."
            )

        story.append(KeepTogether([
            Paragraph(f"<b>{item.item_id}</b> — {item.label}", S["h3"]),
            Paragraph(reason_sentence, S["body"]),
        ]))

    # =========================================================
    # 5. URGENCY FLAGS
    # =========================================================
    urgent_items = [(item, r) for item, r in zip(data.items, data.results) if r.urgency_flag]
    if urgent_items:
        story.append(Paragraph("5. Urgency Flags", S["h2"]))
        story.append(Paragraph(
            f"{len(urgent_items)} evidence item{'s' if len(urgent_items) != 1 else ''} "
            f"{'carry' if len(urgent_items) != 1 else 'carries'} an urgency flag. "
            "These items require immediate attention regardless of priority tier.",
            S["body"],
        ))
        for item, result in urgent_items:
            story.append(Paragraph(
                f"• <b>{item.item_id}</b> ({item.label}): perishability = {item.perishability}. "
                + ("Degrades within hours — immediate processing required." if item.perishability == 3
                   else f"Collected {item.collection_age_hours} hour(s) ago — degradation possible."),
                S["body"],
            ))
    else:
        story.append(Paragraph("5. Urgency Flags", S["h2"]))
        story.append(Paragraph("No evidence items carry an urgency flag in this session.", S["body"]))

    # =========================================================
    # 6. INVESTIGATOR REVIEW / OVERRIDE DECISIONS
    # =========================================================
    story.append(Paragraph("6. Investigator Review and Override Decisions", S["h2"]))

    overridden = [si for si in data.schedule if si.investigator_decision == "overridden"]
    accepted   = [si for si in data.schedule if si.investigator_decision == "accepted"]

    story.append(Paragraph(
        f"The investigator reviewed all {len(data.schedule)} evidence recommendation(s). "
        f"{len(accepted)} recommendation{'s were' if len(accepted) != 1 else ' was'} accepted. "
        + (
            f"{len(overridden)} recommendation{'s were' if len(overridden) != 1 else ' was'} overridden."
            if overridden else "No recommendations were overridden."
        ),
        S["body"],
    ))

    if overridden:
        ov_rows = [["ID", "AI Recommendation", "Investigator Decision", "Override Reason"]]
        for si in overridden:
            ov_rows.append([
                si.item.item_id,
                si.triage_result.priority_tier,
                si.final_tier,
                (si.override_reason or "—")[:80],
            ])
        ov_t = Table(ov_rows, colWidths=[14*mm, 30*mm, 30*mm, 80*mm], repeatRows=1)
        ov_t.setStyle(_hdr_table_style())
        story.append(ov_t)

        for si in overridden:
            story.append(Paragraph(
                f"The investigator reviewed the recommendation for <b>{si.item.item_id}</b> "
                f"({si.item.label}) and changed the priority from "
                f"<b>{si.triage_result.priority_tier}</b> to <b>{si.final_tier}</b>"
                + (f" with the following reason: {si.override_reason}" if si.override_reason else "")
                + ".",
                S["body"],
            ))

    story.append(Paragraph(
        "Note: An investigator override does not automatically indicate that the model was incorrect. "
        "Investigators may apply case-specific knowledge not captured in the evidence features.",
        S["caption"],
    ))

    # =========================================================
    # 7. FSL EXAMINATION SCHEDULE
    # =========================================================
    story.append(PageBreak())
    story.append(Paragraph("7. FSL Examination Schedule", S["h2"]))
    story.append(Paragraph(
        "The FSL examination schedule assigns each evidence item to an examination batch "
        "using deterministic, rule-based logic — not an ML model. "
        "Critical evidence and urgent items are placed in the immediate examination batch. "
        "Less urgent evidence is scheduled according to priority, perishability, and specialist requirements.",
        S["body"],
    ))

    groups = _schedule_summary(data.schedule)

    batch_config = [
        (BATCH_IMMEDIATE, "Batch 1 — Immediate Examination",
         "Critical priority or evidence with an active urgency flag."),
        (BATCH_SECONDARY, "Batch 2 — Secondary Examination",
         "High priority (non-urgent) or Standard priority requiring specialist analysis."),
        (BATCH_ARCHIVE,   "Batch 3 — Archive / Routine Processing",
         "Low priority or Standard priority with no specialist requirement."),
    ]

    for batch_key, batch_title, batch_desc in batch_config:
        batch_items = groups[batch_key]
        story.append(Paragraph(batch_title, S["h3"]))
        story.append(Paragraph(batch_desc, S["caption"]))

        if not batch_items:
            story.append(Paragraph("No items assigned to this batch.", S["caption"]))
            continue

        sched_rows = [["#", "ID", "Description", "Final Tier", "Urgent", "Lead (d)", "Reason"]]
        for rank, si in enumerate(batch_items, 1):
            sched_rows.append([
                str(rank),
                si.item.item_id,
                si.item.label[:40] + ("…" if len(si.item.label) > 40 else ""),
                si.final_tier,
                "YES" if si.triage_result.urgency_flag else "No",
                str(si.item.testing_lead_time),
                si.batch_reason[:50] + ("…" if len(si.batch_reason) > 50 else ""),
            ])

        s_t = Table(sched_rows, colWidths=[8*mm, 14*mm, 40*mm, 20*mm, 14*mm, 14*mm, 50*mm], repeatRows=1)
        s_t.setStyle(_hdr_table_style())
        story.append(s_t)

    story.append(Paragraph(
        "Items within each batch are sorted by degradation risk (highest first) "
        "then by testing lead time (shortest first).",
        S["caption"],
    ))

    # =========================================================
    # 8. MODEL INFORMATION
    # =========================================================
    story.append(Paragraph("8. Model Information", S["h2"]))

    model_descriptions = {
        "Decision Tree": (
            "The Decision Tree model uses a sequence of interpretable evidence characteristics "
            "to reach a priority recommendation. Each decision is based on a clearly visible "
            "rule path — e.g. 'if perishability > 2 and probative value > 2, then Critical'. "
            "Decision Trees are fully transparent: the exact rules applied to each item can be shown."
        ),
        "Random Forest": (
            "The Random Forest model combines predictions from multiple decision trees to produce "
            "a more robust recommendation. It is less susceptible to overfitting than a single "
            "tree. Feature importances show which characteristics the model generally relies on "
            "most across all training data — these are global model properties, not per-item explanations."
        ),
        "Gradient Boosting": (
            "The Gradient Boosting model builds an ensemble sequentially, with each successive tree "
            "focusing on correcting the errors made by earlier trees. This typically produces higher "
            "accuracy on the training distribution but is less directly interpretable than a single tree. "
            "Feature importances are available at the model level."
        ),
    }

    story.append(Paragraph(f"<b>Model used:</b> {data.model_used}", S["body"]))
    desc = model_descriptions.get(data.model_used, "")
    if desc:
        story.append(Paragraph(desc, S["body"]))

    # =========================================================
    # 9. MODEL EVALUATION / ACCURACY TRANSPARENCY
    # =========================================================
    story.append(Paragraph("9. Model Evaluation and Accuracy Transparency", S["h2"]))
    cv_pct = data.cv_accuracy * 100
    story.append(Paragraph(
        f"The {data.model_used} model achieved a cross-validated accuracy of "
        f"<b>{cv_pct:.1f}%</b> on the synthetic demonstration dataset.",
        S["body"],
    ))
    story.append(Paragraph(
        "What does this mean? Model evaluation was performed using 5-fold cross-validation: "
        "the dataset was split into five equal parts, the model was trained on four parts and "
        "tested on the remaining part, five times in total. The reported accuracy is the average "
        "across these five test sets.",
        S["body"],
    ))
    story.append(Paragraph(
        "What dataset was used? The model was trained on a synthetic demonstration dataset of "
        "approximately 180 labelled forensic evidence scenarios. This data was constructed to "
        "illustrate plausible triage patterns and is NOT derived from real forensic casework.",
        S["body"],
    ))
    story.append(Paragraph(
        "Why does this limit real-world conclusions? Because the training data is synthetic, "
        f"the reported {cv_pct:.1f}% accuracy reflects prototype performance on simulated data only. "
        "It CANNOT be interpreted as validated real-world forensic triage accuracy. "
        "Any deployment would require validation against actual forensic casework under appropriate "
        "regulatory and scientific oversight.",
        S["body"],
    ))
    story.append(Paragraph(
        "⚠ " + DATA_DISCLAIMER,
        S["disclaimer"],
    ))

    # =========================================================
    # 10. POLICY INFORMATION
    # =========================================================
    story.append(Paragraph("10. Policy Information", S["h2"]))
    pol_text = _policy_summary(data.policy)
    story.append(Paragraph(f"<b>Policy status:</b> {pol_text}", S["body"]))
    if data.policy:
        p = data.policy
        story.append(Paragraph(
            f"The active policy ({p.label}, version {p.version}) applied PDES weighting to "
            f"the ML recommendations: P-weight={p.weight_P}, D-weight={p.weight_D}, "
            f"E-weight={p.weight_E}, S-weight={p.weight_S}. "
            f"This policy was proposed by {p.proposed_by} and "
            f"approved by {p.approved_by or 'unknown'}.",
            S["body"],
        ))
        story.append(Paragraph(
            "⚠ Policy weights are a CONFIGURABLE PROTOTYPE MECHANISM. "
            "They are NOT official forensic standards, legally validated thresholds, "
            "or scientifically peer-reviewed forensic triage guidelines.",
            S["disclaimer"],
        ))
    else:
        story.append(Paragraph(
            "No active policy. The ML model's raw recommendations were used without PDES weighting.",
            S["body"],
        ))

    # =========================================================
    # 11. DISCLAIMERS
    # =========================================================
    story.append(PageBreak())
    story.append(Paragraph("11. Disclaimers", S["h2"]))
    story.append(Paragraph(
        "<b>AI Recommendation Disclaimer:</b> All priority tiers, urgency flags, and scheduling "
        "recommendations in this report were produced by an AI-assisted decision-support system. "
        "They are recommendations only. The investigator is the final decision-maker for all forensic "
        "triage and evidence submission decisions.",
        S["body"],
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"<b>Synthetic Data Disclaimer:</b> {DATA_DISCLAIMER} "
        "This system has been trained on synthetic/curated demonstration data only and has not been "
        "validated against real-world forensic casework. It must not be used for actual forensic "
        "decisions without appropriate validation and authorisation by a qualified forensic authority.",
        S["body"],
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "<b>Policy Disclaimer:</b> Any policy weights shown in this report are a CONFIGURABLE "
        "PROTOTYPE MECHANISM. They are NOT official forensic standards, legally validated thresholds, "
        "or scientifically peer-reviewed forensic triage guidelines.",
        S["body"],
    ))

    # =========================================================
    # FOOTER (page numbers added via onPage callback)
    # =========================================================
    def _add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#57606a"))
        canvas.drawCentredString(
            A4[0] / 2,
            10 * mm,
            f"EvidencePro (prototype) — {data.context.fir_number} — Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return buf.getvalue()


def generate_pdf_from_md(md_content: str) -> bytes | None:
    """
    Legacy: Converts a Markdown string into an A4-formatted PDF via xhtml2pdf.

    Kept as a fallback for compatibility. The preferred PDF generator is
    generate_pdf_report() which uses ReportLab directly.

    Returns None if xhtml2pdf is not installed or if generation fails.
    """
    if not XHTML2PDF_AVAILABLE:
        return None

    try:
        # Convert Markdown string to HTML (enabling tables)
        html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

        # Custom A4 layout styling for forensic reports
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: A4 portrait;
                    margin: 18mm 15mm 18mm 15mm;
                    @bottom-right {{
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 8pt;
                        font-family: Helvetica, sans-serif;
                        color: #666;
                    }}
                }}
                body {{
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 9.5pt;
                    line-height: 1.4;
                    color: #222222;
                }}
                h1 {{
                    font-size: 16pt;
                    color: #0f2c59;
                    border-bottom: 2px solid #0f2c59;
                    padding-bottom: 4px;
                    margin-top: 0;
                }}
                h2 {{
                    font-size: 12pt;
                    color: #1a365d;
                    border-bottom: 1px solid #cbd5e1;
                    padding-bottom: 2px;
                    margin-top: 14px;
                }}
                h3 {{
                    font-size: 10pt;
                    color: #334155;
                    margin-top: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                th, td {{
                    border: 1px solid #cbd5e1;
                    padding: 5px 7px;
                    text-align: left;
                    font-size: 8.5pt;
                }}
                th {{
                    background-color: #f1f5f9;
                    font-weight: bold;
                    color: #0f172a;
                }}
                code, pre {{
                    font-family: Courier, monospace;
                    background-color: #f8fafc;
                    font-size: 8pt;
                }}
                pre {{
                    padding: 6px;
                    border: 1px solid #e2e8f0;
                }}
                blockquote {{
                    background-color: #fef3c7;
                    border-left: 3px solid #d97706;
                    margin: 10px 0;
                    padding: 6px 10px;
                    font-size: 8.5pt;
                }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """

        pdf_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(io.StringIO(styled_html), dest=pdf_buffer)

        if pisa_status.err:
            return None

        return pdf_buffer.getvalue()

    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None