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


def generate_pdf_from_md(md_content: str) -> bytes | None:
    """Converts a Markdown string into an A4-formatted PDF byte stream.

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