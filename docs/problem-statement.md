# EvidencePro — Problem Statement

## Background

Forensic evidence collected at a crime scene must be submitted to a Forensic
Science Laboratory (FSL) for scientific analysis before it can be presented
in court. Laboratory capacity is limited: not all items can be processed
simultaneously, and the order in which items are submitted directly affects
the speed and quality of the investigation.

In most police forces, evidence prioritisation is an informal process.
Investigators rely on personal experience and institutional knowledge to
decide which items to submit first — a decision that is made under time
pressure, often in the immediate aftermath of a crime, and without structured
AI support.

## The Problem

**There is no structured, AI-assisted decision-support tool for forensic
evidence triage.** Investigators deciding which evidence to prioritise face
three compounding problems:

1. **Perishable evidence degrades silently.** Biological samples (blood, DNA
   swabs), toxicological samples, and gunshot residue all have narrow windows
   for effective collection and analysis. An investigator who does not
   recognise the urgency of a biological sample collected two hours ago may
   submit it to Batch 3 — by which time meaningful degradation may have
   occurred. Current workflows provide no automatic urgency signal based on
   collection time and evidence type.

2. **Priority decisions are inconsistent across investigators.** Without a
   structured framework, two investigators handling the same evidence items
   may assign different priorities. There is no documented rationale, no
   audit trail, and no way to calibrate decisions against a consistent set
   of forensic dimensions.

3. **Laboratory scheduling is manual and opaque.** After triage, building
   the FSL examination schedule — deciding which items go to which batch —
   is done manually. There is no tool that produces a deterministic,
   documented schedule that an investigator can review, override, and
   submit with confidence.

## Who is Affected

**Primary users:** Police investigators and scene-of-crime officers (SOCOs)
responsible for evidence collection, packaging, and FSL submission decisions.
These users typically have forensic training but not data science backgrounds.
They need a system that surfaces AI recommendations in plain language, allows
them to review and correct every decision, and produces a downloadable report
for case documentation.

**Secondary users:** FSL laboratory coordinators who receive evidence batches
and plan examination schedules. A structured, priority-labelled submission
improves their workflow planning.

## Why It Matters

Delayed or misordered evidence processing has direct consequences:

- **Evidence loss:** Perishable biological or toxicological evidence not
  processed within its degradation window may become unusable, weakening
  or destroying key prosecution evidence.
- **Prosecution failure:** The absence of timely forensic analysis has
  contributed to acquittals in cases where evidence quality was compromised
  by delayed submission.
- **Investigator burden:** The time an investigator spends manually assessing
  and documenting evidence priority is time not spent on other investigative
  tasks.

## Why Existing Solutions Fall Short

Current approaches rely on:

1. **Verbal briefings and informal checklists.** These are not documented,
   not reproducible, and not available to investigators who lack direct
   supervision from a senior colleague.

2. **Generic Laboratory Management Systems (LIMS).** Existing LMS tools
   manage laboratory workflows once evidence arrives but do not assist the
   investigator in the field with the upstream prioritisation decision.

3. **Training and experience alone.** Junior investigators or those handling
   unusual evidence types may not have the pattern recognition to identify
   which items are most time-critical.

None of these approaches provide a structured, AI-assisted, investigator-
reviewed, and documented triage workflow for crime-scene evidence.
