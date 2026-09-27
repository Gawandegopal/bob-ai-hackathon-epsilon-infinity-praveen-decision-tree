# EvidencePro — Forensic Evidence Triage Report

**Generated:** 2026-09-26 21:18 UTC
**Case / FIR:** FIR-2024-001
**Offence type:** homicide

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
> Accuracy figures are from a synthetic/demonstration dataset and do not represent real-world forensic performance.

---

## 1. Case / FIR Summary

| Field | Value |
|---|---|
| FIR / Case number | FIR-2024-001 |
| Offence type | homicide |

**Case narrative:**

Victim found at residential address. Multiple items collected from scene.

---

## 2. Evidence Inventory

| # | ID | Label | Evidence Type | Collection Age (h) | Condition |
|---|---|---|---|---|---|
| 1 | E001 | Blood swab — victim's clothing | biological_dna | 3 | Fair |
| 2 | E002 | Latent fingerprint — door frame | fingerprint_latent | Unknown | Fair |
| 3 | E003 | CCTV footage — street camera | digital_cctv | Unknown | Fair |
| 4 | E004 | Soil trace — suspect's boot | trace_soil | Unknown | Fair |
| 5 | E005 | Touch DNA — window handle | biological_touch_dna | 2 | Fair |

---

## 3. Evidence Classification

| # | Label | P | D | E | S (days) | Raw ML Tier | Adj. Tier | Urgency | Specialist |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Blood swab — victim's clothing | 3 | 3 | 3 | 5 (Moderate) | **Critical** | — | YES | Yes |
| 2 | Latent fingerprint — door frame | 2 | 2 | 2 | 5 (Moderate) | **High** | — | No | No |
| 3 | CCTV footage — street camera | 3 | 1 | 2 | 2 (Fast) | **Critical** | High | No | No |
| 4 | Soil trace — suspect's boot | 1 | 1 | 1 | 10 (Moderate) | **Standard** | Low | No | Yes |
| 5 | Touch DNA — window handle | 2 | 3 | 3 | 7 (Moderate) | **Critical** | — | YES | Yes |

_P = Probative Value · D = Degradation Risk · E = Exclusionary Power · S = Processing Speed (testing lead time in days). Adj. Tier = policy-adjusted tier (shown only when different from raw ML tier)._

---

## 4. Priority Explanations

### Blood swab — victim's clothing
**AI recommendation:** Critical
**Final tier (after investigator review):** Critical

```
AI RECOMMENDATION (EvidencePro — Decision Tree): Priority Critical.
URGENT: This evidence degrades within hours. Immediate processing is recommended.
Probative value (P): high | Degradation risk (D): high | Exclusionary power (E): high | Processing speed (S): moderate (est. 5 day(s))
How the Decision Tree reached this recommendation (actual rules applied to this specific item):
Decision path:
  Step 1: [probative_value = 3]  →  3 > 1.5
  Step 2: [probative_value = 3]  →  3 > 2.5
  Step 3: [perishability = 3]  →  3 > 2.5
  Step 4: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 5: LEAF → predicted priority = Critical
Note: This is an AI-assisted recommendation. The investigator is the final decision-maker.
```

**Decision path:**
```
Decision path:
  Step 1: [probative_value = 3]  →  3 > 1.5
  Step 2: [probative_value = 3]  →  3 > 2.5
  Step 3: [perishability = 3]  →  3 > 2.5
  Step 4: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 5: LEAF → predicted priority = Critical
```

### Latent fingerprint — door frame
**AI recommendation:** High
**Final tier (after investigator review):** High

```
AI RECOMMENDATION (EvidencePro — Decision Tree): Priority High.
Probative value (P): medium | Degradation risk (D): medium | Exclusionary power (E): medium | Processing speed (S): moderate (est. 5 day(s))
How the Decision Tree reached this recommendation (actual rules applied to this specific item):
Decision path:
  Step 1: [probative_value = 2]  →  2 > 1.5
  Step 2: [probative_value = 2]  →  2 ≤ 2.5
  Step 3: [perishability = 2]  →  2 > 1.5
  Step 4: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 5: [evidence_type = 'fingerprint_latent']  →  code 5 > threshold 3 ('digital_device')
  Step 6: [exclusionary_power = 2]  →  2 > 1.5
  Step 7: LEAF → predicted priority = High
Note: This is an AI-assisted recommendation. The investigator is the final decision-maker.
```

**Decision path:**
```
Decision path:
  Step 1: [probative_value = 2]  →  2 > 1.5
  Step 2: [probative_value = 2]  →  2 ≤ 2.5
  Step 3: [perishability = 2]  →  2 > 1.5
  Step 4: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 5: [evidence_type = 'fingerprint_latent']  →  code 5 > threshold 3 ('digital_device')
  Step 6: [exclusionary_power = 2]  →  2 > 1.5
  Step 7: LEAF → predicted priority = High
```

### CCTV footage — street camera
**AI recommendation:** Critical
**Policy-adjusted recommendation:** High
**Final tier (after investigator review):** High

```
AI RECOMMENDATION (EvidencePro — Decision Tree): Priority Critical.
Probative value (P): high | Degradation risk (D): low | Exclusionary power (E): medium | Processing speed (S): fast (est. 2 day(s))
How the Decision Tree reached this recommendation (actual rules applied to this specific item):
Decision path:
  Step 1: [probative_value = 3]  →  3 > 1.5
  Step 2: [probative_value = 3]  →  3 > 2.5
  Step 3: [perishability = 1]  →  1 ≤ 2.5
  Step 4: [offence_type = 'homicide']  →  code 4 ≤ threshold 6 ('sexual_assault')
  Step 5: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 6: [offence_type = 'homicide']  →  code 4 ≤ threshold 4 ('homicide')
  Step 7: LEAF → predicted priority = Critical
Note: This is an AI-assisted recommendation. The investigator is the final decision-maker.
```

**Decision path:**
```
Decision path:
  Step 1: [probative_value = 3]  →  3 > 1.5
  Step 2: [probative_value = 3]  →  3 > 2.5
  Step 3: [perishability = 1]  →  1 ≤ 2.5
  Step 4: [offence_type = 'homicide']  →  code 4 ≤ threshold 6 ('sexual_assault')
  Step 5: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 6: [offence_type = 'homicide']  →  code 4 ≤ threshold 4 ('homicide')
  Step 7: LEAF → predicted priority = Critical
```

### Soil trace — suspect's boot
**AI recommendation:** Standard
**Policy-adjusted recommendation:** Low
**Final tier (after investigator review):** Low

```
AI RECOMMENDATION (EvidencePro — Decision Tree): Priority Standard.
Probative value (P): low | Degradation risk (D): low | Exclusionary power (E): low | Processing speed (S): moderate (est. 10 day(s))
How the Decision Tree reached this recommendation (actual rules applied to this specific item):
Decision path:
  Step 1: [probative_value = 1]  →  1 ≤ 1.5
  Step 2: [perishability = 1]  →  1 ≤ 1.5
  Step 3: [evidence_type = 'trace_soil']  →  code 15 > threshold 14 ('trace_glass')
  Step 4: [offence_type = 'homicide']  →  code 4 > threshold 0 ('arson')
  Step 5: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 6: [offence_type = 'homicide']  →  code 4 ≤ threshold 4 ('homicide')
  Step 7: LEAF → predicted priority = Standard
Note: This is an AI-assisted recommendation. The investigator is the final decision-maker.
```

**Decision path:**
```
Decision path:
  Step 1: [probative_value = 1]  →  1 ≤ 1.5
  Step 2: [perishability = 1]  →  1 ≤ 1.5
  Step 3: [evidence_type = 'trace_soil']  →  code 15 > threshold 14 ('trace_glass')
  Step 4: [offence_type = 'homicide']  →  code 4 > threshold 0 ('arson')
  Step 5: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 6: [offence_type = 'homicide']  →  code 4 ≤ threshold 4 ('homicide')
  Step 7: LEAF → predicted priority = Standard
```

### Touch DNA — window handle
**AI recommendation:** Critical
**Final tier (after investigator review):** Critical

```
AI RECOMMENDATION (EvidencePro — Decision Tree): Priority Critical.
URGENT: This evidence degrades within hours. Immediate processing is recommended.
Probative value (P): medium | Degradation risk (D): high | Exclusionary power (E): high | Processing speed (S): moderate (est. 7 day(s))
How the Decision Tree reached this recommendation (actual rules applied to this specific item):
Decision path:
  Step 1: [probative_value = 2]  →  2 > 1.5
  Step 2: [probative_value = 2]  →  2 ≤ 2.5
  Step 3: [perishability = 3]  →  3 > 1.5
  Step 4: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 5: [evidence_type = 'biological_touch_dna']  →  code 1 ≤ threshold 3 ('digital_device')
  Step 6: [exclusionary_power = 3]  →  3 > 2.5
  Step 7: LEAF → predicted priority = Critical
Note: This is an AI-assisted recommendation. The investigator is the final decision-maker.
```

**Decision path:**
```
Decision path:
  Step 1: [probative_value = 2]  →  2 > 1.5
  Step 2: [probative_value = 2]  →  2 ≤ 2.5
  Step 3: [perishability = 3]  →  3 > 1.5
  Step 4: [offence_type = 'homicide']  →  code 4 > threshold 3 ('fraud')
  Step 5: [evidence_type = 'biological_touch_dna']  →  code 1 ≤ threshold 3 ('digital_device')
  Step 6: [exclusionary_power = 3]  →  3 > 2.5
  Step 7: LEAF → predicted priority = Critical
```

---

## 5. FSL Examination Schedule

> Items within each batch are sorted by degradation risk (highest first), then by testing lead time (shortest first).

> **Scheduling rules are deterministic and rule-based — not produced by an ML model.**

### Batch 1: Immediate
_Items requiring immediate laboratory examination. Critical priority or evidence with active urgency flag._
**Items in this batch: 2**

| # | Label | Final Tier | Urgency | Specialist | Lead (days) | Reason |
|---|---|---|---|---|---|---|
| 1 | Blood swab — victim's clothing | **Critical** | YES | Yes | 5 | Priority tier is Critical — immediate FSL examination required. |
| 2 | Touch DNA — window handle | **Critical** | YES | Yes | 7 | Priority tier is Critical — immediate FSL examination required. |

### Batch 2: Secondary
_Items for secondary scheduling. High priority (non-urgent) or Standard priority requiring specialist analysis._
**Items in this batch: 2**

| # | Label | Final Tier | Urgency | Specialist | Lead (days) | Reason |
|---|---|---|---|---|---|---|
| 1 | Latent fingerprint — door frame | **High** | No | No | 5 | Priority tier is High with no immediate urgency — secondary batch. |
| 2 | CCTV footage — street camera | **High** | No | No | 2 | Priority tier is High with no immediate urgency — secondary batch. |

### Batch 3: Archive
_Items for routine archive processing. Low priority or Standard priority with no specialist requirement._
**Items in this batch: 1**

| # | Label | Final Tier | Urgency | Specialist | Lead (days) | Reason |
|---|---|---|---|---|---|---|
| 1 | Soil trace — suspect's boot | **Low** | No | Yes | 10 | Priority tier is Low — archive batch for routine processing. |

---

## 6. Investigator Review Decisions

| Label | AI Recommendation | Final Decision | Decision | Override Reason |
|---|---|---|---|---|
| Blood swab — victim's clothing | Critical | Critical | accepted | — |
| Touch DNA — window handle | Critical | Critical | accepted | — |
| Latent fingerprint — door frame | High | High | accepted | — |
| CCTV footage — street camera | Critical | High | accepted | — |
| Soil trace — suspect's boot | Standard | Low | accepted | — |

_No investigator overrides in this session._

_⚠ indicates the investigator's final decision differs from the AI recommendation._

---

        ## 7. Model and Policy Information

        | Field | Value |
        |---|---|
        | Model used | Decision Tree |
        | Cross-validated accuracy | 79.7% (synthetic data — see disclaimer) |
        | Policy status | Active policy: Homicide Focus v1.0.0 (P=2.0, D=2.0, E=1.0, S=1.0) |
        | Report generated | 2026-09-26 21:18 UTC |

        Accuracy figures are from a synthetic/demonstration dataset and do not represent real-world forensic performance.

        | Policy field | Value |
|---|---|
| Label | Homicide Focus |
| Version | 1.0.0 |
| Description | — |
| Weight P | 2.0 |
| Weight D | 2.0 |
| Weight E | 1.0 |
| Weight S | 1.0 |
| Proposed by | admin1 |
| Approved by | approver1 |

> **DISCLAIMER:** These weights are a CONFIGURABLE PROTOTYPE MECHANISM.
> They are NOT official forensic standards, legally validated thresholds,
> or scientifically peer-reviewed forensic triage guidelines.

---

## Important Notices

**AI Recommendation Disclaimer**
All priority tiers, urgency flags, and scheduling recommendations in this
report were produced by an AI-assisted decision-support system. They are
recommendations only. The investigator is the final decision-maker for
all forensic triage and evidence submission decisions.

**Synthetic Data Disclaimer**
Accuracy figures are from a synthetic/demonstration dataset and do not represent real-world forensic performance.
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