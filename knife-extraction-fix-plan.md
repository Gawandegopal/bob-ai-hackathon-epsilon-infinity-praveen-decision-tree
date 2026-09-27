# Knife Extraction Fix — Plan

> **Status:** DRAFT — awaiting approval before any code changes.
>
> This plan is grounded in full inspection of src/extractor.py and src/app.py.

---

## Top-Level Overview

The reported bug: entering a knife description in Tab 2 produces
`evidence_type = biological_dna`. Root-cause investigation reveals **two
independent issues** that can each cause the symptom:

**Issue A — Stale extraction cache in app.py (primary cause)**
`st.session_state._extraction` persists across evidence items. If an
investigator extracts a blood swab (populating `biological_dna` in
`_extraction`), then types a knife description and submits the form **without
clicking "Extract with AI" again**, the form defaults are driven by the stale
`biological_dna` cache. The form appears to show the correct description but
the evidence_type selectbox is pre-selected to `biological_dna`.

**Issue B — No knife/blade/weapon keywords in extractor.py (secondary
/ correctness gap)**
The knife description contains no keyword that matches any entry in
`_EVIDENCE_KEYWORDS`. This means the heuristic correctly returns `{}` — but
it also means the extractor gives no signal at all for a knife, leaving the
form with default values (whichever EVIDENCE_TYPES[0] is) rather than a
useful suggestion. Additionally, adding "knife", "blade", and "edged weapon"
to the `firearm_ballistic` entry (which covers weapons) would be incorrect
because a knife is not a firearm and its forensic characteristics (no GSR,
no ballistics, no cartridge) are entirely different. The correct category
for a bladed weapon is a new entry — or, pragmatically, a "weapons_other"
type — but since `EVIDENCE_TYPES` is a fixed canonical list defined in
`models/evidence_item.py` and used by the ML model, adding a new type is out
of scope for this fix.

The minimal correct fix for Issue B is: add `"knife"`, `"blade"`,
`"edged weapon"`, and `"machete"` as **exclusion guard terms** — when these
terms are present and no other more-specific evidence keyword matches, the
extractor should not classify the description at all, returning no
`evidence_type` suggestion. This prevents both the stale-cache path and any
future case where a biological keyword (e.g. "blood") is added alongside a
knife description.

**Scope of this fix:**
- Fix the stale cache in `app.py` (Issue A)
- Add an exclusion guard in `_extract_via_heuristics()` in `extractor.py`
  for explicit physical-object terms that identify the evidence as a non-
  biological artefact despite potentially containing biological keywords
  (Issue B)
- Add a regression test for the knife description
- Run the full test suite and Phase 4 checks

**Not in scope:**
- Adding a new `evidence_type` category to the canonical list
- Redesigning the extraction system
- Changing watsonx.ai prompt behaviour (the guard only affects Mode 2)
- Any change to priority, urgency, scheduling, or reporting logic

---

## Sub-Task 1 — Fix stale extraction cache in app.py

**Intent**
Prevent the Tab 2 form from pre-filling with evidence_type (or any other
field) from a prior item's extraction when the investigator has typed a
new description but not yet clicked "Extract with AI".

**Expected Outcomes**
- When the text area content changes and the investigator submits the form
  without re-extracting, the form uses neutral defaults (EVIDENCE_TYPES[0],
  probative_value=2, etc.) — not values from the prior extraction
- The extraction cache is cleared whenever the text area content differs
  from `st.session_state._extract_desc`

**Todo List**
1. In `src/app.py`, at the point where `ex = st.session_state._extraction`
   is read (line 302), add a guard: if `desc_input` (the current text area
   value) does not match `st.session_state._extract_desc`, reset `ex = {}`
   before using it to drive form defaults.
   - This means: if the investigator has typed a new description but not
     clicked Extract, the form shows neutral defaults.
   - Do not clear `st.session_state._extraction` itself at this point —
     only shadow it with a local empty dict for this render. The session
     state value is cleared on successful item add (already done at line 494).

**Relevant Context**
- `src/app.py` lines 302, 306, 360, 378, 387, etc. — all form defaults read from `ex`
- `st.session_state._extract_desc` holds the description that was last extracted
- `desc_input` is the current value of the text area widget (line 306 reads
  `value=st.session_state._extract_desc` as the initial value, but the user
  may have changed it)
- Streamlit re-renders the whole tab on every interaction; `desc_input`
  reflects the current text area state on each render

**Status:** [ ] pending

---

## Sub-Task 2 — Add exclusion guard in extractor.py

**Intent**
Prevent the heuristic extractor from classifying a description as
`biological_dna` (or any biological type) when the description explicitly
identifies the primary evidence object as a physical artefact — specifically
bladed weapons, where biological trace terminology (stain, reddish) might
appear in the description.

**Expected Outcomes**
- `extract_features("A folding knife with a dark reddish-brown stain on the
  blade...", ctx)` returns a dict with no `evidence_type` key (or returns
  empty) rather than `biological_dna`
- `extract_features("blood swab from victim", ctx)` is unaffected — still
  returns `biological_dna`
- `extract_features("phone recovered at scene", ctx)` is unaffected — still
  returns `digital_device`
- All existing Phase 4 tests continue to pass

**Design**
Add a module-level set `_PHYSICAL_OBJECT_EXCLUSIONS` of lowercase terms
that, when matched in the description, suppress any `biological_*` type
result from the heuristic. The set includes:
  `"knife"`, `"blade"`, `"edged weapon"`, `"machete"`, `"sword"`,
  `"crowbar"`, `"hammer"`, `"screwdriver"`, `"wrench"`, `"bat"`,
  `"club"`, `"baton"`

Logic in `_extract_via_heuristics()`:
- After `evidence_type` is matched, check if it starts with `"biological_"`
- If yes, check if any `_PHYSICAL_OBJECT_EXCLUSIONS` term is present in
  the lowercased text
- If an exclusion term is found, suppress the `evidence_type` match
  (set `evidence_type = None`) — the physical object is the evidence, not
  the biological trace on it
- Downstream fields (perishability, specialist_required, etc.) are only
  set when `evidence_type` is not None, so they are automatically suppressed

This approach is targeted: it only fires when (a) a biological type was
matched AND (b) a physical-object exclusion term is present. It does not
affect fingerprint, digital, firearm, toxicology, or any other type. It does
not affect the watsonx.ai path (Mode 1). It does not affect the JSON
validation path.

**Relevant Context**
- `src/extractor.py` `_extract_via_heuristics()` lines 429-470
- `_EVIDENCE_KEYWORDS` list lines 311-347
- The exclusion guard must be placed AFTER the keyword match loop and
  BEFORE the `if evidence_type:` block that populates the result dict
- `_SPECIALIST_KEYWORDS` uses "blood" etc. as specialist triggers — the
  specialist_type suppression is handled automatically because
  `specialist_type` is set independently of `evidence_type`, but for a
  knife with a blood stain the specialist suggestion "DNA analyst" would
  also be misleading. Add the same physical-object exclusion check to the
  specialist_type matching too: suppress `"DNA analyst"` suggestion when a
  physical-object exclusion term is present.

**Status:** [ ] pending

---

## Sub-Task 3 — Add regression test and run full suite

**Intent**
Confirm the fix is correct and does not regress any existing behaviour.

**Expected Outcomes**
- A new test `test_knife_not_biological_dna` in
  `src/tests/test_extractor.py` (new file) passes
- All 63 existing tests continue to pass
- Phase 4 demo passes (20/20)
- Phase 3 smoke test passes

**Todo List**
1. Create `src/tests/test_extractor.py` with:
   a. `test_knife_not_biological_dna` — asserts that the knife description
      produces no `evidence_type` key (or returns empty dict excluding
      metadata keys)
   b. `test_blood_swab_still_biological_dna` — regression: blood swab
      description still correctly returns `biological_dna`
   c. `test_phone_still_digital_device` — regression: mobile phone
      description still correctly returns `digital_device`
   d. `test_empty_description_returns_empty` — already in phase4_demo but
      good to have as a pytest
   e. `test_source_metadata_always_present` — `_source` key always in result
2. Run `python -m pytest src/tests/ -v`
3. Run `python src/phase4_demo.py`
4. Run `python src/phase3_smoke.py`

**Relevant Context**
- `src/tests/` already contains `__init__.py`, `test_triage_models.py`,
  `test_scheduler.py`, `test_report.py`
- `extract_features()` public API in `src/extractor.py` line 131
- The test should call `extract_features(description, ctx)` with no API key
  so it uses heuristic mode

**Status:** [ ] pending

---

## Constraints (must be preserved throughout)

- `src/evidence_triage.py` and `src/run_triage.py` — FROZEN, never modify
- No ML logic, policy formulas, or scheduling rules in `app.py` or
  `extractor.py`
- `extractor.py` never assigns a priority tier
- All AI suggestions labelled as suggestions; investigator review unchanged
- `DEFAULT_POLICY = None` unchanged
- `DATA_DISCLAIMER` unchanged
- Existing Phase 1–3 smoke tests must still pass

---

## Files Changed

| File | Change |
|---|---|
| `src/app.py` | Sub-Task 1: stale-cache guard (1–2 lines) |
| `src/extractor.py` | Sub-Task 2: `_PHYSICAL_OBJECT_EXCLUSIONS` constant + guard in `_extract_via_heuristics()` |
| `src/tests/test_extractor.py` | Sub-Task 3: new test file with 5 tests |
