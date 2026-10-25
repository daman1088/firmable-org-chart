# Title normalisation (Phase 2)

`orgchart.normalization` turns each observed `(title, department, company)` into
a structured, explainable record:

* seniority level (on the shared `orgchart.ladder` scale, 0 intern … 12 CEO);
* track (`executive`, `manager`, `ic`, `support`, `board` or `unknown`);
* whether the person manages people;
* department, specialty and scope (region or unit);
* flags: contractor, intern, interim, founder, non-employee;
* a confidence for each field, an overall `parse_confidence`, reason codes,
  the rules that fired, and a plain-English explanation.

The output is written to `<dataset>/derived/normalized_titles.parquet`. Every
column is listed in [data_dictionary.md](data_dictionary.md).

```bash
orgchart normalize data/full                 # parse all profiles, then evaluate
orgchart evaluate-normalization data/full    # re-score only -> reports/normalization_eval.{json,md}
orgchart parse-title "Snr Software Engr II" --department Engineering --industry "Software Development"
```

## Isolation

Normalisation reads only `orgchart.storage.observed`. Its lexicon
(`normalization/lexicon.py`) is separate from the generator's catalog and uses
its own department keys: `hardware_engineering` and `retail_operations` where
the generator uses `mfg_engineering` and `store_operations`. The evaluator
maps between the two.

`tests/test_isolation.py` scans the package and fails the build if it imports
the generator, truth storage or evaluation code, or if it names truth columns.

## Method

All rules are deterministic and live in four small modules.

1. **`clean.py`: cleaning and tokenising**
   * Strips hiring decorations, `@ Company`, `at <company>`, and contractor,
     interim/acting and founder markers. Each of these sets a flag.
   * Splits the title on separators (`-`, `|`, `/`, `::`, commas) and
     parentheses into a main phrase plus qualifiers.
   * Expands abbreviations: `Sr`/`Snr`, `Mgr`, `VP`, `SWE`, `RN`, `HRBP`, and
     others. Some expansions depend on context: `eng` and `dev` depend on their
     position in the title, and `PM` and `CNC` depend on the industry.
   * Repairs adjacent-letter typos. A repair is made only when exactly one
     lexicon word matches, and it lowers confidence (`TYPO_CORRECTED`).

2. **`parser.py`: the frame**
   Rules are tried in priority order and the first match wins:
   * board and advisor titles;
   * chief of staff and executive assistants, which are placed *before* CEO
     matching so that "EA to the CEO" is not read as the CEO;
   * CEO, MD and president;
   * C-suite acronyms and "Chief X Officer";
   * vague titles;
   * the management vocabulary (VP family, partner family, director family,
     head of, GM, controller, store/plant/area/site manager, manager,
     lead/supervisor);
   * the longest matching individual-contributor role.

   Some IC roles contain "Manager" but carry no reports: Product, Account,
   Customer Success, Program and Project Manager. These are recognised by the
   longest-role match, so the word "Manager" alone does not grant management.

   The prefixes **Staff**, **Principal** and **Lead** on an IC role set level 4
   and are explained as *technical seniority, not a people-management signal*.
   The bare consulting grade "Principal" (level 8) is the one exception.

   Other level modifiers:
   * graduate, junior, trainee: level 1;
   * intern, student, apprentice: level 0;
   * senior: +1;
   * grade suffixes (I–IV): set the level directly.

3. **Industry meanings** (`lexicon.INDUSTRY_LABELS`)
   * **Consulting** uses a grade ladder: Analyst 1, Consultant 2, Senior
     Consultant 3, Manager 5, Senior Manager 6, Principal 8, Partner 9.
     Practice qualifiers ("Manager, Strategy") map to `consulting`.
   * **Healthcare**: nursing ladder (RN → CNS → Nurse Unit Manager → Director
     of Nursing) and allied-health roles.
   * **Retail**: store hierarchy (Sales Assistant → Assistant Store Manager →
     Store Manager → Area Manager).
   * **Manufacturing**: production, quality and site roles.
   * **Fintech**: `CRO`, `CPO`, `CCO` and `CMO` are ambiguous. The department
     field resolves them; without a field the result is flagged
     `AMBIGUOUS_ACRONYM` and given department confidence 0.5.

4. **Choosing the department**
   * The parser reads function keywords from the title (main phrase weighted
     above qualifiers) and the employee's department field.
   * When the title and field agree, the result has the highest confidence.
   * When they conflict, the title wins (`DEPT_CONFLICT`) unless the title's
     evidence is weak (a generic role noun, or a vague or unknown frame). In
     that case the field wins at confidence 0.6.
   * When there is no evidence at all, the consulting industry default applies
     if relevant; otherwise the department is `DEPT_UNRESOLVED`.
   * On train, when the title won, it was right 93.4% of the time. When the
     field won over weak title evidence it was right 64.6% of the time, which
     matches its 0.6 confidence.

5. **`company_context.py`: company pass**
   A qualifier the lexicon cannot explain becomes a `unit` scope if at least
   three people in the same company share it (for example "Payments" or "Aged
   Care").

`parse_confidence` is the minimum of the level, department and management
confidences. It is 0 when any of the three is unresolved.

## Methodology

* Rules were developed by examining errors on **train** companies only.
  Calibration and test were scored, but their errors were not inspected.
* Test companies also carry title styles that never occur in train
  (`holdout:*`): `Mgr`/`Snr`/`SWE` abbreviations, reordered qualifiers, `/`
  and `::` separators, and typos.
* Some noise makes a title unrecoverable by any title parser: stale titles
  (after a promotion), inflated titles, vague titles and empty titles. Results
  are reported on all rows and separately on title-recoverable rows.
* The *title-only ceiling* is an oracle that assigns each distinct cleaned
  title its most common true level. It bounds any method that sees only the
  title.

## Results (600 companies, 207,300 profiles, about 26 s)

| metric | train | calibration | test |
|---|---|---|---|
| seniority coverage % | 99.52 | 99.54 | 99.45 |
| seniority exact % | 91.08 | 91.19 | 91.04 |
| seniority within ±1 % | 98.72 | 98.76 | 98.60 |
| seniority exact %, title-recoverable rows | 97.50 | 97.60 | 97.37 |
| seniority title-only ceiling % (oracle) | 92.26 | 92.51 | 92.51 |
| management (manages people) accuracy % | 98.83 | 98.84 | 98.79 |
| manager precision / recall % | 95.94 / 96.41 | 95.96 / 96.45 | 95.93 / 96.11 |
| Staff/Principal titles flagged as managers % | 0.0 | 0.0 | 0.0 |
| department coverage % | 99.17 | 99.30 | 99.22 |
| department accuracy % | 99.23 | 99.21 | 99.17 |
| department accuracy when the field is missing % | 99.03 | 98.90 | 98.84 |
| specialty precision % | 95.43 | 95.00 | 94.95 |
| scope precision / recall (where shown) % | 99.68 / 88.18 | 99.04 / 96.57 | 99.16 / 86.60 |
| board/advisor precision / recall % | 100 / 97.7 | 100 / 98.0 | 100 / 98.7 |

Exact seniority is 91.0%, within 0.3 points of the 92.5% title-only oracle.
The remaining gap is mostly label-convention disagreement (see Limitations).
Train, calibration and test scores agree within about 0.2 points, so there is
no sign of overfitting to train.

**Held-out title styles (test).** Rows with styles never seen in train scored
92.98% exact seniority, against 90.83% for other test rows. Management
accuracy was 99.34% against 98.73%. By style:

| style | exact seniority % | department % |
|---|---|---|
| abbreviations | 94.62 | 98.96 |
| typos | 94.06 | 99.26 |
| separators | 88.89 | 95.56 |
| reordered qualifiers | 76.78 | 95.79 |

**By industry (test).**

| industry | exact seniority % | within ±1 % | department % | management % |
|---|---|---|---|---|
| consulting | 92.33 | 98.48 | 98.41 | 99.09 |
| fintech | 91.04 | 98.28 | 99.40 | 98.53 |
| healthcare | 90.80 | 98.96 | 99.89 | 99.26 |
| manufacturing | 91.86 | 98.85 | 99.94 | 99.30 |
| retail | 91.66 | 98.64 | 97.79 | 97.78 |
| SaaS | 89.58 | 98.50 | 99.58 | 98.92 |

**Confidence is informative (test).** All three fields (level, department,
management) are correct for:

| parse_confidence | rows | all three correct |
|---|---|---|
| ≥ 0.70 | 81% | about 93% |
| 0.50–0.70 | 15% | 85% |
| below 0.50 | 2% | 45% |

Confidences are rule-assigned and not yet calibrated; calibration is
Phase 5. Within the top band, the 0.85–1.00 bucket is slightly less accurate
than the 0.70–0.85 bucket, which calibration will correct.

## Unresolved titles (all splits)

| reason | rows | meaning |
|---|---|---|
| `EMPTY_TITLE` | 1,012 | no title on the profile; level unresolved |
| `VAGUE_TITLE` | 4,101 | "Manager", "Employee", "Specialist", "Leader", "Team Member", "Associate", "Consultant" outside consulting: level is a weak guess (confidence ≤ 0.3); department taken from the field if present |
| `DEPT_UNRESOLVED` | 1,368 | no function in the title and no department field (e.g. "Intern", "Team Leader", "Senior Manager") |
| `LEVEL_UNRESOLVED` | 1,015 | almost entirely the empty titles |
| `DEPT_CONFLICT` | 6,634 | title and field disagree; the conflict is recorded and confidence lowered |
| `AMBIGUOUS_ACRONYM` | 11 | CRO/CPO/CCO/CMO with no department field |

Overall, 99.22% of profiles have a resolved level and 98.9% a resolved
department. The 603 board and advisor profiles are flagged as non-employees.

## Known limitations

* **Circularity.** The same author wrote the generator and the lexicon. The
  lexicon was written separately and tuned only on train errors, but its
  vocabulary is informed by the generator's. Accuracy on real profiles will be
  lower, mainly through out-of-vocabulary roles.
* **Label-convention disagreements**, kept on purpose rather than fitted to
  the generator's labels:
  * SDR/BDR: the parser says level 1; the generator uses 1 or 2, and level 1
    is the train majority.
  * Paralegal and legal assistant: split between levels 1 and 2.
  * Customer Success Manager: IC in about 84% of train rows.
  * "Head of X": true level ranges from 5 to 9 depending on company size.
    The parser uses 7 with confidence 0.5.
  * Financial Controller: 6 vs 7. Product Lead: 4 vs 5.
  * Senior pharmacist and sonographer.
* **Unrecoverable titles.** Stale titles (1.5% exact seniority on test) and
  inflated titles (7.7%) describe a different level from the person's real
  one; no title parser can fix them. Phase 3 should use structural context and
  start dates instead.
* **Ambiguous roles.**
  * Buyer and visual merchandiser can belong to merchandising, supply chain or
    retail operations.
  * Consulting "Manager, Digital" can belong to marketing or consulting.
  * The department field resolves these only when it is present.
* **Reordered qualifiers** (a held-out style) are the weakest style, at 76.8%
  exact seniority.
* **Scope recall** covers only scopes written in the title. Units are found
  only when at least three colleagues share the qualifier.
* **Typo repair** fixes only single adjacent-letter swaps in words of at least
  five letters.
* **Calibration.** Confidence values are ordinal rule outputs and are not yet
  calibrated probabilities (Phase 5).

## Tests

* `tests/test_normalization.py`: table-driven unit tests across all six
  industries, covering:
  * Staff/Principal as ICs; IC roles named "Manager";
  * consulting grades; the nursing ladder; the retail store hierarchy;
    manufacturing roles; fintech CRO resolution;
  * abbreviations and typos; vague, empty and board titles; EA and chief of
    staff;
  * flags; scope; the company-context pass; immutability of cached results;
  * bounded confidences and non-empty explanations;
  * a hypothesis fuzz test.
* `tests/test_normalization_pipeline.py`: end-to-end on a generated dataset.
  It checks the schema, one row per observed employee, determinism, minimum
  accuracy on the evaluator, and the CLI.
* `tests/test_isolation.py`: covers the new modules automatically.
