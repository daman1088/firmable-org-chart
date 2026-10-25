# Data dictionary

Schemas are defined in `src/orgchart/schemas.py` and enforced on every read
and write.

## observed/companies.parquet — inference input

| Column | Type | Notes |
|---|---|---|
| company_id | string | `C0000`… (planning order; carries no hierarchy information) |
| company_name | string | e.g. "Apex Ridge Health Pty Ltd" |
| domain | string | |
| industry | string | LinkedIn-style industry label |
| hq_city, hq_country | string | |
| headcount_band | string | public-style band from true headcount, e.g. "201-500" |
| founded_year | int16 | |
| offices | list<string> | office cities, HQ first |
| split | string | `train` / `calibration` / `test` |

## observed/employees.parquet — inference input

| Column | Type | Notes |
|---|---|---|
| employee_id | string | random `E` + 12 hex; rows are ordered by this |
| company_id | string | |
| full_name | string | |
| title | string, nullable | raw noisy title; null for about 0.5% |
| department | string, nullable | raw label variant; null for about 30%, mislabelled for about 3% |
| location | string, nullable | "Sydney, New South Wales, Australia", "Greater Sydney Area", "Remote", … |
| start_date | date, nullable | month precision; null for about 20% |

Includes 0–4 board/advisor profiles per company. These are not part of the
reporting tree.

## truth/companies.parquet — evaluation only

| Column | Type | Notes |
|---|---|---|
| company_id | string | |
| archetype | string | saas, fintech, manufacturing, healthcare, retail, consulting |
| scope_kind | string | `none` / `regional` / `divisional` |
| true_headcount | int32 | employees in the true tree |
| observed_employee_count | int32 | observed employees, excluding board profiles (always ≥ 100) |
| observed_profile_count | int32 | observed rows, including board profiles |
| coverage_target, coverage_actual | float64 | |
| max_depth | int16 | CEO = 0 |
| departments | list<string> | canonical department keys present |
| holdout_noise | bool | true for test companies |
| company_seed | uint64 | regenerate a single company from this seed |

## truth/employees.parquet — evaluation only

Contains every true employee, observed or not, plus board profiles.

| Column | Type | Notes |
|---|---|---|
| employee_id | string | joins to observed |
| company_id | string | |
| manager_id | string, nullable | true direct manager; null for the CEO and board profiles |
| level / level_name | int8 / string | 0 intern … 12 CEO (`orgchart.ladder.Level`); null for board |
| department | string | canonical key (`engineering`, `clinical`, `executive`, `board`, …) |
| specialty | string, nullable | team specialty (Backend, FP&A, Nursing, …) |
| scope | string, nullable | region or division |
| role_type | string | ceo, head, manager, ic, assistant, chief_of_staff, board |
| employment_type | string | full_time, contractor, intern, board |
| canonical_title | string | clean title before noise |
| is_people_manager / n_direct_reports | bool / int32 | counts all true reports, observed or not |
| depth | int16, nullable | CEO = 0 |
| is_observed | bool | appears in observed/employees |
| nearest_observed_manager_id | string, nullable | closest observed ancestor; the best achievable answer when the direct manager is missing |
| noise_tags | list<string> | observation transforms applied (`stale_title`, `abbreviate_terms`, `holdout:typo`, `dept_missing`, …) |

## derived/normalized_titles.parquet: Phase 2 output (inference-safe)

Built by `orgchart normalize` from observed tables only. There is one row per
observed employee.

| column | type | notes |
|---|---|---|
| employee_id / company_id | string | join to observed |
| title_raw / department_raw | string, nullable | inputs as observed |
| title_clean | string, nullable | cleaned main phrase plus qualifiers (lower case, abbreviations expanded) |
| level / level_name | int8 / string, nullable | 0 intern … 12 CEO; null when unresolved or for a non-employee |
| level_confidence | float | 0–1 |
| track | string | executive, manager, ic, support, board, unknown |
| manages_people / management_confidence | bool, nullable / float | people-management authority. Staff/Principal ICs are false |
| department / department_confidence | string, nullable / float | normaliser keys (`hardware_engineering`, `retail_operations` differ from the generator's) |
| department_source | string | title, field, title+field, industry, none |
| specialty / specialty_confidence | string, nullable / float | e.g. Backend, Nursing, Payroll |
| scope / scope_type | string, nullable | region (APAC, EMEA, …) or unit (company-recurring qualifier) |
| is_contractor / is_intern / is_interim / is_founder / is_non_employee | bool | flags from markers. Non-employee means board or advisor |
| parse_confidence | float | min(level, department, management) confidence; 0 when any is unresolved |
| reason_codes | list<string> | VAGUE_TITLE, EMPTY_TITLE, LEVEL_UNRESOLVED, DEPT_UNRESOLVED, DEPT_CONFLICT, AMBIGUOUS_ACRONYM, TYPO_CORRECTED, UNKNOWN_ROLE |
| rules_fired | list<string> | rule identifiers, in order |
| explanation | list<string> | plain-English trace of each decision |
| normalizer_version | string | `orgchart.normalization.parser.NORMALIZER_VERSION` |

## derived/predictions.parquet: inference output (inference-safe)

Built by `orgchart infer`. There is one row per observed profile.
`predictions_{rules,learned,naive}.parquet` have the same schema for each
variant; `predictions.parquet` is the selected one.

| column | type | notes |
|---|---|---|
| employee_id / company_id | string | |
| status | string | DIRECT, ANCESTOR, AMBIGUOUS, UNRESOLVED |
| reports_to_id | string, nullable | predicted direct manager (DIRECT only) |
| ancestor_id | string, nullable | someone above the employee, not necessarily direct (ANCESTOR only; never the CEO) |
| candidate_ids | list<string> | 2–3 possible direct managers (AMBIGUOUS only) |
| confidence | float, nullable | calibrated probability that the claim is right (set mass for AMBIGUOUS); null when UNRESOLVED |
| top_candidate_id / direct_confidence | string / float, nullable | best direct candidate and its calibrated probability, whatever the status (drives the dashboard slider) |
| runner_up_id / runner_up_confidence | string / float, nullable | second-best direct candidate |
| n_candidates | int32 | shortlisted candidates |
| reason_codes | list<string> | CONFIDENT_DIRECT, CLOSE_RUNNER_UP, TIED_CANDIDATES, LOW_MARGIN, DIRECT_MANAGER_UNCERTAIN, LIKELY_MANAGER_UNOBSERVED, INSUFFICIENT_EVIDENCE, TOP_OF_ORG, NO_CANDIDATES, LEVEL_UNRESOLVED, NON_EMPLOYEE (naive variant: NAIVE) |
| explanation | list<string> | decision, top feature contributions, runner-up and margin |
| model_variant | string | naive, rules or learned |

## derived/candidates.parquet: ranked candidates (selected variant)

These are the top 5 candidates per employee, plus the reported ancestor.

| column | type | notes |
|---|---|---|
| company_id / employee_id / candidate_id | string | |
| rank | int16 | 0 = best direct candidate |
| q_direct / p_direct | float | listwise share / calibrated probability of being the direct manager |
| p_ancestor | float | calibrated probability of being above the employee |
| score_direct / score_ancestor | float | raw linear scores (sum of weight × feature) |
| e_level / m_level / m_is_ceo | int8 / int8 / bool | |
| n_between_dept / n_between_strict / n_equiv | int16 | managers between them (same department / closest-matching team); equally plausible competitors |
| 40 feature columns | float32 | `orgchart.inference.features.FEATURES`; labels in `FEATURE_LABELS` |

## models/

| file | contents |
|---|---|
| model_rules.json / model_learned.json | feature list, direct and ancestor weights and bias, isotonic knots, thresholds, operating points for 95 / 90 / 80% targets, and calibration-split metrics |
| model_selected.json | copy of the selected variant, plus the selection rule and the measured gain |
| fit_summary.json | calibration-split comparison of the variants (deterministic: no timings) |

