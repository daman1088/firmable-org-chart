# Architecture

This is the approved design, including the review adjustments. All phases
(0–7) are implemented. Decisions made during implementation are recorded under
"Implementation decisions" below.

## Principles (from review)

1. Every company has **≥ 100 employees in the observed dataset**, after
   missing-employee simulation (a hard guarantee checked by validation).
2. Stack: Python, DuckDB, Parquet, pytest, Streamlit.
3. Start with an **explainable deterministic baseline**. Add learned scoring or
   structural optimisation only if they produce measured improvements.
4. Ground-truth reporting relationships are **completely isolated** from
   inference.
5. Precision over recall: **never force** a reporting line when confidence is
   insufficient.
6. 95% precision is a **target, not a guarantee**. Always report the actual
   precision and coverage.
7. Output statuses: `DIRECT`, `ANCESTOR`, `AMBIGUOUS`, `UNRESOLVED`. Direct
   reporting is kept distinct from broader organisational ancestry.
8. Separate **train / calibration / test** companies.
9. Reproducible and fully documented.

## Data flow

```
generation (1)                 normalisation (2)            inference (3-5)                       evaluation
plan -> tree -> titles ->      observed/*.parquet ->        normalized_titles + observed          predictions + truth
people -> coverage + noise     derived/normalized_titles    + models/*.json  -> candidates ->     -> precision, coverage,
  -> observed/  truth/                                      features -> scores -> decide           recall, calibration,
                                                            -> derived/predictions*.parquet        breakdowns, errors
                               learning (4-5): truth for TRAIN (weights) and CALIBRATION (isotonic maps,
                               thresholds, selection) companies -> models/model_*.json
dashboard (6): explore mode = observed + derived + models;  evaluation mode (separate module) = + truth
```

| Package | May read truth? | Notes |
|---|---|---|
| `orgchart.normalization`, `orgchart.inference`, `app/` (explore) | no | statically enforced by `tests/test_isolation.py` |
| `orgchart.learning` | train and calibration companies | output is model JSON only; a test fails if it names the test split |
| `orgchart.evaluation`, `app/evaluation_view.py`, dataset validation | yes | evaluation view is imported lazily, only in evaluation mode |

## Splits

Splits are assigned per company at planning time, stratified by
(industry, size band), in a 70/15/15 ratio (600 companies → 420 / 90 / 90):

* **train**: tuning rules and lexicons, and fitting learned weights if Phase 4 is justified.
* **calibration**: fitting confidence calibration and choosing the acceptance threshold.
* **test**: final reported metrics only. These companies also get held-out title-noise styles.

Learned components may read truth labels for **train** companies only, through
a dedicated training module (Phase 4). That module is not part of the
inference runtime.

## Inference output contract (`derived/predictions.parquet`)

One row per observed profile, with the following fields:

* `status`: one of `DIRECT`, `ANCESTOR`, `AMBIGUOUS`, `UNRESOLVED`.
* `reports_to_id`: present for `DIRECT` only. It is named so that inference
  code never contains the truth column name `manager_id`, which the
  isolation guard forbids.
* `ancestor_id`: present for `ANCESTOR`. The person is somewhere above the
  employee; this is never stored as a direct edge.
* `candidate_ids`: present for `AMBIGUOUS` (2–3 candidates).
* `confidence`: the calibrated probability that the status claim is right.
  For AMBIGUOUS it is the set's probability mass.
* `top_candidate_id`, `direct_confidence`, `runner_up_id`,
  `runner_up_confidence`, `n_candidates`.
* `reason_codes`, plus `explanation`: readable lines with per-feature
  contributions, the runner-up and the margin. Full per-candidate features
  are in `derived/candidates.parquet`.

The evaluation scores each status differently:

* `DIRECT` is correct if `manager_id` equals the true manager.
* `ANCESTOR` is correct if `ancestor_id` is a true ancestor.
* `AMBIGUOUS` is scored by whether the true manager is in the candidate set.

Recall is reported both against all employees and against *resolvable*
employees, meaning those whose true manager is observed (`truth.nearest_observed_manager_id`).
Skip-level DIRECT answers count as errors and are tallied separately.

## Phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | Scaffold, schemas, config, CLI, tests | done |
| 1 | Generator, validation, 600-company dataset | done: all structural checks pass, realism targets met |
| 2 | Title normalisation (independent lexicon) | done: test seniority 91.0% exact / 98.6% ±1, department 99.2%, see title_normalization.md |
| 3 | Evaluation harness + deterministic baseline | done: rules and naive baselines, per-status precision / coverage / recall, breakdowns |
| 4 | Learned scoring / structural optimisation | done: listwise conditional logit adopted (+46 pp any-claim coverage, mostly ANCESTOR, and +0.5 pp DIRECT on calibration); no structural optimiser (see below) |
| 5 | Calibration, threshold selection, explanations | done: isotonic maps, cascade thresholds at 95% (plus 90% and 80% operating points); test ECE 1.1% |
| 6 | Streamlit explorer | done: explore mode and separated evaluation mode, threshold slider, candidate explanations |
| 7 | Hardening, write-up | done: `make verify`, `orgchart pipeline`, determinism checks, docs, demo walkthrough |

## Implementation decisions (Phases 3–7)

* **Strict seniority as the only structural constraint.** Candidates must be
  strictly more senior and hold management authority. That alone guarantees
  acyclic DIRECT edges and at most one manager per employee. A global
  assignment optimiser (span limits, matching) was not added. The dominant
  error source is indistinguishable parallel teams, which an optimiser cannot
  resolve without new evidence.
* **Listwise direct head with an outside option.** This models "manager not
  observed" explicitly and makes competing candidates share probability. It
  replaced a pairwise logistic first version, which reached 76% precision at
  5% coverage on calibration; the listwise model reached 85%.
* **ANCESTOR is never the CEO.** "Reports up to the CEO" is true of everyone,
  so the lowest confident ancestor is reported instead.
* **Thresholds must hold at every higher threshold too**, with at least 100
  supporting predictions, so a noisy precision spike cannot set them.
* **The rules baseline is kept** as the deterministic reference even though
  it is not selected. On test it is more precise for DIRECT (95.9% vs 94.1%)
  but covers 0.33% of employees against 0.87%, and makes no ANCESTOR claims.
* **DIRECT and ANCESTOR are presented separately everywhere** (reports, README, dashboard
  cards, columns, edge styles). An ancestor is drawn as a dashed "ancestor relationship" edge
  across a skipped rank, never as a reporting line.
