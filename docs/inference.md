# Reporting-line inference (Phases 3–5)

This stage predicts, for every observed profile, who that person reports to. Each
prediction carries a calibrated confidence and an explanation, and the system
abstains whenever the evidence is not strong enough.

```bash
orgchart fit      data/full   # learned weights (train), calibration + thresholds + selection (calibration)
orgchart infer    data/full   # predictions for all 600 companies (observed data + model files only)
orgchart check-predictions data/full
orgchart evaluate data/full   # metrics vs ground truth -> reports/inference_eval.{json,md}
orgchart explain  E0da949b316a4 --dataset data/full
```

## Isolation

| Package | Reads | Role |
|---|---|---|
| `orgchart.inference` | observed data, `derived/normalized_titles.parquet`, `models/*.json` | builds candidates, scores, decides |
| `orgchart.learning` | ground truth for **train** (weights) and **calibration** (calibration maps, thresholds, selection) companies | writes `models/*.json`; never reads test companies |
| `orgchart.evaluation` | ground truth | scores predictions; headline numbers come from **test** companies |

`tests/test_isolation.py` fails the build if inference or the explorer imports
generation, truth, evaluation or learning code, or references truth paths,
tables or columns. `tests/test_inference_pipeline.py` fails if the learning
code names the test split at all.

## 1. Candidate generation (`inference/features.py`)

Candidates for employee *e* are observed people in **the same company** who:

* hold people-management authority (manager or executive track, from Phase 2);
* are **strictly more senior** than *e* (normalised level).

These two hard constraints have direct consequences:

* Cross-company edges and self-reporting are impossible.
* Every set of predicted edges is acyclic.
* Board and advisor profiles, and profiles whose level could not be read,
  never take part in candidate generation.

A cheap prefilter keeps the top 20 candidates by direct fit and the top 5 by
ancestor fit. The true manager is in the shortlist for 88% of training
employees whose manager is observed. Most of the rest fall into two groups:

* large groups of interchangeable candidates (e.g. 30 Store Managers);
* Phase 2 label conventions, e.g. a "Senior Legal Counsel" who manages people.

## 2. Interpretable features

There are 40 features per (employee, candidate) pair:

* **Level gap**, separately for ICs and managers.
* **Managers sitting between them** in the same department, in the
  closest-matching team (department, specialty, scope, office), and with an
  unreadable department.
* **Matches and mismatches** in department, specialty, region or unit scope,
  and office city or country.
* **Equally plausible competitors**: same level, same department, same match
  pattern.
* **Function head reporting to an executive**, and executive support roles.
* **Shared role vocabulary** between the two titles.
* **Uncertainty flags** from Phase 2: low seniority confidence for the
  employee, uncertain management authority for the candidate.
* **Start-date order**.

Each feature has a plain-English label, used in explanations.

## 3. Scoring (`inference/scoring.py`)

Both variants are linear, so every score decomposes into per-feature
contributions:

| Variant | Weights |
|---|---|
| `naive` | no scoring: the nearest more senior manager in the same department, always predicted DIRECT (the "always guess" reference) |
| `rules` | hand-set from domain reasoning (`RULE_DIRECT`, `RULE_ANCESTOR`); the deterministic explainable baseline |
| `learned` | fitted on **train** companies with the same features |

Each variant has two heads:

* **Direct head (listwise).** For each employee,
  `q_m = exp(s_m) / (1 + Σ_k exp(s_k))`. The `1` is an explicit outside option
  meaning "the manager is not among the observed candidates". Probabilities sum
  to at most 1 per employee, competing candidates share probability, and a
  missing manager is modelled directly. Learned weights come from a
  conditional-logit fit with Newton's method and line search
  (`learning/estimators.py`).
* **Ancestor head (pairwise).** Logistic regression for "m is somewhere above e".
  Several candidates can be true ancestors at once, so this head is not listwise.

### Why a listwise model

A pairwise logistic first version reached 76% precision at 5% coverage on
calibration. The listwise model reached 85% at the same coverage, because it
lets equally good candidates compete.

## 4. Calibration and thresholds (`learning/fit.py`, calibration companies only)

* **Calibration maps** (isotonic, fitted with pool-adjacent-violators):
  * the top candidate's q to P(it is the direct manager);
  * the raw ancestor score to P(true ancestor).
* **Decision cascade** (`inference/decide.py`):
  1. **DIRECT** if the calibrated top probability ≥ `t_direct`.
  2. **AMBIGUOUS** if the 2–3 best candidates together reach `t_ambiguous`
     probability mass.
  3. **ANCESTOR** if a non-CEO candidate's ancestor probability ≥
     `t_ancestor`. The lowest-level such candidate is reported, because it is
     the most informative. The CEO is excluded because everyone sits under the
     CEO.
  4. **UNRESOLVED** otherwise, with a reason code.
* **Threshold selection.** Each threshold is the lowest one at which precision
  reaches the **95% target**, at that threshold and at every higher one with
  at least 100 predictions. They are chosen in cascade order on calibration
  companies. Thresholds for **90%** and **80%** targets are stored too.
* **Model selection.** `learned` is adopted only if, on calibration, it
  improves any-claim ("resolved") coverage by ≥ 1 pp and does not lower DIRECT coverage.
  Measured: +46.0 pp coverage of any claim (almost all of it ANCESTOR) and
  +0.5 pp DIRECT coverage, so `learned` was selected.

## 5. Explanations

Every prediction stores reason codes and an explanation list:

* the chosen manager and its probability;
* the top weighted feature contributions;
* the runner-up and the probability margin.

`candidates.parquet` keeps the five best candidates per employee with feature
values and calibrated probabilities. The dashboard and `orgchart explain` show
contribution charts from it.

Some features describe only the employee (e.g. "number of plausible managers").
They shift probability between "some candidate" and "manager not observed"
but never reorder candidates, so they are left out of candidate explanations.

Reason codes:

* DIRECT: `CONFIDENT_DIRECT`, `CLOSE_RUNNER_UP`.
* AMBIGUOUS: `TIED_CANDIDATES`, `LOW_MARGIN`.
* ANCESTOR: `DIRECT_MANAGER_UNCERTAIN`, `LIKELY_MANAGER_UNOBSERVED`.
* UNRESOLVED: `INSUFFICIENT_EVIDENCE`, `TOP_OF_ORG`, `NO_CANDIDATES`,
  `LEVEL_UNRESOLVED`, `NON_EMPLOYEE`.

## Results: held-out test companies (90 companies, 29,512 employees)

Thresholds and model choice were fixed on calibration companies before test
was scored. Full report: `data/full/reports/inference_eval.md`.

**Scoring rules.** The statuses make different claims, so each is scored by
its own rule and reported separately:

* **DIRECT** is correct only if it names the exact true manager. A
  skip-level manager counts as wrong.
* **ANCESTOR** is correct if the named person is anywhere above the employee.
  It is **not** a direct-manager claim, so its precision is not manager
  accuracy. On test, 13% of ANCESTOR claims happen to name the direct manager,
  and correct ones are a median of 2 levels up.
* **AMBIGUOUS** is correct if the true manager is in the set.
* "Any claim" pools all three. It answers "how often is a statement we make
  correct?", not "how often do we name the right manager?".

**Definitions:**
* Coverage = share of employees that receive the status.
* Recall (resolvable) = correct DIRECT / employees whose true manager is
  observed.

### Baseline comparison (95% target)

| Test metric | naive (always guess) | rules | learned (selected) |
|---|---|---|---|
| DIRECT coverage / precision | 99.1% / **19.7%** | 0.33% / 95.9% | 0.87% / 94.1% |
| DIRECT recall (resolvable) | 23.3% | 0.38% | 0.97% |
| AMBIGUOUS coverage / containment | — | — | 0.58% / 90.6% |
| ANCESTOR coverage / precision | — | — | 39.8% / 95.4% |
| Any claim, mixed (not manager accuracy): coverage / precision | 99.1% / 19.7% | 0.33% / 95.9% | 41.3% / 95.3% |
| Forced top-1 accuracy (no abstention) | — | 26.8% | 31.7% |

### Operating points (selected model; thresholds set on calibration, applied once to test)

| Precision target | DIRECT (exact manager) cov / prec | ANCESTOR (in chain above) cov / prec | AMBIGUOUS cov / in set | Any claim, mixed: cov / prec |
|---|---|---|---|---|
| 95% | 0.87% / 94.1% | 39.8% / 95.4% | 0.58% / 90.6% | 41.3% / 95.3% |
| 90% | 3.3% / 89.7% | 50.2% / 90.9% | off | 53.5% / 90.8% |
| 80% | 10.6% / 81.2% | 59.2% / 80.8% | 3.1% / 83.2% | 72.9% / 81.0% |

"Off" means no AMBIGUOUS threshold reached 90% on calibration, so that status
is not used at that target. The dashboard's **Precision targets** tab shows
these tables, and its sidebar selector applies each target to any company.

### Calibration (test)

| Quantity | ECE | Brier |
|---|---|---|
| Top-candidate direct probability (29,258 employees) | 1.1% | 0.155 |
| DIRECT confidences | 2.2% | — |
| ANCESTOR confidences | 1.0% | — |

In each probability decile, observed accuracy is within about 3 points of the mean
predicted probability. For example, the 0.9–1.0 bin has mean 0.929 and
observed 0.929.

### Breakdowns (test, selected model)

The slice figures pool all claims. DIRECT-only figures are given in the notes where they matter;
every slice is split by status in `data/full/reports/inference_eval.md`.

| Slice | Any-claim (mixed) coverage / precision | Notes |
|---|---|---|
| SaaS · fintech · healthcare · manufacturing · retail | coverage 55.3% · 42.0% · 36.7% · 36.9% · 44.5%, precision 94.7–96.8% | |
| Consulting | 22% / 92.7% | flat grade titles ("Manager", "Consultant") are least informative |
| Companies of 100–199 / 200–499 / 500–999 / 1000+ | 54% / 41% / 36% / 48% coverage | |
| ICs (L0–3) | 41.6% / 95.9% | mostly ANCESTOR |
| Managers (L5–6) | 47.1% / 92.4% | DIRECT 4.1% at 93.6% |
| Directors (L7–8) | 25.6% / 88.6% | DIRECT 11.0% at 94.3% |
| Functional / regional / divisional structure | 43% / 42% / 26% coverage | precision 94.8–97.1% |
| Manager not observed (16% of test employees) | 36.8% / 95.3% | almost all ANCESTOR, which is the right fallback |

DIRECT errors on test number 15:

* **12 skip-level.** Each named the manager's own manager. In most of them the
  true manager has a vague title ("Manager", "Department Mgr.") with no
  department.
* **2 wrong-branch.**
* **1 where the true manager is unobserved.**

## Why DIRECT coverage is low at 95%, and what that means

* **Missing managers.** Without abstention, the best top-1 accuracy is 31.7%.
  About 16% of employees have an unobserved true manager, so most direct
  probabilities are capped near 0.85.
* **Parallel teams.** 77% of employees are ICs in teams the observed fields
  often cannot tell apart. Titles like "Software Engineer" in an office with
  five engineering managers give no specialty to separate them.
* **Ceiling.** A diagnostic with *true* department, specialty, scope, office
  and level finds the true manager unique among candidates only 69% of the
  time.
* **Abstention.** Rather than guess, the system answers what it can support:
  it names a confident ancestor for 40% of employees at 95% precision, and the
  threshold slider in the dashboard exposes the measured trade-off.

## Limitations

* **Synthetic data.** Results are measured on synthetic data from the same
  author as the lexicon (see `title_normalization.md`). Real profile data will
  have a longer tail of titles and noise.
* **Linear scorer.** Interactions are limited to a few hand-chosen crosses.
  Gradient-boosted trees might rank better, but they would lose exact additive
  explanations. They were not tried, because the remaining errors are mostly
  inherent ambiguity, not a modelling limit.
* **One manager per person.** There are no dotted lines or matrix reporting,
  matching the generator.
* **Level-1 calibration.** Calibration is fitted on the top candidate. Other
  candidates' probabilities use the same monotone map, which makes them
  approximate.
* **Small evaluation sets.** At the 95% target, test has only 256 DIRECT and
  170 AMBIGUOUS predictions, so those precisions have confidence intervals of
  roughly ±3–5 points. Test DIRECT precision (94.1%) and AMBIGUOUS containment
  (90.6%) fall just short of the 95% target. ANCESTOR (95.4%) and the overall
  claim precision (95.3%) meet it.
* **Bounded tree depth.** Strict seniority ordering forbids same-level
  reporting. Real organisations occasionally have it.
