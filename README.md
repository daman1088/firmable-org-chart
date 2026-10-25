# Org Chart Inference Platform

This platform infers who reports to whom inside companies from public-profile
style data: job title, department, location and start date. Every predicted
reporting line carries a calibrated confidence and a feature-level
explanation.

Precision is prioritised over recall. Every employee gets exactly one of four
statuses, and only the first names a reporting line:

| Status | What it claims | Scored correct when |
|---|---|---|
| **DIRECT** | *X is this person's direct manager.* | X is the exact true manager (a skip-level answer is wrong) |
| **AMBIGUOUS** | *The direct manager is one of these 2–3 people.* | the true manager is in the set |
| **ANCESTOR** | *X is somewhere in the reporting chain above this person.* It does not say whether X is the direct manager or someone higher (on test, 13% of these name the direct manager). | X is anywhere in the true chain above the person |
| **UNRESOLVED** | No claim, with a reason code. | — |

DIRECT and ANCESTOR are different claims with different scoring rules, so this
README, the reports and the dashboard always report them separately.

Real org charts are rarely available as ground truth, so the project ships a
**synthetic data generator**. It builds true org trees for 600 companies and
derives a noisy, incomplete "observed" view of them. Inference sees only the
observed view, evaluation compares its output with the hidden truth, and
tests enforce that boundary.

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffolding: package, config, schemas, storage, CLI, tests | done |
| 1 | Synthetic data generator and validation (600 companies) | done |
| 2 | Title normalisation | done |
| 3 | Candidate generation, deterministic baseline, naive baseline, evaluation harness | done |
| 4 | Learned scoring: listwise conditional logit, adopted on measured gain | done |
| 5 | Isotonic calibration, threshold selection, explanations, reason codes | done |
| 6 | Streamlit dashboard: explore mode plus separated evaluation mode | done |
| 7 | Validation, `make verify`, reproducibility, documentation | done |

Documents:

* [docs/architecture.md](docs/architecture.md): design and data flow.
* [docs/data_generation.md](docs/data_generation.md): the synthetic data.
* [docs/title_normalization.md](docs/title_normalization.md): Phase 2.
* [docs/inference.md](docs/inference.md): methodology and full results.
* [docs/data_dictionary.md](docs/data_dictionary.md): every column.
* [docs/demo.md](docs/demo.md): a 10-minute reviewer walkthrough.

## Quickstart

**Python 3.12–3.15.** Tested end to end on CPython 3.15.0 (macOS arm64): a fresh checkout,
`pip install -r requirements.lock`, the full pipeline, the test suite and `make verify`'s checks.

| Python | `requirements.lock` |
|---|---|
| 3.15 | Installed and tested. PyYAML 6.0.3 has no 3.15 wheel, so pip builds it from source (pure-Python build; no compiler was needed). |
| 3.12, 3.13, 3.14 | Resolver check only (`pip install --dry-run --only-binary=:all:`): every pin has a wheel for Linux x86_64/aarch64, macOS Intel/arm64 and Windows. Not run end to end. |
| 3.11 or older | Not supported: the pinned numpy 2.5.4 needs Python ≥ 3.12. |

Generated data does not ship with the code: `data/` is excluded from the submission and is rebuilt
by the pipeline (about 2 minutes). The dashboard shows a "Dataset is not built yet" message until then.

```bash
make setup                 # venv + pinned deps (requirements.lock) + editable install
                           # uses python3; pick another: make setup PYTHON=python3.13
make pipeline GENERATE=1   # generate 600 companies -> normalise -> fit -> infer -> check -> evaluate (~110 s)
make verify                # tests + dataset checks + prediction invariants + delivery checks; non-zero on failure
make app                   # dashboard at http://localhost:8501
```

Without `make` (this is the sequence that was run on a fresh checkout):

```bash
python3.15 -m venv .venv   # or any 3.12–3.15 interpreter
.venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e '.[dev]' --no-deps
.venv/bin/orgchart pipeline data/full --generate
.venv/bin/python -m pytest
.venv/bin/orgchart validate data/full
.venv/bin/orgchart check-predictions data/full
.venv/bin/orgchart check-predictions data/full --variant naive
.venv/bin/orgchart verify data/full
.venv/bin/streamlit run app/streamlit_app.py
```

If the lock cannot be installed on your platform, `pip install -e '.[dev]'` installs from the
version ranges in `pyproject.toml`. That route is untested, and other library versions may give
slightly different generated data and metrics from the figures in this README.

`make package` (or `sh scripts/package.sh`) writes `dist/orgchart-submission.zip`. The zip contains source,
tests, configs and docs only. It leaves out `.venv/`, `data/`, `.hypothesis/`, `.pytest_cache/`, `__pycache__/`,
`*.egg-info/`, `build/`, `dist/` and `.env`/secrets files.

## Pipeline and CLI

| Step | Command | Reads | Writes |
|---|---|---|---|
| generate | `orgchart generate --out data/full` | config | `observed/`, `truth/`, `manifest.json`, `reports/validation.json` |
| normalise | `orgchart normalize data/full` | observed | `derived/normalized_titles.parquet`, `reports/normalization_eval.*` |
| fit | `orgchart fit data/full` | observed + truth (train, calibration only) | `models/model_{rules,learned,selected}.json`, `models/fit_summary.json` |
| infer | `orgchart infer data/full` | observed + models | `derived/predictions*.parquet`, `derived/candidates.parquet` |
| check | `orgchart check-predictions data/full` | predictions + observed | exit 1 on any structural violation |
| evaluate | `orgchart evaluate data/full` | predictions + truth | `reports/inference_eval.{json,md}` |
| all | `orgchart pipeline data/full [--generate]` | | `reports/pipeline_runtime.json` |
| verify | `orgchart verify data/full` | everything | exit 1 if any delivery check fails |

Other commands:

* `orgchart explain EMPLOYEE_ID`: a prediction with its ranked candidates and
  feature contributions.
* `orgchart parse-title "Snr SWE II"`: Phase 2 parsing of one title.
* `orgchart validate`, `orgchart stats`, `orgchart sql "..." [--with-truth]`.
* `orgchart show-company C0042`: prints the **true** tree, for dataset review only.

**Measured runtime.** The full pipeline on 600 companies and 207,300 profiles
takes 104.6 s on an 8-core laptop:

| Step | Seconds |
|---|---|
| generate | 5.4 |
| normalise | 29.7 |
| fit | 23.4 |
| infer (rules, learned and naive) | 29.8 |
| check | 1.4 |
| evaluate | 14.9 |

## How inference works

Full details are in [docs/inference.md](docs/inference.md).

1. **Candidates.** People in the same company who have management authority
   and are **strictly more senior**. Cross-company edges, self-reporting and
   cycles are therefore impossible by construction. Board profiles and
   unreadable titles are excluded.
2. **Features.** 40 interpretable features: seniority gap; managers sitting
   in between; department, specialty, scope and office match; equally
   plausible competitors; function heads under executives; shared title
   vocabulary; and parse-uncertainty flags.
3. **Scoring.** Linear, so every score decomposes into per-feature
   contributions.
   * The **direct** head is *listwise*, with an explicit "manager not
     observed" option. Competing candidates share probability, and a missing
     manager is modelled directly.
   * The **ancestor** head is pairwise.
   * Variants: `naive` always guesses the nearest senior manager in the same
     department; `rules` uses hand-set weights (the deterministic baseline);
     `learned` uses weights fitted on train companies.
4. **Calibration and thresholds** use calibration companies only:
   * isotonic probability maps;
   * a decision cascade **DIRECT → AMBIGUOUS (≤ 3 candidates) → ANCESTOR
     (never the CEO) → UNRESOLVED**, with each threshold the lowest at which
     precision stays ≥ 95%;
   * stored alternative operating points for 90% and 80%.
5. **Selection.** `learned` is used only if it beats `rules` on calibration
   companies. It did: +46.0 pp coverage of any claim (mostly ANCESTOR), with
   DIRECT coverage not lower (+0.5 pp).

## Results on held-out test companies

The test split has 90 companies and 29,512 employees. It was never used for
fitting, calibration, thresholds or model selection. Coverage is the share of
test employees that receive the status.

**Exact direct manager (DIRECT):**

| Test metric | naive (always guess) | rules baseline | **selected (learned)** |
|---|---|---|---|
| DIRECT coverage / precision | 99.1% / 19.7% | 0.33% / 95.9% | **0.87% / 94.1%** (256 claims) |
| DIRECT recall (employees whose manager is observed) | 23.3% | 0.38% | 0.97% |
| Top-1 accuracy if forced to always answer | — | 26.8% | 31.7% |
| Calibration: ECE of the direct probability | — | — | 1.1% (Brier 0.155) |

**Not exact-manager claims, scored by their own rules:**

| Test metric | naive | rules | **selected (learned)** |
|---|---|---|---|
| ANCESTOR coverage / precision (named person is truly above) | — | — | **39.8% / 95.4%** (11,757 claims) |
| ANCESTOR named person happens to be the direct manager | — | — | 13% (median 2 levels up when correct) |
| AMBIGUOUS coverage / true manager in set | — | — | 0.58% / 90.6% (mean 2.5 candidates) |

Combining all claims, the selected model makes a statement about 41.3% of test
employees, and 95.3% of those statements are correct. This figure mixes
exact-manager, candidate-set and ancestor claims, so it is **not** a
direct-manager accuracy.

**Precision targets.** Thresholds were chosen on calibration companies for
each target, then applied once to test.

| Target | DIRECT (exact manager): coverage / precision | ANCESTOR (in chain above): coverage / precision | AMBIGUOUS: coverage / in set |
|---|---|---|---|
| 95% | 0.87% / 94.1% | 39.8% / 95.4% | 0.58% / 90.6% |
| 90% | 3.3% / 89.7% | 50.2% / 90.9% | off (no threshold met 90% on calibration) |
| 80% | 10.6% / 81.2% | 59.2% / 80.8% | 3.1% / 83.2% |

**What this means.**

* **Forcing an answer is unreliable.** About 16% of employees have an
  unobserved true manager, and most ICs sit in parallel teams that observed
  titles cannot tell apart. Even the best forced top-1 guess is right only
  31.7% of the time.
* **Exact direct managers at 95% precision are rare.** Only 0.87% of
  employees get one. DIRECT precision is 94.1%, just under the target, on
  256 test claims (roughly ±3 points). Relaxing the target to 80% raises
  DIRECT coverage to 10.6% at 81.2% precision.
* **Confident ancestors are common, but they are not direct-manager claims.**
  For 39.8% of employees, the system names someone confidently in the chain
  *above* them (95.4% precision). That person may be the direct manager or
  someone higher; the claim does not establish which.
* **Calibration makes the trade-off adjustable.** The dashboard's precision
  target selector (95 / 90 / 80% or a custom DIRECT threshold) applies these
  operating points to any company.

Breakdowns by industry, company size, seniority and organisation structure,
plus error analysis, are in `data/full/reports/inference_eval.md` and
[docs/inference.md](docs/inference.md). Phase 2 title normalisation scores
91.0% exact and 98.6% within ±1 level on test seniority, and 99.2% on
department. See [docs/title_normalization.md](docs/title_normalization.md).

## Dashboard

Run `make app` or `.venv/bin/streamlit run app/streamlit_app.py`. Set
`ORGCHART_DATASET` to point at another dataset.

**Explore mode** reads observed data, model files and predictions only:

* Company picker for all 600 companies, with an industry filter and
  cross-company employee search.
* **Status cards** labelled by what each status claims, e.g. "DIRECT · exact
  manager" and "ANCESTOR · in chain above".
* **Org chart.** Every chart is a readable top-down card.
  * **Views:** "Managers and leaders" (the default, about 20 boxes, adjustable
    from 10 to 60), "Around an employee" (their claimed manager or ancestor
    chain, their reports, and the people under the same person, with a "go
    to a nearby person" picker), "One department", and "Whole company".
  * **Grouping.** Only people who anchor predicted structure get their own
    box. Everyone else is listed in a box per (parent, kind of claim), with
    their own probability. People with no claim are not drawn; they are in
    the Review queue.
  * **Separate branches.** At high precision, most leaders have no confident
    claim above them, so the chart is a set of separate branches. Each branch
    is drawn on its own card rather than under an assumed common manager.
  * **Continued cards.** A branch too wide to read continues on further
    cards ("Team of …"). Layout → Auto falls back to left-to-right only for a
    card that cannot fit top-down.
  * **Edge styles.** DIRECT edges are solid green, labelled with their
    probability. ANCESTOR edges are dashed blue, labelled "ancestor
    relationship", drawn across a skipped level, and lead to lists headed
    "N employees linked through hierarchy / direct manager not identified".
    They are never drawn as reporting lines. AMBIGUOUS edges are dotted
    orange "?" links to each candidate.
* **Ask the Org Chart.** A question box for the selected company, with three
  example questions taken from that company. No language model or external
  service is used: questions are matched to three fixed patterns
  (`app/ask.py`), and answers are read from the same decisions the chart
  draws at the chosen precision target.
  * *"Who does Jane Smith report to?"* (also "Who is Jane Smith's manager?").
    A DIRECT claim gives the manager, title and probability. For ANCESTOR the
    answer says the direct manager is **not confidently identified** and
    names the person somewhere above, who may be the direct manager or
    someone higher. AMBIGUOUS lists the candidate set and its probability,
    and UNRESOLVED gives the reason codes.
  * *"Who is above Jane Smith?"* Follows DIRECT and ANCESTOR claims upwards,
    one claim per step, with each step's kind and probability. It stops where
    the next step is not claimed.
  * *"Who runs the Sales team?"* (also "leads", "heads", "head of"). Gives the
    most senior person in the department by normalised job title, labelled
    as an **inferred** leader rather than a reporting claim. A tie is
    reported as "no single leader", and the answer shows how many DIRECT
    claims name that person as manager. Department names and common labels
    ("HR", "revenue") are matched through the title lexicon.
  * People are found by employee ID, full name or whole words of the name.
    Several matches are listed for the user to choose from. An unknown name
    points to the companies where it does occur. Unsupported questions get
    the list of supported forms.
* **Precision target selector:** 95% (default), 90%, 80% or a custom DIRECT
  threshold. Each stored target re-applies its own calibration-chosen
  thresholds, and the sidebar shows its measured held-out DIRECT and
  ANCESTOR results separately.
* **Employees:** a searchable table with separate columns for exact manager,
  "in chain above (direct manager not identified)" and candidate sets, each with its own
  probability. Selecting a row shows the decision in words, the explanation,
  the ranked candidates and a feature contribution chart.
* **Precision targets** tab: the 95 / 90 / 80% comparison, with separate
  tables for DIRECT, ANCESTOR and AMBIGUOUS. It is built from aggregate
  numbers in `reports/inference_eval.json`, never from per-employee truth.
* **Review queue:** employees without an exact manager (AMBIGUOUS,
  UNRESOLVED, optionally ANCESTOR), with reason codes.
* **Model:** the thresholds in use, the selection rule and all weights.

**Evaluation mode** is explicitly separated in `app/evaluation_view.py` and
loaded only when selected. It provides:

* per-company comparison against ground truth, with separate DIRECT and
  ANCESTOR columns and the true number of levels for each ancestor claim;
* test metrics, the baseline comparison and the precision-target tables;
* the precision–coverage curve, the reliability diagram and breakdowns.

## Validation and reproducibility

* **`make verify`** runs pytest, then `orgchart validate`,
  `orgchart check-predictions` (for the selected and naive variants) and
  `orgchart verify`. It stops with a non-zero exit at the first failure.
  Current status: **243 tests passed**, all 16 dataset checks pass, all 10
  prediction invariants pass for every variant, and there are 600 companies
  with at least 100 observed employees each.
* **Reproducibility.**
  * A single seed drives generation; per-company seeds make output
    byte-identical for any `--workers`.
  * Regenerating `data/full` reproduced all manifest file hashes.
  * Re-running `fit` and `infer` reproduces the model JSON files and
    prediction Parquet files byte for byte.
  * `orgchart verify` re-runs inference on sample companies and compares the
    decisions.
* **Pinned environment.** Dependencies are pinned in `requirements.lock`.
  `manifest.json` records the resolved config, its fingerprint, file hashes
  and library versions.

## Ground-truth isolation

* Observed and truth tables live in separate directories with separate
  schemas, and observed schemas carry no hierarchy columns.
* `orgchart.inference`, `orgchart.normalization` and the explorer read only
  `orgchart.storage.observed`, derived files and model JSON files.
* `tests/test_isolation.py` statically scans them. It fails the build on any
  import of generation, truth storage, evaluation or learning code, and on any
  reference to truth paths, tables, attributes or columns. The one exempt file
  is the dashboard's evaluation module, which may only be imported lazily
  (also tested).
* `orgchart.learning` reads truth for **train** companies (weights) and
  **calibration** companies (calibration, thresholds, selection). A test fails
  if it ever names the test split.
* Employee IDs are random and rows are sorted by ID, so neither leaks the
  hierarchy. Some title-noise styles appear only in test companies.

## Known limitations

* **Synthetic data, one author.** The same author wrote the generator and the
  title lexicon. Real profiles have a longer tail of titles, duplicates and
  stale records. Expect lower accuracy, and re-fit and recalibrate on labelled
  real data before relying on thresholds.
* **Low DIRECT coverage at 95% precision (0.9%).** This is inherent to data
  with missing managers and indistinguishable parallel teams. The system
  reports ancestors and ambiguous sets instead.
* **Small evaluation samples.** At the 95% target, test DIRECT (n = 256) and
  AMBIGUOUS (n = 170) precision land at 94.1% and 90.6%, below target, with
  wide intervals.
* **Linear model.** Exact explanations come at some ranking power. Tree
  ensembles were not tried.
* **Tree constraints.** Exactly one manager per person, and strictly
  decreasing seniority: no matrix, dotted-line or same-level reporting. A
  wrong seniority parse can exclude the true manager from the candidates; 12%
  of train employees whose manager is observed lose their true manager at
  shortlisting.
* **Approximate calibration for non-top candidates.** Calibration is fitted on
  the top candidate, so the other candidates' probabilities are approximate.
* **Large org charts.** The dashboard draws at most 450 nodes per chart. Use
  the "around an employee" or department views for large companies.

## Project structure

```
configs/generation.yaml        generation parameters
src/orgchart/
  ladder.py config.py schemas.py paths.py cli.py
  storage/                     parquet_io, observed (inference-safe), truth (evaluation only)
  generation/                  synthetic companies, noise, validation (Phase 1)
  normalization/               title lexicon, cleaning, parsing, company context (Phase 2)
  inference/                   location, data, features, scoring, decide, naive, pipeline, validate (observed only)
  learning/                    estimators (logit, conditional logit, isotonic), fit (train + calibration only)
  evaluation/                  labels, normalization_eval, inference_eval (truth side)
app/                           streamlit_app.py, app_data.py, chart.py, ask.py + ask_view.py (explore),
                               evaluation_view.py (evaluation)
tests/                         unit, property-based, end-to-end, isolation, CLI, dashboard (243 tests)
docs/                          architecture, data generation, title normalisation, inference, data dictionary, demo
```
