# Demo walkthrough (about 10 minutes)

All commands run from the repository root. They assume the dataset and models
are already built. If they aren't, run the full build first (about 105 s):

```bash
.venv/bin/orgchart pipeline data/full --generate
```

## 1. Trust the build (1 min)

```bash
make verify        # or, without make: see the Makefile's verify target
```

This runs:

* the test suite (243 tests);
* the 16 dataset structural checks;
* the prediction invariants: one manager at most, no self-reporting, no
  cycles, same company, more senior manager, board members unassigned;
* the delivery checks: 600 companies, at least 100 observed employees each,
  and inference that is deterministic on re-run.

It exits non-zero on any failure.

## 2. Show the honest headline (2 min)

```bash
.venv/bin/orgchart evaluate data/full
```

Start by defining the claims, because they are scored differently:

* **DIRECT** names the exact manager. A skip-level answer counts as wrong.
* **ANCESTOR** names someone somewhere above the employee. It is **not** a
  manager claim.
* **AMBIGUOUS** names 2–3 candidates, one of whom is the manager.

Then point out four things:

* **Exact managers.** Always guessing (naive) gives a DIRECT for everyone and
  is right 19.7% of the time. At the 95% target, the selected model gives a
  DIRECT to only 0.87% of test employees, but it is right 94.1% of the time.
* **Ancestors.** For a further 39.8% of employees it names someone confidently
  above them (95.4% precision), without claiming a manager.
* **The test split was held out.** Thresholds were chosen on calibration
  companies; test was scored once.
* **Calibration holds.** Top-candidate ECE is 1.1%, so a "0.8" means about
  80%.

Then open `data/full/reports/inference_eval.md` and show the operating-points
table. At the 80% target, DIRECT covers 10.6% of employees at 81.2% precision,
and ANCESTOR covers 59.2% at 80.8%.

## 3. Dashboard (5 min)

```bash
make app           # or: .venv/bin/streamlit run app/streamlit_app.py
```

1. **Pick a company.** In the sidebar, choose **C0150 · Tidalwave HQ Limited**,
   a SaaS company in the test split. All 600 companies are available, with an
   industry filter and a cross-company employee search.
2. **Status cards.** They read "DIRECT · exact manager",
   "ANCESTOR · in chain above", and so on. Hover a card for the exact
   meaning.
3. **Org chart → "Managers and leaders"** (the default view).
   * Each card is one predicted branch, drawn top-down. Branches are separate
     because no confident claim links their top people.
   * Solid green edges are DIRECT, labelled with their calibrated probability.
   * Dashed blue edges labelled "ancestor relationship" are ANCESTOR: the
     upper person is somewhere above, possibly the direct manager, possibly
     higher. They skip a level and lead to lists headed "N employees linked
     through hierarchy: direct manager not identified".
   * Dotted orange "?" edges point to AMBIGUOUS candidates.
   * At 80%, the CEO's branch continues on "Team of …" cards.
   * Switch View to **"Around an employee"** and use **"Go to a nearby
     person"** to walk up the chain or across to peers.
4. **Ask the Org Chart tab.** Click the three example buttons, then type:
   * `Who does E0da949b316a4 report to?`: a DIRECT answer naming the Sales
     Director with p = 0.97.
   * `Who does E0298018f2451 report to?`: "direct manager is **not
     confidently identified**". The Head of Marketing is listed as ANCESTOR,
     in the chain above and possibly the direct manager or someone higher.
   * `Who is above E0da949b316a4?`: the chain of claims upwards, each step
     marked DIRECT or ANCESTOR with its own probability.
   * `Who runs the Marketing team?`: the **inferred** leader (most senior
     job title in the department), stated as an inference from titles, not
     a reporting claim.
   * `Who does James report to?` lists the four people called James, and an
     unsupported question lists the supported forms. Answers follow the
     sidebar's precision target, and no language model is involved.
5. **Precision targets tab.** This has the 95 / 90 / 80% comparison, with
   DIRECT, ANCESTOR and AMBIGUOUS in separate tables. The red note explains
   why ANCESTOR precision is not manager accuracy.
6. **Switch the sidebar's "Precision target"** from 95% to 80%. DIRECT for
   this company rises from 10 to 56, and the sidebar shows the measured
   held-out DIRECT and ANCESTOR results for 80%. Choose "Custom DIRECT
   threshold" for a free slider.
7. **Employees tab.** Switch back to 95%. The exact-manager, "somewhere
   above (direct manager not identified)" and candidate-set columns are separate. Search
   "Sales Manager" and select the row (`E0da949b316a4`). The panel shows:
   * a green box: "DIRECT: exact direct manager", *SALES DIRECTOR*, p = 0.97;
   * the explanation bullets;
   * the ranked candidates table with p(direct) and p(above in chain);
   * the per-feature contribution chart.
8. **Ambiguous example.** "CSM" (`E17704045aeef`) shows an orange box: "the
   direct manager is one of 2" (combined p = 0.94).
9. **Ancestor example.** "Senior Product Marketing Manager" (`E0298018f2451`)
   shows a blue box: "ANCESTOR: in the reporting chain; direct manager not
   identified". Head of Marketing is somewhere above (p = 0.97), possibly as
   the direct manager, possibly higher. Below it is the best direct
   candidate and why it was not claimed.
10. **Review queue tab.** Lists employees without an exact manager (AMBIGUOUS
   and UNRESOLVED by default; ANCESTOR optional) with reason codes. These are
   the hand-off to a human reviewer or another data source.
11. **Model tab.** Shows the thresholds, the selection rule, and every feature
   weight. Every score is a sum of these weights times feature values.
12. **Switch Mode to "Evaluation (ground truth)".** This is explicitly
   separated and is the only part of the app that reads truth.
   * **This company vs truth:** separate DIRECT and ANCESTOR columns, the
     true manager, and how many levels above each ancestor really is.
   * **Held-out test results:** the baseline comparison, operating points,
     the precision–coverage curve, the reliability diagram, and breakdowns by
     industry, size, seniority and structure.

## 4. One explanation from the CLI (1 min)

```bash
.venv/bin/orgchart explain E0da949b316a4 --dataset data/full
.venv/bin/orgchart parse-title "Snr SWE II" --department Engineering
```

## Talking points

* **Ground-truth isolation is enforced by tests, not by convention.** The
  inference package cannot import truth, evaluation or model-fitting code, and
  the fitting code cannot name the test split.
* **Precision over recall.** The system never forces an edge.
  * When the direct manager is probably missing from the data, it reports a
    confident ancestor instead.
  * When the data supports several managers equally, it reports them as
    ambiguous.
* **Learned scoring only where justified.** The learned linear model was
  adopted because it measurably beat the hand-set rules on calibration
  companies: +46 pp any-claim coverage, mostly ANCESTOR, and +0.5 pp DIRECT.
  It keeps the same features and additive explanations.
