# Synthetic data generation

The generator builds each company in five stages. Every stage is driven by the
company's own seed, so results don't depend on parallelism.

## 1. Planning (`generator.plan_companies`)

The global seed determines, for each company:

* the industry archetype;
* the HQ country and city;
* a target **observed** size, drawn log-normally (median 260, clipped to [100, 3000]);
* a coverage rate, drawn from Beta(mean 0.86) and clipped to [0.70, 0.97];
* the founding year;
* a unique name;
* its own seed.

The true size is `ceil(observed_target / coverage)`. Splits are then assigned,
stratified by industry × size band.

## 2. True tree (`structure.py`)

The tree is built top down.

**Executive office.** The CEO comes first. Companies with ≥ 150 people may
also have an Executive Assistant, and companies with ≥ 400 a Chief of Staff.

**Departments.**
* Headcount is allocated using the archetype's department weights, perturbed by a Dirichlet draw.
* Core departments are always present. Optional ones (Legal, Design, Data, ...) become more likely as company size grows.

**Department head level** depends on department size:
* C-level is possible if the department has a C-title. The probability scales with company size (e.g. a CTO is common even in a 150-person SaaS company).
* Otherwise: SVP (≥ 250 people), VP (≥ 120), Director or VP (≥ 40), Director or Senior Manager (≥ 12), and Manager, Senior Manager or Director for small teams.

**Subtree growth.** Each manager receives a headcount budget and splits it:
* *Frontline teams* are sized like teams. Department-specific caps apply: 9 by default, up to 18 for retail stores and clinical units, 14 for production and warehouses.
* *Senior managers* have spans of 4–8. Their child manager levels come from the size of each child's subtree, with a 15% chance of a skip-level.
* Directors and above occasionally have staff ICs (Principal/Staff) reporting to them directly.
* Fragments of 1–2 people report directly instead of becoming a "team".

**Specialties and scopes.**
* Child teams get specialties (Backend, Payments, FP&A, Nursing, ...).
* In regional or divisional companies, scopable departments split by region (ANZ/EMEA/... or states) or by division at the head level.

**Ownership.** Some heads report to another head instead of the CEO, for
example Legal → CFO, Design → CPO, IT → COO. Only one level of nesting is
allowed, and the owner must be more senior.

**Other people.**
* Executive assistants for C-levels in companies of ≥ 1,000.
* Contractors (4%) and interns.
* 0–4 detached **board/advisor profiles**. These have no manager in truth; the correct inference for them is "no manager".

The tree guarantees one CEO root, no cycles, and **strictly decreasing levels
along every edge**.

## 3. Canonical titles (`titles.py`, `catalog.py`)

Titles come from templates per department, level and track (IC, manager or
department head), with specialty overrides (e.g. Nursing: Enrolled Nurse →
Clinical Nurse → Nurse Unit Manager → Director of Nursing).

The vocabulary deliberately includes the ambiguity found in real data:

* IC titles containing "Manager": Product Manager, Customer Success Manager, Account Manager, Program Manager.
* "Lead" on both tracks: Lead Accountant (IC) vs Team Lead (manager).
* Bare consulting titles: Manager, Director, Partner.
* "Director" at different levels: Managing Director (CEO), Finance Director (L9), Director of Engineering (L7).
* Colliding acronyms: CRO (Revenue vs Risk) and CPO (Product vs People).
* "Head of X" spanning levels 5–9.
* "General Manager, X".

## 4. People (`people.py`)

**Offices and locations.** Companies have offices in their HQ country and
abroad. Teams sit in an office; regional teams sit in their region's office.
About 5% of people are remote.

**Names** come from Faker locales mixed per country.

**Start dates** are derived from a tenure distribution that grows with level.
They are a weak signal, not a hierarchy leak.

**IDs** are random 48-bit hex, so they carry no ordering information.

## 5. Observed view (`noise.py`)

**Missing employees.** A person's chance of being missing scales with their
level: middle management is about 1.4× more likely to be missing than ICs,
and executives about 0.3×. The CEO is observed 97% of the time. Rates are
renormalised to the company's coverage. If fewer than 100 employees remain
observed, dropped employees are restored at random until the guarantee holds.
Truth records `is_observed` and `nearest_observed_manager_id`.

**Title noise.** Each transform that fires is recorded in `truth.noise_tags`.

| Transform | Example |
|---|---|
| stale title (previous level) | Senior Engineer shown as "Software Engineer" |
| inflated title (small companies) | Manager shown as "Head of Marketing" |
| term abbreviations | Vice President → VP, Account Executive → AE, Operations → Ops |
| seniority abbreviations | Senior → Sr./Sr, Junior → Jr. |
| expansions | VP → Vice President |
| qualifier drop / separator / parentheses | "Engineering Manager, Payments" → "Engineering Manager" / "… \| Payments" / "… (Payments)" |
| contractor markers | "(Contract)", "Contract …" |
| decorations | "@ Acme", "\| Hiring" |
| case | lower / UPPER |
| vague / empty | "Manager", "Team Member", missing |
| **test-only** abbreviations | Mgr, Snr, SWE, Engr, Coord. |
| **test-only** reorder | "Payments Engineering Manager", "Engineering Director" |
| **test-only** separators | " / ", " :: " |
| **test-only** typos | "Slaes Manager" |

**Department field.** Missing 30% of the time; mislabelled 3%; otherwise one
of several label variants (Engineering / R&D / Technology / Product &
Engineering).

**Location.** Formatted as "City, State, Country", "Greater X Area", a country
alone, or a remote variant. Missing 5% of the time.

**Start date.** Missing 20% of the time.

## Validation

`orgchart validate` runs 16 hard checks, most of them in DuckDB SQL:
* unique IDs;
* observed rows consistent with the truth flags;
* ≥ 100 observed employees per company;
* a single CEO root;
* no dangling managers;
* acyclic and connected (recursive CTE);
* strictly decreasing levels;
* consistent report counts;
* correct nearest-observed-manager;
* detached board profiles;
* no truth columns in observed schemas;
* proportional splits;
* diversity per split (every industry in every split, all three structure types present; datasets of 100+ companies);
* holdout noise only in test companies;
* manifest hashes.

It also reports realism statistics against `realism_targets`: span of control,
manager ratio, depth, coverage and title diversity.

## Known limitations

* Each person has exactly one manager. There are no matrix or dotted-line
  relationships, no co-CEOs and no shared executive assistants.
* Levels strictly decrease along every edge. Real orgs occasionally have
  same-level reporting (e.g. a Senior Manager under a Senior Manager acting up).
* Titles are English-only and drawn from a finite template set. Real titles
  have a much longer tail (the test-only noise styles partly address this).
* The data has no duplicate profiles, career histories, or profiles of former
  employees that are still marked current.
* Realised coverage varies around the target for small companies because of
  sampling (observed range 63–98%).
