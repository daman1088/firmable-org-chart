"""Candidate generation and pairwise features for one company.

For every eligible employee e the candidate managers m are the observed people
in the same company who:

* hold people-management authority (manager or executive track);
* are strictly more senior than e (normalised level).

The strict level order makes any set of predicted edges acyclic by
construction. A cheap prefilter keeps a shortlist per employee: the top
`k_direct` by a direct-fit score plus the top `k_ancestor` by an
ancestor-fit score. Interpretable features are then computed for every
shortlisted pair. All inputs are observed or normalised fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from orgchart.inference.location import parse_location

FEATURES: tuple[str, ...] = (
    "ic_gap1", "ic_gap2", "ic_gap3", "ic_gap4p",
    "mg_gap1", "mg_gap2", "mg_gap3p",
    "log_between_dept",
    "log_between_strict", "log_between_levels",
    "dept_match", "dept_mismatch",
    "m_exec", "e_top_x_m_exec", "e_top_x_dept_mismatch",
    "spec_match", "spec_mismatch",
    "scope_match", "scope_mismatch",
    "same_city", "diff_city", "diff_country",
    "log_equiv",
    "m_ceo", "m_lead",
    "m_mgmt_conf_low", "e_level_conf_low",
    "start_before", "start_after",
    "e_support_x_m_exec",
    "m_frontline", "m_has_spec", "spec_match_x_frontline", "same_city_x_frontline",
    "title_overlap", "title_overlap_x_frontline",
    "e_top", "e_is_manager", "log_n_candidates", "log_between_unknown",
)

FEATURE_LABELS: dict[str, str] = {
    "ic_gap1": "IC one level below candidate",
    "ic_gap2": "IC two levels below candidate",
    "ic_gap3": "IC three levels below candidate",
    "ic_gap4p": "IC four or more levels below candidate (skip-level)",
    "mg_gap1": "manager one level below candidate",
    "mg_gap2": "manager two levels below candidate",
    "mg_gap3p": "manager three or more levels below candidate",
    "log_between_dept": "same-department managers sit between them",
    "log_between_strict": "closer-matching managers (team, scope, office) sit between them",
    "log_between_levels": "closer-matching managers occupy levels between them",
    "dept_match": "same department",
    "dept_mismatch": "different department",
    "m_exec": "candidate is an executive",
    "e_top_x_m_exec": "employee heads their function; candidate is an executive",
    "e_top_x_dept_mismatch": "employee heads their function; cross-function owner",
    "spec_match": "same specialty / team",
    "spec_mismatch": "different specialty / team",
    "scope_match": "same region / unit",
    "scope_mismatch": "different region / unit",
    "same_city": "same office city",
    "diff_city": "different city, same country",
    "diff_country": "different country",
    "log_equiv": "other equally plausible candidates at the same level",
    "m_ceo": "candidate is the CEO",
    "m_lead": "candidate is a team lead (level 4)",
    "m_mgmt_conf_low": "candidate's management authority is uncertain",
    "e_level_conf_low": "employee's seniority is uncertain",
    "start_before": "candidate joined before the employee",
    "start_after": "candidate joined after the employee",
    "e_support_x_m_exec": "executive support role; candidate is an executive",
    "m_frontline": "candidate is a frontline / middle manager (level 4-6)",
    "m_has_spec": "candidate's title names a team / specialty",
    "spec_match_x_frontline": "same team, frontline candidate",
    "same_city_x_frontline": "same office, frontline candidate",
    "title_overlap": "titles share role vocabulary",
    "title_overlap_x_frontline": "titles share role vocabulary, frontline candidate",
    "e_top": "employee heads their function",
    "e_is_manager": "employee is a people manager",
    "log_n_candidates": "number of plausible managers in the company",
    "log_between_unknown": "managers with unreadable department sit at levels between them",
}

# Words that say how senior or managerial a title is, not what the work is.
TITLE_STOPWORDS = frozenset("""
    senior junior graduate intern internship trainee assistant associate principal staff lead leader head chief
    manager management director vice president vp svp evp officer executive general group team of the and & for
    in to at i ii iii iv deputy acting interim regional national global area district country
""".split())

# Properties of the employee alone: identical for all of an employee's candidates, so they move probability
# between "some candidate" and "manager not observed" but never reorder candidates. Explanations of why one
# candidate beats another leave them out.
EMPLOYEE_LEVEL_FEATURES = frozenset({"e_top", "e_is_manager", "log_n_candidates", "e_level_conf_low"})

# Cheap prefilter weights (shortlisting only; final scores come from the scorer).
_DIRECT_PREFILTER = {"dept_match": 3.0, "dept_mismatch": -4.0, "spec_match": 1.5, "spec_mismatch": -1.5,
                     "scope_match": 2.0, "scope_mismatch": -3.0, "same_city": 1.0, "diff_country": -1.0}
_ANCESTOR_PREFILTER = {"dept_match": 3.0, "dept_mismatch": -4.0, "scope_match": 2.0, "scope_mismatch": -3.0,
                       "spec_mismatch": -1.0}


@dataclass(frozen=True)
class CandidateSettings:
    k_direct: int = 20
    k_ancestor: int = 5


EXECUTIVE_DEPT = "executive"
N_LEVELS = 13  # 0 .. 12


def company_people(frame: pd.DataFrame) -> pd.DataFrame:
    """Add eligibility flags and parsed locations to one company's normalised rows."""
    df = frame.reset_index(drop=True).copy()
    level = df["level"].astype("float").fillna(-1).astype(int).to_numpy()
    df["lvl"] = level
    non_emp = df["is_non_employee"].to_numpy(bool)
    manages = df["manages_people"].fillna(False).to_numpy(bool)
    df["is_exec"] = (df["track"].to_numpy() == "executive") | (level >= 11)
    df["eligible_employee"] = ~non_emp & (level >= 0)
    df["eligible_candidate"] = ~non_emp & (level >= 0) & (manages | df["is_exec"].to_numpy())
    locs = [parse_location(x) for x in df["location"]]
    df["city"] = [loc.city for loc in locs]
    df["country"] = [loc.country for loc in locs]
    return df


def _codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values, use_na_sentinel=True)[0].astype(np.int32)


def _states(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0 = unknown on either side, 1 = match, 2 = mismatch."""
    known = (a >= 0) & (b >= 0)
    match = known & (a == b)
    return match, known & ~match, np.where(known, np.where(match, 1, 2), 0)


def build_candidates(people: pd.DataFrame, settings: CandidateSettings = CandidateSettings()) -> pd.DataFrame:
    """Shortlisted (employee, candidate) pairs with feature columns for one company."""
    e_idx = np.flatnonzero(people["eligible_employee"].to_numpy())
    c_idx = np.flatnonzero(people["eligible_candidate"].to_numpy())
    if len(e_idx) == 0 or len(c_idx) == 0:
        return pd.DataFrame(columns=["employee_id", "candidate_id", *FEATURES])

    lvl = people["lvl"].to_numpy()
    dept = _codes(people["department"])
    spec = _codes(people["specialty"])
    scope = _codes(people["scope"])
    city = _codes(people["city"])
    country = _codes(people["country"])
    exec_code = (people["department"] == EXECUTIVE_DEPT).to_numpy()
    is_exec = people["is_exec"].to_numpy(bool)
    manages = people["manages_people"].fillna(False).to_numpy(bool)
    track = people["track"].to_numpy()
    start = pd.to_datetime(people["start_date"]).to_numpy("datetime64[D]")
    start_known = ~np.isnat(start)

    Le, Lc = lvl[e_idx][:, None], lvl[c_idx][None, :]
    valid = Lc > Le
    d_e, d_c = dept[e_idx][:, None], dept[c_idx][None, :]
    dept_match, dept_mis, _ = _states(d_e, d_c)
    dept_mis &= ~exec_code[c_idx][None, :]  # the executive office is compatible with every function
    spec_match, spec_mis, spec_state = _states(spec[e_idx][:, None], spec[c_idx][None, :])
    scope_match, scope_mis, scope_state = _states(scope[e_idx][:, None], scope[c_idx][None, :])
    city_match, city_mis, city_state = _states(city[e_idx][:, None], city[c_idx][None, :])
    country_match, country_mis, _ = _states(country[e_idx][:, None], country[c_idx][None, :])

    # --- shortlist -------------------------------------------------------------------------------
    gap = (Lc - Le).astype(np.float32)
    mats = {"dept_match": dept_match, "dept_mismatch": dept_mis, "spec_match": spec_match, "spec_mismatch": spec_mis,
            "scope_match": scope_match, "scope_mismatch": scope_mis, "same_city": city_match,
            "diff_country": country_mis}
    head_like = (Le >= 7) & is_exec[c_idx][None, :]
    direct_pre = sum(w * mats[k] for k, w in _DIRECT_PREFILTER.items()) - 0.5 * gap + 2.0 * head_like
    ancestor_pre = sum(w * mats[k] for k, w in _ANCESTOR_PREFILTER.items()) + 0.3 * Lc + 2.0 * head_like
    keep = np.zeros_like(valid)
    for score, k in ((direct_pre, settings.k_direct), (ancestor_pre, settings.k_ancestor)):
        score = np.where(valid, score, -np.inf)
        k = min(k, score.shape[1])
        top = np.argpartition(-score, k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(len(e_idx)), k)
        keep[rows, top.ravel()] = True
    keep &= valid
    pi, pj = np.nonzero(keep)
    if len(pi) == 0:
        return pd.DataFrame(columns=["employee_id", "candidate_id", *FEATURES])
    ei, ci = e_idx[pi], c_idx[pj]
    le, lm = lvl[ei], lvl[ci]

    # --- shared role vocabulary (Jaccard over content words of the cleaned titles) ---------------
    vocab: dict[str, int] = {}
    token_rows = [[vocab.setdefault(t, len(vocab)) for t in set(str(x).replace(",", " ").split())
                   if t not in TITLE_STOPWORDS and len(t) > 1] if isinstance(x, str) else []
                  for x in people["title_clean"]]
    T = np.zeros((len(people), max(1, len(vocab))), dtype=np.float32)
    for r, toks in enumerate(token_rows):
        T[r, toks] = 1.0
    sizes = T.sum(axis=1)
    inter = np.einsum("pv,pv->p", T[ei], T[ci])
    union = sizes[ei] + sizes[ci] - inter
    title_overlap = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    # --- managers sitting between e and m --------------------------------------------------------
    # cc[d + 1, l] = number of candidates in department d with level < l (d = -1: unknown department)
    cc = np.zeros((dept.max() + 2, N_LEVELS + 1), dtype=np.int32)
    np.add.at(cc, (dept[c_idx] + 1, lvl[c_idx] + 1), 1)
    cc = np.cumsum(cc, axis=1)
    d_ref = np.where(dept[ei] >= 0, dept[ei], dept[ci]) + 1
    n_between_dept = cc[d_ref, lm] - cc[d_ref, le + 1]
    e_top = (dept[ei] >= 0) & (cc[dept[ei] + 1, N_LEVELS] - cc[dept[ei] + 1, le + 1] == 0)
    n_between_unknown = np.where(dept[ei] >= 0, cc[0, lm] - cc[0, le + 1], 0)  # row 0 = unknown department

    def compat(a: np.ndarray) -> np.ndarray:
        return (a[e_idx][:, None] < 0) | (a[c_idx][None, :] < 0) | (a[e_idx][:, None] == a[c_idx][None, :])

    same_dept_ref = np.where(d_e >= 0, d_c == d_e, False)
    strict = valid & same_dept_ref & compat(spec) & compat(scope) & compat(city)
    hist = np.zeros((len(e_idx), N_LEVELS + 1), dtype=np.int32)
    occupied = np.zeros_like(hist)
    for level in range(N_LEVELS):
        at_level = (strict & (Lc == level)).sum(axis=1)
        hist[:, level + 1] = hist[:, level] + at_level
        occupied[:, level + 1] = occupied[:, level] + (at_level > 0)
    n_between_strict = hist[pi, lm]
    n_between_levels = occupied[pi, lm]

    # --- equally plausible competitors (same level, department and match pattern) ---------------
    sig = spec_state * 9 + scope_state * 3 + city_state
    key = ((Lc.astype(np.int64) * 1024 + (d_c + 1)) * 27 + sig) + np.arange(len(e_idx))[:, None].astype(np.int64) * (1 << 32)
    valid_keys = key[valid]
    uniq, counts = np.unique(valid_keys, return_counts=True)
    n_equiv = counts[np.searchsorted(uniq, key[pi, pj])]

    e_manager = manages[ei] & (track[ei] != "support")
    frontline = (lm >= 4) & (lm <= 6) & ~is_exec[ci]
    g = lm - le
    out = {
        "employee_id": people["employee_id"].to_numpy()[ei],
        "candidate_id": people["employee_id"].to_numpy()[ci],
        "e_level": le.astype(np.int8),
        "m_level": lm.astype(np.int8),
        "m_is_ceo": lm == 12,
        "n_between_dept": n_between_dept.astype(np.int16),
        "n_between_strict": n_between_strict.astype(np.int16),
        "n_equiv": n_equiv.astype(np.int16),
        "prefilter_direct": direct_pre[pi, pj].astype(np.float32),
    }
    f = {
        "ic_gap1": ~e_manager & (g == 1), "ic_gap2": ~e_manager & (g == 2), "ic_gap3": ~e_manager & (g == 3),
        "ic_gap4p": ~e_manager & (g >= 4),
        "mg_gap1": e_manager & (g == 1), "mg_gap2": e_manager & (g == 2), "mg_gap3p": e_manager & (g >= 3),
        "log_between_dept": np.log1p(n_between_dept),
        "log_between_strict": np.log1p(n_between_strict),
        "log_between_levels": np.log1p(n_between_levels),
        "dept_match": dept_match[pi, pj], "dept_mismatch": dept_mis[pi, pj],
        "m_exec": is_exec[ci], "e_top_x_m_exec": e_top & is_exec[ci], "e_top_x_dept_mismatch": e_top & dept_mis[pi, pj],
        "spec_match": spec_match[pi, pj], "spec_mismatch": spec_mis[pi, pj],
        "scope_match": scope_match[pi, pj], "scope_mismatch": scope_mis[pi, pj],
        "same_city": city_match[pi, pj], "diff_city": city_mis[pi, pj] & ~country_mis[pi, pj],
        "diff_country": country_mis[pi, pj],
        "log_equiv": np.log1p(n_equiv - 1),
        "m_ceo": lm == 12, "m_lead": lm == 4,
        "m_mgmt_conf_low": people["management_confidence"].to_numpy()[ci] < 0.6,
        "e_level_conf_low": people["level_confidence"].to_numpy()[ei] < 0.6,
        "start_before": start_known[ei] & start_known[ci] & (start[ci] <= start[ei]),
        "start_after": start_known[ei] & start_known[ci] & (start[ci] > start[ei]),
        "e_support_x_m_exec": (track[ei] == "support") & is_exec[ci],
        "m_frontline": frontline, "m_has_spec": spec[ci] >= 0,
        "spec_match_x_frontline": spec_match[pi, pj] & frontline, "same_city_x_frontline": city_match[pi, pj] & frontline,
        "title_overlap": title_overlap, "title_overlap_x_frontline": title_overlap * frontline,
        "e_top": e_top, "e_is_manager": e_manager, "log_n_candidates": np.log1p(valid.sum(axis=1)[pi]),
        "log_between_unknown": np.log1p(n_between_unknown),
    }
    for name in FEATURES:
        out[name] = np.asarray(f[name], dtype=np.float32)
    return pd.DataFrame(out)
