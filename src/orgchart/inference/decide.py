"""Turn scored candidate pairs into one reporting-line decision per employee.

Decision cascade, applied to each eligible employee e:

1. **DIRECT** when the top candidate's calibrated direct probability is at
   least `thresholds.direct`.
2. **AMBIGUOUS** when the smallest set of 2 to `max_ambiguous` top candidates
   whose listwise probability mass reaches `thresholds.ambiguous` exists. The
   true manager is probably in that set.
3. **ANCESTOR** when some non-CEO candidate is very likely above e in the chain
   (calibrated p_ancestor >= `thresholds.ancestor`). The lowest-level such
   candidate is reported, because it is the most informative. This is how the
   system handles a direct manager who is missing from the observed data. The
   CEO is excluded because "reports up to the CEO" is true of everyone and
   carries no information.
4. **UNRESOLVED** otherwise, with a reason code.

The cascade never forces an answer. Edges only point to strictly more senior
people in the same company, so DIRECT edges are acyclic and each employee has
at most one direct manager.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from orgchart.inference.features import EMPLOYEE_LEVEL_FEATURES, FEATURE_LABELS, FEATURES
from orgchart.inference.scoring import Model

STATUSES = ("DIRECT", "ANCESTOR", "AMBIGUOUS", "UNRESOLVED")
TOP_CANDIDATES_KEPT = 5
_PAIR_MASK = np.array([f not in EMPLOYEE_LEVEL_FEATURES for f in FEATURES], dtype=np.float64)


def top_contributions(rows: pd.DataFrame, weights: np.ndarray, k: int = 4) -> list[list[str]]:
    """Human-readable largest per-feature contributions (weight x value) for each row."""
    if len(rows) == 0:
        return []
    contrib = rows[list(FEATURES)].to_numpy(np.float64) * (weights * _PAIR_MASK)[None, :]
    order = np.argsort(-np.abs(contrib), axis=1, kind="stable")[:, :k]
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    return [[f"{contrib[r, i]:+.2f} {labels[i]}" for i in idx if abs(contrib[r, i]) > 1e-9]
            for r, idx in enumerate(order)]


def _wide(ranked: pd.DataFrame, column: str, width: int, fill) -> pd.DataFrame:
    top = ranked[ranked["rank"] < width]
    wide = top.pivot(index="employee_id", columns="rank", values=column).reindex(columns=range(width))
    return wide if fill is None else wide.fillna(fill)


def decide(people: pd.DataFrame, ranked: pd.DataFrame, model: Model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`ranked` comes from `Model.score_pairs`. Returns (one prediction per profile, top candidates kept)."""
    th = model.thresholds
    width = max(2, th.max_ambiguous)
    title = dict(zip(people["employee_id"], people["title_raw"].fillna("(no title)"), strict=True))

    q = _wide(ranked, "q_direct", width, 0.0)
    pos = {eid: i for i, eid in enumerate(q.index)}
    q_arr = q.to_numpy(np.float64)
    p_arr = _wide(ranked, "p_direct", width, 0.0).to_numpy(np.float64)
    c_arr = _wide(ranked, "candidate_id", width, None).to_numpy(object)
    c_arr[pd.isna(c_arr)] = None
    n_cand = ranked.groupby("employee_id").size().to_dict()

    top = ranked[ranked["rank"] == 0]
    top_expl = dict(zip(top["employee_id"], top_contributions(top, model.direct.vector()), strict=True))

    anc = ranked[(~ranked["m_is_ceo"]) & (ranked["p_ancestor"] >= th.ancestor)]
    anc = anc.sort_values(["employee_id", "m_level", "p_ancestor", "candidate_id"],
                          ascending=[True, True, False, True], kind="mergesort").drop_duplicates("employee_id")
    anc_expl = dict(zip(anc["employee_id"], top_contributions(anc, model.ancestor.vector()), strict=True))
    anc_pick = dict(zip(anc["employee_id"], zip(anc["candidate_id"], anc["p_ancestor"], strict=True), strict=True))

    rows = []
    for emp in people.itertuples(index=False):
        eid = emp.employee_id
        rec = {"employee_id": eid, "company_id": emp.company_id, "status": "UNRESOLVED", "reports_to_id": None,
               "ancestor_id": None, "candidate_ids": [], "confidence": None, "top_candidate_id": None, "direct_confidence": None,
               "runner_up_id": None, "runner_up_confidence": None, "n_candidates": int(n_cand.get(eid, 0)),
               "reason_codes": [], "explanation": []}
        expl, codes = rec["explanation"], rec["reason_codes"]
        rows.append(rec)
        if emp.is_non_employee:
            codes.append("NON_EMPLOYEE")
            expl.append("board member / advisor profile: not part of the reporting tree")
            continue
        if pd.isna(emp.level):
            codes.append("LEVEL_UNRESOLVED")
            expl.append("seniority could not be read from the title, so no manager can be placed")
            continue
        if eid not in pos:
            codes.append("TOP_OF_ORG" if emp.level >= 11 else "NO_CANDIDATES")
            expl.append("no more senior person with management authority is observed in this company"
                        + (" (top of the organisation)" if emp.level >= 11 else ""))
            continue

        i = pos[eid]
        qs, ps, cs = q_arr[i], p_arr[i], c_arr[i]
        rec["top_candidate_id"], rec["direct_confidence"] = cs[0], round(float(ps[0]), 4)
        if cs[1] is not None:
            rec["runner_up_id"], rec["runner_up_confidence"] = cs[1], round(float(ps[1]), 4)
        margin = float(qs[0] - qs[1])
        cum = np.cumsum(qs)
        amb_size = next((r + 1 for r in range(1, th.max_ambiguous)
                         if cs[r] is not None and cum[r] >= th.ambiguous), None)

        if ps[0] >= th.direct:
            rec.update(status="DIRECT", reports_to_id=cs[0], confidence=round(float(ps[0]), 4))
            codes.append("CONFIDENT_DIRECT")
            if margin < 0.2:
                codes.append("CLOSE_RUNNER_UP")
            expl.append(f"reports to {title[cs[0]]} (p={ps[0]:.2f})")
            expl += top_expl[eid]
            if cs[1] is not None:
                expl.append(f"runner-up: {title[cs[1]]} (p={ps[1]:.2f}); probability margin {margin:.2f}")
        elif amb_size:
            ids = list(cs[:amb_size])
            rec.update(status="AMBIGUOUS", candidate_ids=ids, confidence=round(float(cum[amb_size - 1]), 4))
            codes.append("TIED_CANDIDATES" if margin < 0.1 else "LOW_MARGIN")
            expl.append(f"manager is probably one of {amb_size} candidates (combined p={cum[amb_size - 1]:.2f}): "
                        + "; ".join(f"{title[x]} ({p:.2f})" for x, p in zip(ids, qs[:amb_size], strict=True)))
        elif eid in anc_pick:
            a_id, a_p = anc_pick[eid]
            rec.update(status="ANCESTOR", ancestor_id=a_id, confidence=round(float(a_p), 4))
            codes.append("DIRECT_MANAGER_UNCERTAIN")
            if cum[-1] < 0.5 and qs[0] < 0.3:
                codes.append("LIKELY_MANAGER_UNOBSERVED")
            expl.append(f"sits under {title[a_id]} (p={a_p:.2f}); direct manager not identified "
                        f"(best direct p={ps[0]:.2f})")
            expl += anc_expl[eid]
        else:
            codes.append("INSUFFICIENT_EVIDENCE")
            expl.append(f"best direct candidate {title[cs[0]]} has p={ps[0]:.2f}, below the {th.direct:.2f} "
                        f"threshold, and no confident ancestor")

    predictions = pd.DataFrame(rows)
    keep = ranked[ranked["rank"] < TOP_CANDIDATES_KEPT]
    if len(anc):
        extra = ranked.merge(anc[["employee_id", "candidate_id"]], on=["employee_id", "candidate_id"])
        keep = pd.concat([keep, extra]).drop_duplicates(["employee_id", "candidate_id"])
    return predictions, keep.reset_index(drop=True)
