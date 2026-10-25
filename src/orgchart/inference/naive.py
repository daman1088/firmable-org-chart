"""Naive comparison baseline: always guess, never abstain.

Each employee is assigned to the closest more senior person with management
authority in the same department. Ties go to the same office city, then to
the lowest employee ID. If the department has no such person, the employee is
assigned to the most senior person in the company. Every eligible employee
gets a DIRECT edge with no confidence. This is what forcing a prediction
looks like, and it is the reference the abstaining system must beat on
precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_predictions(people: pd.DataFrame) -> pd.DataFrame:
    """`people` is one company's frame from `features.company_people`."""
    lvl = people["lvl"].to_numpy()
    dept = people["department"].to_numpy(object)
    city = people["city"].to_numpy(object)
    ids = people["employee_id"].to_numpy(object)
    cand = np.flatnonzero(people["eligible_candidate"].to_numpy())
    rows = []
    for i in range(len(people)):
        rec = {"employee_id": ids[i], "company_id": people["company_id"].iat[i], "status": "UNRESOLVED",
               "reports_to_id": None, "reason_codes": ["NAIVE"]}
        if people["eligible_employee"].iat[i]:
            above = cand[lvl[cand] > lvl[i]]
            if len(above):
                same = above[dept[above] == dept[i]] if dept[i] is not None else above[:0]
                if len(same):
                    key = sorted(same, key=lambda j: (lvl[j], city[j] != city[i] or city[i] is None, ids[j]))
                else:
                    key = sorted(above, key=lambda j: (-lvl[j], ids[j]))
                rec.update(status="DIRECT", reports_to_id=ids[key[0]])
        rows.append(rec)
    return pd.DataFrame(rows)
