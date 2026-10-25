"""Small, dependency-free estimators: L2 logistic regression (Newton) and isotonic regression (PAV)."""

from __future__ import annotations

import numpy as np


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 25, tol: float = 1e-6,
                 chunk: int = 500_000) -> tuple[float, np.ndarray]:
    """Return (bias, weights) minimising logistic loss + l2/2 * ||w||^2 (bias unpenalised)."""
    n, d = X.shape
    beta = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[0] = 0.0
    y = y.astype(np.float64)

    def chunks():
        for start in range(0, n, chunk):
            yield (np.hstack([np.ones((min(chunk, n - start), 1)), X[start:start + chunk].astype(np.float64)]),
                   y[start:start + chunk])

    def nll(b: np.ndarray) -> float:
        total = 0.5 * float(np.sum(reg * b * b))
        for Xc, yc in chunks():
            z = Xc @ b
            total += float(np.sum(np.logaddexp(0.0, z) - yc * z))
        return total

    current = nll(beta)
    for _ in range(max_iter):
        grad = reg * beta
        hess = np.diag(reg + 1e-9)
        for Xc, yc in chunks():
            p = 1.0 / (1.0 + np.exp(-(Xc @ beta)))
            grad += Xc.T @ (p - yc)
            hess += (Xc * (p * (1 - p))[:, None]).T @ Xc
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:
            cand = beta - t * step
            value = nll(cand)
            if value <= current + 1e-12 or t < 1e-4:
                break
            t *= 0.5
        beta, current = cand, value
        if np.max(np.abs(t * step)) < tol:
            break
    return float(beta[0]), beta[1:]


def fit_isotonic(score: np.ndarray, y: np.ndarray, max_bins: int = 400) -> tuple[list[float], list[float]]:
    """Increasing map score -> P(y=1), as knots for linear interpolation.

    Scores are first grouped into at most `max_bins` quantile bins (exact
    unique values when there are fewer). Weighted pool-adjacent-violators then
    runs on the bin means.
    """
    order = np.argsort(score, kind="mergesort")
    s, t = np.asarray(score, dtype=np.float64)[order], np.asarray(y, dtype=np.float64)[order]
    uniq, inverse = np.unique(s, return_inverse=True)
    if len(uniq) > max_bins:
        edges = np.unique(np.quantile(s, np.linspace(0, 1, max_bins + 1)))
        inverse = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    groups = np.bincount(inverse)
    keep = groups > 0
    w = groups[keep].astype(np.float64)
    xm = (np.bincount(inverse, weights=s)[keep]) / w
    ym = (np.bincount(inverse, weights=t)[keep]) / w
    # pool adjacent violators
    vals, wts, xs = [], [], []
    for xv, yv, wv in zip(xm, ym, w, strict=True):
        vals.append(yv), wts.append(wv), xs.append(xv * wv)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            xs[-2] += xs[-1]
            wts[-2] = wsum
            vals.pop(), wts.pop(), xs.pop()
    knots_x = [x / wv for x, wv in zip(xs, wts, strict=True)]
    return [round(v, 6) for v in knots_x], [round(min(max(v, 0.0), 1.0), 6) for v in vals]


def fit_conditional_logit(X: np.ndarray, y: np.ndarray, group_start: np.ndarray, l2: float = 1.0,
                          max_iter: int = 30, tol: float = 1e-6) -> tuple[float, np.ndarray]:
    """Listwise logit with an outside option: P(candidate k) = exp(s_k) / (1 + sum_j exp(s_j)), s = b + X w.

    Rows are candidates grouped by employee (`group_start` = first row of each group, rows contiguous);
    `y` marks the true manager (at most one per group; none = manager not among the candidates).
    Returns (bias, weights), fitted by Newton's method with an L2 penalty on the weights.
    """
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X.astype(np.float64)])
    beta = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[0] = 0.0
    group_id = np.repeat(np.arange(len(group_start)), np.diff(np.append(group_start, n)))
    y = y.astype(np.float64)

    def objective(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        s = Xb @ b
        gmax = np.maximum(np.maximum.reduceat(s, group_start), 0.0)
        e = np.exp(s - gmax[group_id])
        denom = np.exp(-gmax) + np.add.reduceat(e, group_start)
        nll = float(-(y @ s) + np.sum(np.log(denom) + gmax) + 0.5 * np.sum(reg * b * b))
        return nll, e / denom[group_id], s

    nll, p, _ = objective(beta)
    for _ in range(max_iter):
        grad = Xb.T @ (p - y) + reg * beta
        m = np.add.reduceat(Xb * p[:, None], group_start, axis=0)
        hess = (Xb * p[:, None]).T @ Xb - m.T @ m + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:  # backtracking line search keeps Newton monotone
            cand = beta - t * step
            new_nll, new_p, _ = objective(cand)
            if new_nll <= nll + 1e-12 or t < 1e-4:
                break
            t *= 0.5
        beta, nll, p = cand, new_nll, new_p
        if np.max(np.abs(t * step)) < tol:
            break
    return float(beta[0]), beta[1:]
