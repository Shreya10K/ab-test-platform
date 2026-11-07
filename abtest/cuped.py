import numpy as np


def cuped(x, y):
    """x must be measured BEFORE the experiment. Variance shrinks by ρ²."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    return {"y_adj": y_adj, "theta": theta,
            "rho": np.corrcoef(x, y)[0, 1],
            "var_reduction": 1 - y_adj.var(ddof=1) / y.var(ddof=1)}
