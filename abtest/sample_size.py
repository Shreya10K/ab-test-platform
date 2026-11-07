import numpy as np
from scipy.stats import norm


def _z(alpha, power):
    return norm.ppf(1 - alpha / 2), norm.ppf(power)


def n_proportions(p1, mde_rel, alpha=0.05, power=0.8):
    """Users per group to detect a relative lift `mde_rel` on baseline rate p1."""
    p2 = p1 * (1 + mde_rel)
    za, zb = _z(alpha, power)
    pbar = (p1 + p2) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar))
           + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def n_means(sd, mde_abs, alpha=0.05, power=0.8):
    """Users per group to detect an absolute difference `mde_abs` in a mean."""
    za, zb = _z(alpha, power)
    return int(np.ceil(2 * sd**2 * (za + zb) ** 2 / mde_abs**2))
