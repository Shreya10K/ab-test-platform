import numpy as np
from scipy.stats import norm, ttest_ind, t as t_dist


def z_test_proportions(conv_a, n_a, conv_b, n_b, alpha=0.05):
    pa, pb = conv_a / n_a, conv_b / n_b
    # Test assumes H0 (one shared rate) → pooled SE
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se_pool
    p_value = 2 * (1 - norm.cdf(abs(z)))

    # CI describes the real difference → unpooled SE
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    zc, diff = norm.ppf(1 - alpha / 2), pb - pa
    return {"a": pa, "b": pb, "diff": diff, "lift": diff / pa,
            "stat": z, "p_value": p_value, "ci": (diff - zc * se, diff + zc * se)}


def welch_t_test(a, b, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    stat, p = ttest_ind(b, a, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))  # Welch–Satterthwaite
    tc, diff = t_dist.ppf(1 - alpha / 2, df), b.mean() - a.mean()
    return {"a": a.mean(), "b": b.mean(), "diff": diff, "lift": diff / a.mean(),
            "stat": stat, "p_value": p, "ci": (diff - tc * se, diff + tc * se)}
