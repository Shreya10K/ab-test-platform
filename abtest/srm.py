from scipy.stats import chisquare


def srm_check(n_a, n_b, expected_ratio=0.5, threshold=0.001):
    """Chi-square goodness-of-fit on group sizes. expected_ratio = control share."""
    total = n_a + n_b
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    stat, p = chisquare([n_a, n_b], f_exp=expected)
    return {"chi2": stat, "p_value": p, "srm": p < threshold}
