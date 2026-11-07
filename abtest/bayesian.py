import numpy as np


def _summarize(sa, sb):
    lift = (sb - sa) / sa
    return {"prob_b_better": (sb > sa).mean(),
            "loss_ship_b": np.maximum(sa - sb, 0).mean(),
            "loss_ship_a": np.maximum(sb - sa, 0).mean(),
            "lift_ci": np.percentile(lift, [2.5, 97.5]),
            "lift_samples": lift}


def beta_binomial(conv_a, n_a, conv_b, n_b, prior=(1, 1), draws=100_000, seed=0):
    """Beta prior + binomial data → Beta posterior (conjugacy), then Monte Carlo."""
    rng = np.random.default_rng(seed)
    sa = rng.beta(prior[0] + conv_a, prior[1] + n_a - conv_a, draws)
    sb = rng.beta(prior[0] + conv_b, prior[1] + n_b - conv_b, draws)
    return _summarize(sa, sb)


def normal_means(a, b, draws=100_000, seed=0):
    """Flat prior, large n: posterior of mean ≈ N(x̄, s²/n) by the CLT."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    sa = rng.normal(a.mean(), a.std(ddof=1) / np.sqrt(len(a)), draws)
    sb = rng.normal(b.mean(), b.std(ddof=1) / np.sqrt(len(b)), draws)
    return _summarize(sa, sb)
