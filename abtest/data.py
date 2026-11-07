import numpy as np
import pandas as pd


def make_data(n=20000, effect=2.0, conv_lift=0.08, split=0.5, seed=42):
    """Synthetic A/B data with a known effect. corr(pre, post) ≈ 0.46 → ρ² ≈ 0.21."""
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(n) < split, "B", "A")
    is_b = group == "B"

    pre = rng.normal(100, 20, n)
    # corr = 0.5*20 / sqrt(10^2 + 19.3^2) ≈ 0.46
    post = 50 + 0.5 * pre + rng.normal(0, 19.3, n) + np.where(is_b, effect, 0)

    logit = -2.2 + 0.01 * (pre - 100) + np.where(is_b, conv_lift, 0)
    converted = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({"user_id": np.arange(n), "group": group,
                         "pre_metric": pre, "metric": post, "converted": converted})
