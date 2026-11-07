import numpy as np

from abtest import cuped, sample_size as ss, srm
from abtest.data import make_data
from abtest.frequentist import welch_t_test, z_test_proportions


def test_sample_size_textbook():
    assert abs(ss.n_proportions(0.10, 0.10) - 14750) < 100


def test_srm_flags_imbalance():
    assert srm.srm_check(10300, 9700)["srm"]
    assert not srm.srm_check(10010, 9990)["srm"]


def test_worked_example_z_test():
    r = z_test_proportions(1512, 15120, 1637, 14880)
    assert abs(r["stat"] - 2.83) < 0.02
    assert abs(r["p_value"] - 0.005) < 0.001


def test_cuped_reduction_equals_rho_sq():
    df = make_data(n=50000)
    r = cuped.cuped(df.pre_metric, df.metric)
    assert abs(r["var_reduction"] - r["rho"] ** 2) < 0.01
    assert abs(r["y_adj"].mean() - df.metric.mean()) < 1e-9  # mean preserved


def test_aa_false_positive_rate():
    # A/A: no true effect → ~5% significant at α = 0.05
    rng = np.random.default_rng(1)
    hits = sum(welch_t_test(rng.normal(0, 1, 500), rng.normal(0, 1, 500))["p_value"] < 0.05
               for _ in range(1000))
    assert 30 < hits < 70
