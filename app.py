import pandas as pd
import plotly.express as px
import streamlit as st

from abtest import bayesian as by, cuped as cp, frequentist as fq, sample_size as ss, srm
from abtest.data import make_data

st.set_page_config(page_title="A/B Test Analyzer", layout="wide")
st.title("A/B Test Analysis Platform")
make_data = st.cache_data(make_data)

with st.sidebar:
    st.header("Data")
    if st.radio("Source", ["Synthetic", "Upload CSV"]) == "Upload CSV":
        f = st.file_uploader("CSV", type="csv")
        df = pd.read_csv(f) if f else None
    else:
        df = make_data(n=int(st.number_input("Users", 1000, 500_000, 20_000, 1000)))
    alpha = st.slider("α", 0.01, 0.10, 0.05)

t_ss, t_srm, t_freq, t_bayes, t_cuped = st.tabs(
    ["Sample size", "SRM", "Frequentist", "Bayesian", "CUPED"])

with t_ss:
    power = st.slider("Power", 0.5, 0.99, 0.8)
    kind = st.radio("Metric type", ["Proportion", "Mean"], horizontal=True)
    if kind == "Proportion":
        p1 = st.number_input("Baseline rate", 0.001, 0.99, 0.10)
        mde = st.number_input("Relative MDE", 0.001, 2.0, 0.10)
        n = ss.n_proportions(p1, mde, alpha, power)
    else:
        sd = st.number_input("Std dev", 0.01, value=20.0)
        mde = st.number_input("Absolute MDE", 0.001, value=1.0)
        n = ss.n_means(sd, mde, alpha, power)
    st.metric("Users per group", f"{n:,}")

if df is None:
    st.info("Upload a CSV to run the analyses.")
    st.stop()

cols = list(df.columns)
with st.sidebar:
    gcol = st.selectbox("Group column", cols, index=cols.index("group") if "group" in cols else 0)
    labels = sorted(df[gcol].astype(str).unique().tolist())
    ctrl = st.selectbox("Control", labels, index=0)
    trt = st.selectbox("Treatment", labels, index=min(1, len(labels) - 1))
    mcol = st.selectbox("Metric", cols, index=cols.index("metric") if "metric" in cols else 0)
    binary = set(df[mcol].dropna().unique()) <= {0, 1}

g = df[gcol].astype(str)
A, B = df[g == ctrl], df[g == trt]
a, b = A[mcol].to_numpy(), B[mcol].to_numpy()

with t_srm:
    ratio = st.number_input("Expected control share", 0.01, 0.99, 0.5)
    r = srm.srm_check(len(A), len(B), ratio)
    st.write(f"Control: {len(A):,} | Treatment: {len(B):,} | χ² = {r['chi2']:.2f} | p = {r['p_value']:.4g}")
    if r["srm"]:
        st.error("SRM detected — don't trust results")
    else:
        st.success("No SRM")


def show(res, fmt):
    c = st.columns(4)
    c[0].metric("Control", fmt(res["a"]))
    c[1].metric("Treatment", fmt(res["b"]))
    c[2].metric("Lift", f"{res['lift']:+.2%}")
    c[3].metric("p-value", f"{res['p_value']:.4f}")
    st.write(f"{1 - alpha:.0%} CI of difference: [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
    if res["p_value"] < alpha:
        st.success("Significant")
    else:
        st.warning("Not significant")


with t_freq:
    if binary:
        show(fq.z_test_proportions(a.sum(), len(a), b.sum(), len(b), alpha), lambda v: f"{v:.2%}")
    else:
        show(fq.welch_t_test(a, b, alpha), lambda v: f"{v:.3f}")

with t_bayes:
    r = (by.beta_binomial(a.sum(), len(a), b.sum(), len(b)) if binary
         else by.normal_means(a, b))
    c = st.columns(3)
    c[0].metric("P(B > A)", f"{r['prob_b_better']:.1%}")
    c[1].metric("Expected loss if ship B", f"{r['loss_ship_b']:.5f}")
    c[2].metric("95% credible lift", f"[{r['lift_ci'][0]:+.2%}, {r['lift_ci'][1]:+.2%}]")
    st.plotly_chart(px.histogram(x=r["lift_samples"], nbins=100,
                                 labels={"x": "Relative lift"}), use_container_width=True)

with t_cuped:
    options = [c for c in cols if c not in (gcol, mcol)]
    if not options:
        st.info("No pre-experiment column available.")
    else:
        xcol = st.selectbox("Pre-experiment covariate", options,
                            index=options.index("pre_metric") if "pre_metric" in options else 0)
        both = pd.concat([A, B])
        r = cp.cuped(both[xcol], both[mcol])
        both = both.assign(adj=r["y_adj"])
        bg = both[gcol].astype(str)
        c = st.columns(3)
        c[0].metric("θ", f"{r['theta']:.3f}")
        c[1].metric("ρ", f"{r['rho']:.3f}")
        c[2].metric("Variance / sample-size reduction", f"{r['var_reduction']:.1%}")
        raw = fq.welch_t_test(a, b, alpha)
        adj = fq.welch_t_test(both.loc[bg == ctrl, "adj"], both.loc[bg == trt, "adj"], alpha)
        st.dataframe(pd.DataFrame({
            "Raw": [raw["diff"], raw["p_value"], raw["ci"][1] - raw["ci"][0]],
            "CUPED": [adj["diff"], adj["p_value"], adj["ci"][1] - adj["ci"][0]]},
            index=["Effect", "p-value", "CI width"]))
