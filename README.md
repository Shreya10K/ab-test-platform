# A/B Test Analysis Platform

Streamlit app for experiment analysis: sample size, SRM check, frequentist and Bayesian testing, and CUPED variance reduction.

## Run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

## CSV format (for upload)

One row per user: a group column (e.g. `A`/`B`), a metric column (0/1 for conversion, or numeric), and optionally a pre-experiment column for CUPED.

## Structure

| File | Purpose |
| --- | --- |
| `abtest/data.py` | Synthetic data with a known effect (ρ ≈ 0.46) |
| `abtest/sample_size.py` | Power analysis |
| `abtest/srm.py` | Sample ratio mismatch (chi-square) |
| `abtest/frequentist.py` | Two-proportion z-test, Welch t-test, CIs |
| `abtest/bayesian.py` | Beta-Binomial and normal posteriors, P(B>A), expected loss |
| `abtest/cuped.py` | CUPED adjustment |
| `app.py` | Streamlit UI |
| `tests/test_stats.py` | Statistical sanity tests |
