"""Streamlit app for A/B Test Analyzer
Run: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.utils import (
    summary_by_variant,
    proportion_z_test,
    proportion_confint,
    two_sample_ttest,
    mean_confint,
    proportion_effect_size,
    cohens_d,
    required_sample_size_proportions,
    recommend_variant
)

st.set_page_config(page_title="A/B Test Analyzer", layout="wide")
st.title("A/B Test Analyzer — Significance, CI, and Winner Recommendation")

st.markdown("""
Upload a CSV with at least:
- `user_id` (optional), `variant` (values: A or B), and one metric column:
  - For binary metric (conversion): `converted` with values 0/1
  - For continuous metric (revenue): `revenue` (float)
You can include both; choose the metric to test.
""")

uploaded = st.file_uploader("Upload CSV (or use sample)", type=["csv"])
use_sample = False
if uploaded is None:
    use_sample = st.checkbox("Use sample dataset", value=True)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    sample_data = {
        'user_id': list(range(1, 201)),
        'variant': ['A']*100 + ['B']*100,
    }
    rng = np.random.default_rng(42)
    sample_data['converted'] = np.concatenate([
        rng.binomial(1, 0.08, 100),
        rng.binomial(1, 0.12, 100)
    ])
    rev_A = rng.normal(10, 2, 100)
    rev_B = rng.normal(10.5, 2.2, 100)
    sample_data['revenue'] = np.concatenate([rev_A, rev_B])
    df = pd.DataFrame(sample_data)
    st.write("Using sample dataset (simulated)")

if 'df' in locals():
    st.subheader("Raw data (first 10 rows)")
    st.write(df.head(10))

    st.sidebar.header("Test configuration")
    variant_col = st.sidebar.text_input("Variant column name", value="variant")
    metric_type = st.sidebar.selectbox("Metric type", options=["binary (conversion)", "continuous (revenue)"])
    if metric_type.startswith("binary"):
        metric_col = st.sidebar.text_input("Binary metric column (0/1)", value="converted")
    else:
        metric_col = st.sidebar.text_input("Continuous metric column", value="revenue")

    alpha = st.sidebar.number_input("Significance level (alpha)", value=0.05, min_value=0.001, max_value=0.2, step=0.005)
    power = st.sidebar.number_input("Target power for sample size calc", value=0.8, min_value=0.5, max_value=0.99, step=0.01)

    if variant_col not in df.columns or metric_col not in df.columns:
        st.warning(f"Please ensure the file has columns: {variant_col} and {metric_col}")
    else:
        if metric_type.startswith("binary"):
            s = summary_by_variant(df, variant_col, metric_col, metric_type='binary')
            st.subheader("Conversion Summary")
            st.write(pd.DataFrame(s).T)

            try:
                a = s['A']
                b = s['B']
            except KeyError:
                st.error("Variants must include both 'A' and 'B'")
                st.stop()

            z, p, pooled = proportion_z_test(a['successes'], a['n'], b['successes'], b['n'], alternative='two-sided')
            ci_a = proportion_confint(a['successes'], a['n'], alpha=alpha)
            ci_b = proportion_confint(b['successes'], b['n'], alpha=alpha)
            diff, lift = proportion_effect_size(a['prop'], b['prop'])
            n_req = required_sample_size_proportions(a['prop'], b['prop'], alpha=alpha, power=power)

            st.write("**Proportion z-test**")
            st.write(f"z-statistic: {z:.4f}    p-value: {p:.4f}")
            st.write(f"A: prop={a['prop']:.4f}, 95% CI={ci_a}")
            st.write(f"B: prop={b['prop']:.4f}, 95% CI={ci_b}")
            st.write(f"Absolute difference (B - A): {diff:.4f}")
            st.write(f"Relative lift (vs A): {lift:.2%} (if A>0 else inf)")
            st.write(f"Approx. required sample per group to detect this lift (power={power}): {n_req}")

            rec = recommend_variant(p, metric_col, diff, alpha=alpha)
            if rec == 'B':
                st.success("Recommendation: **Variant B is better** (statistically significant).")
            elif rec == 'A':
                st.info("Recommendation: **Variant A is better** (statistically significant).")
            else:
                st.warning("Recommendation: **Inconclusive** — not statistically significant.")

        else:
            s = summary_by_variant(df, variant_col, metric_col, metric_type='continuous')
            st.subheader("Continuous Metric Summary")
            st.write(pd.DataFrame(s).T)

            try:
                a = s['A']; b = s['B']
            except KeyError:
                st.error("Variants must include both 'A' and 'B'")
                st.stop()

            t_stat, p = two_sample_ttest(a['mean'], a['std'], a['n'], b['mean'], b['std'], b['n'], alternative='two-sided')
            ci_a = mean_confint(a['mean'], a['std'], a['n'], alpha=alpha)
            ci_b = mean_confint(b['mean'], b['std'], b['n'], alpha=alpha)
            d = cohens_d(a['mean'], b['mean'], a['std'], b['std'], a['n'], b['n'])

            st.write("**Two-sample t-test (Welch approx)**")
            st.write(f"t-statistic: {t_stat:.4f}    p-value: {p:.4f}")
            st.write(f"A: mean={a['mean']:.4f}, std={a['std']:.4f}, 95% CI={ci_a}")
            st.write(f"B: mean={b['mean']:.4f}, std={b['std']:.4f}, 95% CI={ci_b}")
            st.write(f"Cohen's d (effect size): {d:.4f}")

            diff = b['mean'] - a['mean']
            rec = recommend_variant(p, metric_col, diff, alpha=alpha)
            if rec == 'B':
                st.success("Recommendation: **Variant B is better** (statistically significant).")
            elif rec == 'A':
                st.info("Recommendation: **Variant A is better** (statistically significant).")
            else:
                st.warning("Recommendation: **Inconclusive** — not statistically significant.")
