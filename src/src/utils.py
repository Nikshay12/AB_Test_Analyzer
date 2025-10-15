"""
Statistical helpers for A/B Test Analyzer
- supports:
  * proportion z-test for conversions (binary)
  * two-sample t-test for continuous metrics (revenue)
  * confidence intervals
  * effect size calculation
  * basic sample size estimation for proportions
  * recommendation logic
"""

import numpy as np
from math import sqrt
from scipy import stats

def summary_by_variant(df, variant_col, metric_col, metric_type='binary'):
    groups = df.groupby(variant_col)
    summary = {}
    for name, g in groups:
        n = len(g)
        if metric_type == 'binary':
            success = int(g[metric_col].sum())
            prop = success / n if n > 0 else 0.0
            summary[name] = {'n': n, 'successes': success, 'prop': prop}
        else:
            mean = g[metric_col].mean()
            std = g[metric_col].std(ddof=1)
            summary[name] = {'n': n, 'mean': mean, 'std': std}
    return summary

def proportion_z_test(success_a, n_a, success_b, n_b, alternative='two-sided'):
    p1 = success_a / n_a if n_a>0 else 0
    p2 = success_b / n_b if n_b>0 else 0
    pooled = (success_a + success_b) / (n_a + n_b) if (n_a + n_b)>0 else 0
    se = sqrt(pooled * (1 - pooled) * (1/n_a + 1/n_b)) if (n_a>0 and n_b>0) else 0
    if se == 0:
        return np.nan, np.nan, pooled
    z = (p2 - p1) / se
    if alternative == 'two-sided':
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'larger':
        p = 1 - stats.norm.cdf(z)
    else:
        p = stats.norm.cdf(z)
    return z, p, pooled

def proportion_confint(success, n, alpha=0.05):
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha/2)
    phat = success / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    margin = (z * sqrt((phat*(1-phat))/n + z**2/(4*n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def two_sample_ttest(mean_a, std_a, n_a, mean_b, std_b, n_b, alternative='two-sided'):
    se = sqrt((std_a**2)/n_a + (std_b**2)/n_b) if (n_a>0 and n_b>0) else 0
    if se == 0:
        return np.nan, np.nan
    t_stat = (mean_b - mean_a) / se
    # Welch-Satterthwaite df
    num = (std_a**2/n_a + std_b**2/n_b)**2
    den = (std_a**4)/((n_a**2)*(n_a-1)) + (std_b**4)/((n_b**2)*(n_b-1)) if (n_a>1 and n_b>1) else 0
    df = num / den if den != 0 else max(n_a + n_b - 2, 1)
    if alternative == 'two-sided':
        p = 2 * stats.t.sf(abs(t_stat), df)
    elif alternative == 'larger':
        p = stats.t.sf(t_stat, df)
    else:
        p = stats.t.cdf(t_stat, df)
    return t_stat, p

def mean_confint(mean, std, n, alpha=0.05):
    if n <= 1:
        return mean, mean
    se = std / sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    lower = mean - t_crit * se
    upper = mean + t_crit * se
    return lower, upper

def proportion_effect_size(p_a, p_b):
    diff = p_b - p_a
    lift = (diff / p_a) if p_a != 0 else np.inf
    return diff, lift

def cohens_d(mean_a, mean_b, sd_a, sd_b, n_a, n_b):
    pooled_sd = sqrt(((n_a-1)*sd_a**2 + (n_b-1)*sd_b**2) / (n_a + n_b - 2)) if (n_a + n_b - 2) > 0 else 0
    if pooled_sd == 0:
        return 0.0
    return (mean_b - mean_a) / pooled_sd

def required_sample_size_proportions(p1, p2, alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    se_part = p1*(1-p1) + p2*(1-p2)
    delta = abs(p2 - p1)
    if delta == 0:
        return float('inf')
    n = ((z_alpha * sqrt(se_part/2) + z_beta * sqrt(se_part/2))**2) / (delta**2)
    return int(np.ceil(n))

def recommend_variant(p_value, metric, direction, alpha=0.05):
    if np.isnan(p_value):
        return 'inconclusive'
    if p_value < alpha:
        return 'B' if direction > 0 else 'A'
    return 'inconclusive'
