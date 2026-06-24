# Test Statistics for Event Studies

All 13 test statistics as implemented in Kaspereit's eventstudy2 (v3.2b),
transcribed from the Mata source code (lines 1646-1841) and cross-referenced
with the original papers.

## Notation

| Symbol | Meaning |
|--------|---------|
| AR_{i,tau} | Abnormal return for firm i at estimation-window time tau |
| ARE_it | Abnormal return for firm i at event-window time t |
| T_i | Number of non-missing estimation-window observations for firm i |
| N_t | Number of firms with valid ARs at event time t |
| df | Number of additional factors beyond the market (0 for MM, 2 for FF3) |
| sigma_hat_i | Residual std dev: sqrt(sum AR^2_{i,tau} / (T_i - 2 - df)) |
| STDF_it | Standard deviation of forecast (Theil prediction error) |
| C_it | Theil correction: (STDF_it / sigma_hat_i)^2 = 1 + x'_t(X'X)^{-1}x_t |
| AAR_t | Average abnormal return: (1/N_t) * sum_i ARE_it |
| CAAR_t | Cumulative AAR from event start through day t |

## 1. Cross-Sectional t-test (Serra 2002)

**Null hypothesis**: AAR_t = 0 (CAAR = 0 for cumulative version).

**AAR formula**:

    sigma_AAR_t = sqrt( sum_i STDF^2_it / N^2_t )

    t_t = AAR_t / sigma_AAR_t

**CAAR formula**: Uses the Theil (1971) prediction error correction for
cumulative forecasts. For each firm-event, the cumulative prediction error
variance accounts for the growing uncertainty as more days are accumulated:

    C_it^cum = L + L/T_i + [sum_{s=1}^{L} (R_m,s - L*R_m_bar)]^2 / sum_tau(R_m,tau - R_m_bar)^2

    sigma_CAAR = sqrt( (1/N^2) * sum_i (sigma_hat_i^2 * C_it^cum) )

    t_CAAR = CAAR / sigma_CAAR

where L is the number of event days cumulated and R_m_bar is the mean market
return from the estimation window.

**Distribution**: t(T_est - 2 - df). In practice T_est is large enough that
this is approximately standard normal.

**Applies to**: AAR and CAAR.

**Python**:
```python
from scipy.stats import t as t_dist
t_stat = aar / np.sqrt(np.sum(stdf_event**2) / n_firms**2)
p_val = 2 * t_dist.sf(np.abs(t_stat), df=T_est - 2 - n_factors)
```

## 2. Brown-Warner CDA (1980/1985)

**Null hypothesis**: AAR_t = 0.

**Intuition**: Uses the time-series variance of AARs from the estimation
window as the benchmark. This is a "crude" dependence adjustment because it
implicitly captures cross-correlation to the extent that it was present in the
estimation window.

**AAR formula**:

    sigma^2_CDA = Var(AAR_tau) * (T_est - 1) / (T_est - 2 - df)

where Var(AAR_tau) is the sample variance of the T_est estimation-window
average abnormal returns. The (T-1)/(T-2-df) adjustment corrects degrees of
freedom.

    z_t = AAR_t / sigma_CDA

**CAAR formula**:

    sigma^2_CAAR_CDA = L * sigma^2_CDA

    z_CAAR = CAAR / sqrt(L * sigma^2_CDA)

where L is the number of event days in the CAR window.

**Distribution**: t(T_est - 2 - df).

**Applies to**: AAR and CAAR.

**Python**:
```python
aar_est = estimation_window_aar  # shape (T_est,)
var_cda = np.var(aar_est, ddof=1) * (T_est - 1) / (T_est - 2 - df)
z = aar_event / np.sqrt(var_cda)
# For CAAR:
z_caar = caar / np.sqrt(L * var_cda)
```

## 3. Patell (1976)

**Null hypothesis**: AAR_t = 0.

**Intuition**: Standardizes each firm's AR by its own prediction error (STDF),
producing unit-variance standardized abnormal returns (SARs). Aggregates SARs
across firms. More powerful than the CDA test when firms have heterogeneous
variances.

**Standardized abnormal return (SAR)**:

    SAR_it = ARE_it / STDF_it

Equivalently: SAR_it = ARE_it / (sigma_hat_i * sqrt(C_it))

Under the null, Var(SAR_it) = (T_i - 2 - df) / (T_i - 4 - df) (the
denominator adjustment accounts for estimation uncertainty).

**AAR formula**:

    Z_Patell_t = sum_i SAR_it / sqrt( sum_i (T_i - 2 - df) / (T_i - 4 - df) )

**CAAR formula** (cumulative standardized prediction errors):

    L_t = running count of non-missing event-window days through time t

    SCAR_it = cumsum_{s=1}^{t} [ SAR_is / sqrt(L_s) ]

    Z_Patell_CAAR = sum_i SCAR_it / sqrt( sum_i (T_i - 2 - df) / (T_i - 4 - df) )

The division by sqrt(L_s) at each step ensures the running sum maintains unit
variance as days accumulate.

**Distribution**: N(0, 1) (standard normal).

**Applies to**: AAR and CAAR.

**Python**:
```python
from scipy.stats import norm
sar = ar_event / stdf_event  # shape (T_event, N_firms)
var_sar = (T_firms - 2 - df) / (T_firms - 4 - df)  # shape (N_firms,)
z_patell = np.nansum(sar, axis=1) / np.sqrt(np.nansum(var_sar))
p_val = 2 * norm.sf(np.abs(z_patell))
```

> **Implementation note**: eventstudy2 v3.2b computes Cit as STDF/sigma
> (which equals sqrt(C_it)), then divides ARE by sigma and by sqrt(Cit),
> yielding a denominator of sigma * C_it^(1/4) rather than the textbook
> sigma * C_it^(1/2). The practical difference is negligible for typical
> estimation windows (T > 100). This skill uses the textbook formulation,
> which is internally consistent with the (T-2-df)/(T-4-df) variance
> adjustment in the Patell aggregation.

## 4. Patell + Kolari-Pynnonen (2010) Adjustment

**Null hypothesis**: AAR_t = 0, accounting for cross-correlation.

**Formula**: Multiply the Patell Z by the ADJ factor:

    Z_Patell_KP = Z_Patell * ADJ

    ADJ = sqrt( (1 - r_bar) / (1 + (N-1) * r_bar) )

where r_bar is the mean pairwise cross-correlation of estimation-window
standardized residuals. See `kolari_pynnonen.md` for the ADJ computation.

**Distribution**: N(0, 1).

**Applies to**: AAR and CAAR.

## 5. Boehmer et al. (1991) BMP

**Null hypothesis**: AAR_t = 0, robust to event-induced variance changes.

**Intuition**: Cross-sectional t-test on SARs. Unlike Patell (which assumes
SARs have the same variance across firms), BMP uses the actual cross-sectional
variance of SARs, making it robust to event-induced changes in volatility.

**AAR formula**:

    BMP_t = mean(SAR_it) / (sd_cs(SAR_it) / sqrt(N_t))

More precisely:

    BMP_t = (sum_i SAR_it / N_t) / sqrt( (1/N_t) * Var_cs(SAR_it) )

where Var_cs is the cross-sectional variance of SAR_it across firms at time t.

**CAAR formula**: Same structure using SCARs:

    BMP_CAAR = (sum_i SCAR_it / N) / sqrt( (1/N) * Var_cs(SCAR_it) )

**Distribution**: t(T_est - 2 - df). In practice, approximately N(0,1) for
large samples.

**Applies to**: AAR and CAAR.

**Python**:
```python
sar_t = sar[t, :]  # SARs across firms at event day t
bmp = np.nanmean(sar_t) / (np.nanstd(sar_t, ddof=1) / np.sqrt(n_firms))
```

## 6. Kolari-Pynnonen Adjusted BMP

**Formula**:

    BMP_KP = BMP * ADJ

Applies the same cross-correlation adjustment factor as for Patell.

**Distribution**: t(T_est - 2 - df).

**Applies to**: AAR and CAAR.

## 7. Corrado (1989) Rank Test

**Null hypothesis**: AAR_t = 0 (nonparametric).

**Intuition**: Ranks ARs across the combined estimation + event window for
each firm, then tests whether event-window ranks deviate from the expected
rank. Robust to non-normality of returns.

**Procedure**:

1. Stack estimation-window ARs and event-window ARs into a single vector per
   firm: `AR_total_i = [AR_est_i ; AR_event_i]` (length T_i + L).

2. Rank within each firm: `K_it = rank(AR_total_i)` for t = 1, ..., T_i + L.
   Missing values get missing ranks.

3. Expected rank: `E[K] = (T_total + 1) / 2` where T_total is the count of
   non-missing observations.

4. For each event day t, compute the mean rank deviation:

       S_t = (1/N_t) * sum_i (K_it - E[K_i])

5. Compute the standard deviation from the full time series of S_t:

       SK = sqrt( (1/T_total) * sum_t S^2_t )

6. Test statistic:

       T_Corrado = S_t / SK

**Distribution**: N(0, 1).

**Applies to**: AAR only. For CAAR, use the Cowan (1992) cumulative rank test.

**Python**:
```python
from scipy.stats import rankdata
# Per firm: rank across combined est + event window
ranks = rankdata(ar_combined, nan_policy='omit')
expected_rank = (n_valid + 1) / 2
rank_deviation = ranks - expected_rank
# Aggregate and normalize as above
```

## 8. Corrado-Zivney (1992) Volatility-Adjusted Rank Test

**Null hypothesis**: AAR_t = 0 (nonparametric, robust to event-induced
volatility).

**Procedure**:

1. Standardize ARs by their time-series standard deviation (from estimation
   window):

       AR_std_it = AR_it / sqrt(Var_ts(AR_i))

2. For event-window ARs, further standardize by the cross-sectional standard
   deviation:

       AR_dblstd_it = AR_std_event_it / sd_cs(AR_std_event_t)

3. Stack `AR_std_est` and `AR_dblstd_event` into a single vector per firm.

4. Rank within each firm, transform to U_it = rank / (T+1), then subtract 0.5.

5. Aggregate:

       T_Zivney = (1/sqrt(N_t)) * sum_i (U_it - 0.5) / S_U

   where S_U is the standard deviation of the mean-U series from the
   estimation window.

**Distribution**: N(0, 1).

**Applies to**: AAR only. For CAAR, use the Zivney-Cowan cumulative version.

## 9. Cowan (1992) Cumulative Rank Test

**Null hypothesis**: CAAR = 0 (nonparametric).

Extension of the Corrado rank test to cumulative windows.

**Procedure**:

1. Compute ranks as in test #7.

2. For the cumulation window [1, L]:

       KD_bar = (1/L) * sum_{s=1}^{L} mean_rank_s

   where mean_rank_s is the average rank across firms at event day s.

3. Expected cumulative rank (accounting for the growing window).

4. Standard deviation estimated from the full time-series of average ranks:

       denom = sqrt( (1/T_total) * sum_t (K_t_bar - expected)^2 )

5. Test statistic:

       Z_Cowan = sqrt(L) * (KD_bar - expected) / denom

**Distribution**: N(0, 1).

**Applies to**: CAAR only.

## 10. Zivney-Cowan Cumulative Rank Test

**Null hypothesis**: CAAR = 0 (nonparametric, volatility-adjusted).

Same as test #9 but uses the volatility-adjusted ranks from test #8 instead of
the raw Corrado ranks. Accounts for event-induced changes in volatility.

**Distribution**: N(0, 1).

**Applies to**: CAAR only.

## 11. GRANK-T (Kolari-Pynnonen 2011)

**Null hypothesis**: CAAR = 0 (nonparametric, accounts for cross-correlation).

The most robust nonparametric test. Combines time-series standardization,
cross-sectional standardization, rank transformation, and the
Kolari-Pynnonen cross-correlation adjustment.

> Read `references/kolari_pynnonen.md` for the full GRANK-T derivation.

**Summary of procedure**:

1. Compute cumulative SARs (SCARs) adjusted by ADJ and cross-sectional std:

       SCAR*_i = (SCAR_i * ADJ) / sd_cs(SCAR)

2. Take the final-row SCAR* (the full-window SCAR*).

3. Compute estimation-window SARs: SAR_tau = AR_tau / sigma_hat_i.

4. Stack: [SAR_est ; SCAR*_event] — estimation-window SARs on top, final
   SCAR* values on bottom.

5. Rank the stacked vector per firm.

6. Transform to U_it = rank / (T_total + 1) - 0.5.

7. Compute U_bar_t = mean(U_it) across firms for each time period.

8. S_U = sqrt( (1/T_est) * sum_{est window} (N_est/N_event * U_bar_t^2) )

9. Z_GRANK = U_bar_event / S_U

10. Transform to t-statistic:

        t_GRANK = Z_GRANK * sqrt( (T_est - 2) / (T_est - 1 - Z_GRANK^2) )

**Distribution**: t(T_est - 2).

**Applies to**: CAAR only.

## 12. Generalized Sign Test (Cowan 1992)

**Null hypothesis**: The fraction of positive CARs equals the expected
fraction from the estimation window.

**Intuition**: A nonparametric test that doesn't assume symmetry of the CAR
distribution. Instead, it estimates the expected fraction of positive ARs from
the estimation window and tests whether the event window has significantly
more (or fewer) positive ARs.

**Procedure**:

1. Estimate p_hat from the estimation window:

       p_hat = (number of positive estimation-window ARs) / (2 * total est ARs) + 0.5

   Note: the `sign()` function maps positive to +1, negative to -1, zero to
   0. The formula `sum(sign(AR)) / (2*N) + 0.5` converts this to a proportion.

2. Count non-negative event-window ARs (ARE >= 0 counts as positive,
   following eventstudy2's `sign(sign(ARE)+1)` convention):

       n_nonneg = count(ARE_it >= 0)

3. Test statistic (binomial z-test):

       Z_gsign = (n_nonneg - N * p_hat) / sqrt(N * p_hat * (1 - p_hat))

**CAAR version**: Apply the sign test to cumulative ARs rather than daily ARs.
Count the number of firms with positive CAR over the window.

**Distribution**: N(0, 1).

**Applies to**: AAR and CAAR.

**Python**:
```python
# Estimation window
p_hat = np.sum(np.sign(ar_est)) / (2 * np.sum(~np.isnan(ar_est))) + 0.5
# Event window (>= 0, not > 0: zero ARs count as positive)
n_nonneg = np.sum(ar_event >= 0)
n_total = np.sum(~np.isnan(ar_event))
z_gsign = (n_nonneg - n_total * p_hat) / np.sqrt(n_total * p_hat * (1 - p_hat))
p_val = 2 * norm.sf(np.abs(z_gsign))
```

## 13. Wilcoxon Signed-Rank Test

**Null hypothesis**: Median AR = 0 (AAR = 0 for aggregated version).

**Intuition**: A nonparametric alternative to the t-test that uses ranks of
absolute values rather than the values themselves. Robust to outliers and
non-normality.

**AAR formula** (for each event day t):

1. Take the N_t event-window ARs at time t.

2. Rank the absolute values: R_i = rank(|ARE_it|) for i = 1, ..., N_t.

3. Signed rank sum using indicator I(ARE >= 0) rather than sign():

       S_n = sum_i I(ARE_it >= 0) * R_i

   This follows the indicator formulation as implemented in eventstudy2,
   which counts zero ARs as positive (line 1810: `sign(sign(ARE)+1)`).

4. Expected value under H0:

       E[S_n] = N_t * (N_t + 1) / 4

5. Variance under H0:

       Var[S_n] = N_t * (N_t + 1) * (2*N_t + 1) / 24

6. Test statistic:

       Z_Wilcox = (S_n - E[S_n]) / sqrt(Var[S_n])

**CAAR formula**: For the cumulation window [1, L], vectorize all L*N ARs into
a single vector, compute ranks and the signed-rank statistic on the pooled
set.

Specifically, for cumulation from day 1 to day r:
- Flatten ARE[1:r, :] into a single vector of length r*N_t
- Apply the Wilcoxon formula to this pooled vector

**Distribution**: N(0, 1) (by normal approximation for N >= 10).

**Applies to**: AAR and CAAR.

**Python**:
```python
from scipy.stats import rankdata, norm

ar_t = ar_event[t, :]  # ARs across firms at event day t
ar_t = ar_t[~np.isnan(ar_t)]
n = len(ar_t)
ranks = rankdata(np.abs(ar_t))
S_n = np.sum((ar_t >= 0).astype(float) * ranks)
E_Sn = n * (n + 1) / 4
Var_Sn = n * (n + 1) * (2 * n + 1) / 24
z_wilcox = (S_n - E_Sn) / np.sqrt(Var_Sn)
p_val = 2 * norm.sf(np.abs(z_wilcox))
```

---

## Summary Table

| # | Test | Type | H0 Dist | AAR | CAAR | Cross-corr robust |
|---|------|------|---------|-----|------|--------------------|
| 1 | Serra t-test | Parametric | t(T-2-df) | Yes | Yes | No |
| 2 | Brown-Warner CDA | Parametric | t(T-2-df) | Yes | Yes | Partially |
| 3 | Patell | Parametric | N(0,1) | Yes | Yes | No |
| 4 | Patell + KP | Parametric | N(0,1) | Yes | Yes | Yes |
| 5 | BMP | Parametric | t(T-2-df) | Yes | Yes | No |
| 6 | BMP + KP | Parametric | t(T-2-df) | Yes | Yes | Yes |
| 7 | Corrado | Nonparametric | N(0,1) | Yes | No | No |
| 8 | Corrado-Zivney | Nonparametric | N(0,1) | Yes | No | No |
| 9 | Cowan cumul. rank | Nonparametric | N(0,1) | No | Yes | No |
| 10 | Zivney-Cowan cumul. | Nonparametric | N(0,1) | No | Yes | No |
| 11 | GRANK-T | Nonparametric | t(T-2) | No | Yes | Yes |
| 12 | Generalized sign | Nonparametric | N(0,1) | Yes | Yes | No |
| 13 | Wilcoxon signed-rank | Nonparametric | N(0,1) | Yes | Yes | No |

**Recommended minimum set**: Patell (#3), BMP (#5), Kolari-Pynnonen adjusted
BMP (#6), and generalized sign test (#12). This provides one parametric test,
one event-induced-variance-robust parametric test, one cross-correlation-robust
test, and one nonparametric test. For maximum rigor, add GRANK-T (#11).
