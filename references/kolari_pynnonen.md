# Kolari-Pynnonen Cross-Correlation Adjustment

## Overview

Standard event-study test statistics (Patell, BMP) assume cross-sectional
independence of abnormal returns. When events cluster in calendar time (e.g.,
regulatory announcements affecting all firms on the same date), abnormal
returns are cross-correlated, inflating test statistics and producing
spuriously low p-values.

Kolari and Pynnonen (2010) propose a simple multiplicative adjustment factor
`ADJ` that corrects for this cross-correlation. Kolari and Pynnonen (2011)
extend the approach to a nonparametric rank-based test (GRANK-T).

## The ADJ Factor (Kolari-Pynnonen 2010)

### Step 1: Standardize Estimation-Window Residuals

For each firm i and estimation-window day tau:

    V_i,tau = AR_i,tau / STDF_i,tau

where STDF_i,tau is the standard deviation of forecast (Theil prediction error
correction) at estimation-window time tau. This produces standardized
residuals with approximately unit variance.

Note: The STDF values used here are from the **estimation window**, not the
event window.

### Step 2: Compute Pairwise Correlations

For all N(N-1) unique pairs of firms (i, j) where i != j:

    rho_ij = Corr(V_i, V_j)

computed as the Pearson correlation of the two firms' standardized residual
time series over the estimation window.

### Step 3: Compute Mean Cross-Correlation

    r_bar = (1 / (N*(N-1))) * sum_{i != j} rho_ij

This is the mean of all N(N-1) pairwise correlations (excluding the diagonal
i = j which would be 1.0).

### Step 4: Compute the Adjustment Factor

    ADJ = sqrt( (1 - r_bar) / (1 + (N-1) * r_bar) )

Properties:
- When r_bar = 0 (no cross-correlation): ADJ = 1, no adjustment.
- When r_bar > 0 (positive cross-correlation): ADJ < 1, shrinks the test
  statistic toward zero, making inference more conservative.
- When r_bar < 0 (negative cross-correlation): ADJ > 1, amplifies the test
  statistic.

### Application

Multiply any standard test statistic by ADJ:

    Z_adjusted = Z_original * ADJ

This applies to: Patell Z, BMP t, and the GRANK-T procedure.

## KOLARIBLK: Block-Diagonal Variant

When the sample has a natural block structure (e.g., firms grouped by country
or industry), cross-correlation may exist within blocks but not across blocks.

The KOLARIBLK variant:
1. Only computes pairwise correlations for firms **in the same block**
   (KLBK[i] == KLBK[j]).
2. Sets rho_ij = 0 for pairs in different blocks.
3. Computes r_bar and ADJ using the same formulas.

This gives a less aggressive adjustment when the cross-correlation is driven
by block-level clustering rather than market-wide dependence.

## Computational Complexity

The pairwise correlation computation is O(N^2 * T_est):
- N^2 pairs, each requiring O(T_est) for the correlation.

For N < 500 firms, this is fast (a few seconds). For N > 500:
- Consider the KOLARIBLK variant to reduce the effective N.
- For very large N (>2000), consider subsampling: randomly select ~500 firms,
  compute r_bar from the subsample, and use it as an estimate.
- Shrinkage estimators (Ledoit-Wolf) can also be used to regularize the
  correlation matrix, though eventstudy2 does not implement this.

## Python Implementation

```python
import numpy as np

def compute_kolari_adj(
    ar_est: np.ndarray,      # (T_est, N_firms) estimation-window ARs
    stdf_est: np.ndarray,    # (T_est, N_firms) estimation-window STDFs
    df: int = 0,             # additional factors
    block: np.ndarray = None  # (N_firms,) block assignments (optional)
) -> float:
    """
    Compute the Kolari-Pynnonen (2010) ADJ factor.

    Parameters
    ----------
    ar_est : estimation-window abnormal returns
    stdf_est : estimation-window STDFs
    df : number of additional factors beyond market
    block : optional block assignments for KOLARIBLK variant

    Returns
    -------
    ADJ : the adjustment factor
    """
    N = ar_est.shape[1]

    # Standardize residuals
    V = ar_est / stdf_est  # (T_est, N_firms)

    # Compute pairwise correlations
    correlations = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if block is not None and block[i] != block[j]:
                correlations.append(0.0)
                continue
            vi = V[:, i]
            vj = V[:, j]
            # Drop pairs where either is NaN
            valid = ~(np.isnan(vi) | np.isnan(vj))
            if valid.sum() < 3:
                continue
            rho = np.corrcoef(vi[valid], vj[valid])[0, 1]
            if not np.isnan(rho):
                correlations.append(rho)

    r_bar = np.mean(correlations)
    ADJ = np.sqrt((1 - r_bar) / (1 + (N - 1) * r_bar))

    return ADJ
```

**Performance note**: For large N, the nested loop is slow. Use pairwise-complete
correlation:

```python
# Vectorized version (much faster for large N)
# Use pairwise-complete observations to handle missing data correctly
import numpy as np

def _pairwise_complete_corr(V):
    """Correlation matrix using only pairwise-complete observations."""
    N = V.shape[1]
    corr = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(i + 1, N):
            valid = ~(np.isnan(V[:, i]) | np.isnan(V[:, j]))
            if valid.sum() >= 3:
                corr[i, j] = corr[j, i] = np.corrcoef(
                    V[valid, i], V[valid, j]
                )[0, 1]
    return corr

corr_matrix = _pairwise_complete_corr(V)
r_bar = np.nanmean(corr_matrix)
ADJ = np.sqrt((1 - r_bar) / (1 + (N - 1) * r_bar))
```

> For samples where all firms have complete estimation-window data (no missing
> days within the window), the faster `np.corrcoef(V.T)` can be used directly.
> The pairwise-complete version above is the safe default.

## GRANK-T (Kolari-Pynnonen 2011)

The GRANK-T is a nonparametric test for CAARs that combines:
- Time-series standardization (SARs/SCARs)
- Cross-sectional standardization
- Kolari-Pynnonen cross-correlation adjustment
- Rank transformation
- A t-test with robust degrees of freedom

### Procedure

#### Step 1: Compute Cross-Correlation-Adjusted SCARs

From the CAAR test statistic computation:
- SCAR_i = cumulative sum of SAR_is / sqrt(L_s) (the Patell CAAR quantity)

Adjust for cross-correlation and cross-sectional dispersion:

    SCAR*_i = (SCAR_i * ADJ) / sd_cs(SCAR_i)

where sd_cs is the cross-sectional standard deviation of SCARs across firms.

Take the **final row** (full CAR window) of SCAR*: this gives one value per
firm.

#### Step 2: Compute Estimation-Window SARs

    SAR_i,tau = AR_i,tau / sigma_hat_i

for each firm i and estimation-window day tau.

#### Step 3: Stack and Rank

Create a combined matrix:

    TOTAL_i = [SAR_est_i ; SCAR*_event_i]

where the estimation-window SARs are stacked on top and the single
SCAR*_event value is appended at the bottom.

Rank within each firm's combined vector:

    rank_i = rank(TOTAL_i)

Handle missing values by excluding them from the ranking (assign NaN rank).

#### Step 4: Transform to Uniform Scale

    U_it = rank_it / (T_total + 1) - 0.5

where T_total is the count of non-missing values in the combined vector.

#### Step 5: Aggregate

    U_bar_t = mean_i(U_it)    for each time period t

The estimation-window U_bar values serve as the null distribution.

#### Step 6: Compute Standard Error

    S_U = sqrt( (1/T_est) * sum_{est window} (N_est/N_event * U_bar_t^2) )

where N_est is the number of firms at each estimation-window time point and
N_event is the number of firms at the event time.

#### Step 7: Z and t Statistics

    Z_GRANK = U_bar_event / S_U

    t_GRANK = Z_GRANK * sqrt( (T_est - 2) / (T_est - 1 - Z_GRANK^2) )

#### Step 8: p-value

    p = 2 * P(|T| > |t_GRANK|)    where T ~ t(T_est - 2)

### Why GRANK-T is Preferred

- Nonparametric: does not assume normality of returns
- Accounts for event-induced variance changes (via SARs)
- Accounts for cross-correlation (via ADJ)
- The rank transformation is robust to outliers
- The t(T-2) distribution provides better finite-sample performance than
  the normal approximation

Kolari and Pynnonen (2011) show GRANK-T has the best size and power properties
among the tests they compare, especially under cross-correlation and
event-induced volatility.

## References

- Kolari, J. W., Pynnonen, S. 2010. Event study testing with cross-sectional
  correlation of abnormal returns. *Review of Financial Studies* 23(11),
  3996-4025.
- Kolari, J. W., Pynnonen, S. 2011. Nonparametric rank tests for event
  studies. *Journal of Empirical Finance* 18(5), 953-971.
