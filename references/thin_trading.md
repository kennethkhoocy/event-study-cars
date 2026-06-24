# Thin-Trading Adjustment: Maynes-Rumsey (1993) Trade-to-Trade Transformation

## Why Thin Trading Matters

When a stock does not trade on a given day, its return is unobserved. Naive
approaches either skip that day (losing information) or set the return to zero
(biasing CARs toward zero). The trade-to-trade transformation handles this
correctly by recognizing that the return observed on the next trading day is a
**multi-period return** spanning the non-trading gap.

This adjustment is especially important for:
- Emerging market stocks
- Small-cap and micro-cap samples
- Markets with frequent trading halts
- Any sample where >5% of firm-day observations have missing returns

It should be the **default** for any event study unless the sample is known to
be highly liquid (e.g., S&P 500 constituents).

## Constructing cum_periods

`cum_periods` counts how many calendar trading days are spanned by each
return observation.

### Algorithm

For each stock, sorted by date:

1. If a day has a valid return and the previous day also had a valid return:
   `cum_periods = 1` (normal single-day return).

2. If a day has a valid return but the previous day(s) had missing returns:
   `cum_periods = 1 + number_of_consecutive_missing_days_before_this_trade`.
   The return on this day spans `cum_periods` trading days.

3. Days with missing returns: `cum_periods` accumulates (though these days
   are ultimately dropped from the regression since their return is missing).

### Thin-trading threshold (optional)

A return can be flagged as "thinly traded" even if observed, based on dollar
volume:

    if price_t * volume_t < threshold: set return_t = missing

This reclassifies low-activity days as non-trading days, causing their returns
to be absorbed into the next genuine trade.

## The Transformation

Given a day t where the stock last traded on day t-d (so `cum_periods = d`):

### Stock returns
The observed return R_it already spans d days (it's computed from prices:
R = P_t / P_{t-d} - 1, or in logs: r = ln(P_t) - ln(P_{t-d})).

Transform:

    y*_it = R_it / sqrt(d)

### Factor returns
Market/factor returns exist every day, so they must be **cumulated** over the
gap period first:

    R*_m,t = sum_{s=t-d+1}^{t} R_m,s    (cumulated factor return over d days)

Transform:

    x*_m,t = R*_m,t / sqrt(d)

### Intercept regressor
Replace the standard constant (1) with:

    w_t = 1 / sqrt(d)

## Regression Specification

Run OLS with **no constant** (`nocons` in Stata):

    y*_it = alpha * w_t + beta * x*_m,t + [beta_k * x*_k,t ...] + epsilon*_t

The `nocons` is **essential**. The intercept regressor `w_t = 1/sqrt(d)`
already serves as the intercept. Adding a standard constant would double-count
the intercept and produce incorrect estimates.

## Why This Works

Under the null hypothesis (market model holds each day), the multi-period
return satisfies:

    R_it = alpha * d + beta * sum R_m,s + eta_t

where eta_t ~ N(0, d * sigma^2) because it's the sum of d independent
daily innovations.

Dividing by sqrt(d):

    R_it/sqrt(d) = alpha * d/sqrt(d) + beta * (sum R_m,s)/sqrt(d) + eta_t/sqrt(d)

Wait — this gives alpha * sqrt(d), not alpha / sqrt(d). The correct
transformation requires:

    R_it/sqrt(d) = alpha * (1/sqrt(d)) + beta * (sum R_m,s)/sqrt(d) + epsilon*_t

This works because the OLS intercept regressor is `1/sqrt(d)`, so:

    y* = alpha * w + beta * x* + epsilon*

where epsilon* = eta/sqrt(d) ~ N(0, sigma^2). All observations now have the
same error variance, making OLS efficient (this is essentially a GLS
correction).

## STDF Under the Transformation

The prediction error variance for observation t is:

    STDF^2_it = sigma_hat^2 * (1 + x'_t (X'X)^{-1} x_t)

where `x_t` and `X` are in the **un-transformed** (original) scale.

In eventstudy2, this is computed by:
1. Estimating OLS on the transformed data
2. Temporarily un-transforming the regressors (multiplying by sqrt(d))
3. Computing `predict, stdf` on the un-transformed values
4. Restoring the transformed values

In Python, compute STDF directly:

```python
# After OLS on transformed data:
# X_est: estimation-window design matrix (transformed)
# sigma_hat: sqrt(SSR / (T - 2 - df))

# For event-window observation t (un-transformed regressors):
# x_t_untrans = [1.0, R_m_cumulated, factor1_cumulated, ...]

XtX_inv = np.linalg.inv(X_est.T @ X_est)
h_t = x_t_untrans @ XtX_inv @ x_t_untrans
STDF_t = sigma_hat * np.sqrt(1 + h_t)
```

Note: `X_est` uses the transformed regressors (divided by sqrt(d)), but
`x_t_untrans` uses the un-transformed values (the raw cumulated factor
returns and intercept = 1).

## Boundary Contamination Guard

When computing CARs over a window `[lb, ub]`:

**First day check**: If `cum_periods > 1` at day `lb`, the return on that day
includes price changes from **before** the window start. The CAR would be
contaminated. Set CAR = NaN for this firm-event.

**Last day check**: If the AR at day `ub` is missing (NaN), the firm-event
lacks coverage at the window boundary. Set CAR = NaN.

**For AAR (day-by-day analysis)**: Any observation with `cum_periods > 1` has
its AR set to NaN, because the multi-period return cannot be attributed to a
single event day.

## Self-Contained Python Example

```python
import numpy as np

def apply_thin_trading_transform(
    stock_returns: np.ndarray,    # shape (T,), NaN for non-trading days
    factor_returns: np.ndarray,   # shape (T, K), always observed
) -> tuple:
    """
    Apply Maynes-Rumsey (1993) trade-to-trade transformation.

    Returns:
        y_star:      transformed stock returns (NaN for non-trading days)
        X_star:      transformed design matrix [intercept, factors]
        cum_periods: array of period counts
    """
    T, K = factor_returns.shape
    cum_periods = np.ones(T, dtype=int)
    factor_cumulated = factor_returns.copy()

    for t in range(1, T):
        if np.isnan(stock_returns[t - 1]):
            # Previous day was a non-trading day: accumulate gap
            cum_periods[t] = cum_periods[t - 1] + 1
            factor_cumulated[t] += factor_cumulated[t - 1]

    # Build transformed variables
    sqrt_d = np.sqrt(cum_periods).astype(float)

    # Stock returns: already multi-period, just divide by sqrt(d)
    y_star = stock_returns / sqrt_d

    # Intercept regressor
    intercept = 1.0 / sqrt_d

    # Factor regressors: cumulated and divided by sqrt(d)
    X_factors_star = factor_cumulated / sqrt_d[:, None]

    # Design matrix: [intercept, factor1, factor2, ...]
    X_star = np.column_stack([intercept, X_factors_star])

    return y_star, X_star, cum_periods


def estimate_with_thin_trading(
    y_star: np.ndarray,
    X_star: np.ndarray,
    cum_periods: np.ndarray,
    est_mask: np.ndarray,          # boolean mask for estimation window
    min_obs: int = 120,
) -> dict | None:
    """
    Estimate OLS on transformed data, compute STDF on un-transformed data.

    Returns dict with: alpha, betas, sigma_hat, residuals, stdf (per obs).
    """
    y_est = y_star[est_mask]
    X_est = X_star[est_mask, :]

    # Drop rows where y is NaN
    valid = ~np.isnan(y_est)
    y_est = y_est[valid]
    X_est = X_est[valid, :]

    if len(y_est) < min_obs:
        return None

    # OLS with nocons (X already includes the intercept regressor)
    coef, _, _, _ = np.linalg.lstsq(X_est, y_est, rcond=None)

    residuals = y_est - X_est @ coef
    T_est = len(y_est)
    k = X_est.shape[1]  # number of regressors
    sigma_hat = np.sqrt(np.sum(residuals**2) / (T_est - k))

    # Compute STDF for ALL observations (un-transformed regressors)
    XtX_inv = np.linalg.inv(X_est.T @ X_est)
    n_total = len(y_star)

    stdf = np.full(n_total, np.nan)
    for t in range(n_total):
        # Un-transform: X_star[t] = [1/sqrt(d), cumulated/sqrt(d)]
        # so X_star[t] * sqrt(d) = [1, cumulated_factor_returns]
        x_untrans = X_star[t, :] * np.sqrt(cum_periods[t])
        h_t = x_untrans @ XtX_inv @ x_untrans
        stdf[t] = sigma_hat * np.sqrt(1 + h_t)

    # Predicted returns (in un-transformed scale)
    predicted = np.full(n_total, np.nan)
    for t in range(n_total):
        x_untrans = X_star[t, :] * np.sqrt(cum_periods[t])
        predicted[t] = x_untrans @ coef

    return {
        "alpha": coef[0],
        "betas": coef[1:],
        "sigma_hat": sigma_hat,
        "nobs": T_est,
        "r2": 1 - np.sum(residuals**2) / np.sum(
            (y_est - np.mean(y_est))**2
        ) if np.sum((y_est - np.mean(y_est))**2) > 0 else 0.0,
        "stdf": stdf,
        "predicted": predicted,
    }
```

## References

- Maynes, E., Rumsey, J. 1993. Conducting event studies with thinly traded
  stocks. *Journal of Banking and Finance* 17(1), 145-157.
- Theil, H. 1971. *Principles of Econometrics*. Wiley.
