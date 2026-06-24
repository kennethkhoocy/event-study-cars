# Estimation Models for Event Studies

## Notation

- R_it: return of security i on day t (log or simple, depending on `logreturns` setting)
- R_mt: market/index return on day t
- f_kt: return on factor k on day t
- rf_t: risk-free rate on day t
- T_i: number of valid estimation-window observations for firm i
- [esw_lb, esw_ub]: estimation window boundaries (relative trading days)
- [evw_lb, evw_ub]: event window boundaries

## RAW Returns

No model estimated. "Abnormal return" is the raw return itself.

    AR_it = R_it

Use case: baseline diagnostic only. Not recommended for inference since it
ignores systematic risk.

## COMEAN (Constant Mean Model)

Benchmark is the firm's mean return from the estimation window.

    E[R_it] = mu_i = (1/T_i) * sum_{tau in est_window} R_{i,tau}

    AR_it = R_it - mu_i

Under the thin-trading transformation, the regression is:

    cum_return / sqrt(d) = mu * (1/sqrt(d)) + epsilon

Run with `nocons`; the intercept regressor is `1/sqrt(cum_periods)`.

## MA (Market-Adjusted Returns)

Benchmark is the market return. No estimation needed.

    AR_it = R_it - R_mt

Simple but assumes beta = 1 and alpha = 0 for all firms. Useful when factor
data is unavailable or the sample is small.

Under the thin-trading transformation, excess returns
`(R_it - cumulated_R_mt)` are divided by `sqrt(cum_periods)`.

## FM (Factor Model)

The general specification encompasses the market model and multi-factor models.

### Market Model (1 factor)

    R_it = alpha_i + beta_i * R_mt + epsilon_it

Estimated via OLS over the estimation window. Abnormal return:

    AR_it = R_it - (alpha_hat_i + beta_hat_i * R_mt)

### Fama-French 3-Factor Model

    (R_it - rf_t) = alpha_i + b_m * (R_mt - rf_t) + b_s * SMB_t + b_h * HML_t + epsilon_it

Excess returns on LHS and RHS. Abnormal return:

    AR_it = (R_it - rf_t) - (alpha_hat + b_m_hat * (R_mt - rf_t) + b_s_hat * SMB_t + b_h_hat * HML_t)

### Fama-French 5-Factor and Carhart 4-Factor

Same structure with additional factors (RMW, CMA for FF5; MOM for Carhart).
eventstudy2 supports up to 12 additional factors.

### Degrees of Freedom

The `df` parameter equals the number of factors **beyond** the market return:
- Market model: df = 0
- FF3: df = 2 (SMB + HML)
- Carhart: df = 3 (SMB + HML + MOM)
- FF5: df = 4 (SMB + HML + RMW + CMA)

This affects sigma_hat: `sigma_hat = sqrt(SSR / (T - 2 - df))` and the
denominator adjustment in Patell's test.

### Under the Thin-Trading Transformation

All variables are divided by `sqrt(cum_periods)`:

    R_it / sqrt(d) = alpha * (1/sqrt(d)) + beta * R_mt_cum / sqrt(d) + epsilon

Where `R_mt_cum` is the sum of market returns over the `d` days of the
non-trading gap. Run with `nocons`.

The risk-free rate, if specified, is subtracted BEFORE the transformation:
- `R_it <- R_it - rf_t`
- `R_mt <- R_mt - rf_t`

### Log Return Conversion

If input returns are simple (discrete): convert to log returns via
`r = ln(1 + R)` before estimation. This applies to both stock and factor
returns. The conversion ensures additivity of multi-period returns, which is
essential for the thin-trading transformation.

If input returns are already log returns: no conversion needed.

BHAR models are the exception — they use simple (discrete) returns for
compounding.

## BHAR (Buy-and-Hold Abnormal Returns)

For long-horizon event studies. Computes buy-and-hold returns over the event
window.

    BHAR_i = product_{t in window} (1 + R_it) - product_{t in window} (1 + R_mt)

Missing returns are set to zero (holding assumption: no price change on
non-trading days). This is the only model where zero-filling is appropriate.

### Statistical Inference

Standard t-tests are severely biased for BHARs due to positive skewness (the
compounding effect). Use the **skewness-adjusted bootstrapped t-ratio** (Lyon,
Barber, and Tsai 1999):

    t_sa = sqrt(N) * (S + (1/3) * gamma * S^2 + (1/(6N)) * gamma)

where S = BHAR_bar / sigma_BHAR and gamma is the sample skewness coefficient.

eventstudy2 implements this with 1000 bootstrap replications and reports
percentile confidence intervals at 90%, 95%, and 99% levels.

### BHAR_raw

Raw buy-and-hold returns without subtracting a benchmark. The benchmark is
implicitly zero.

    BHAR_raw_i = product_{t in window} (1 + R_it) - 1
