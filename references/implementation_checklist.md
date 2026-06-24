# Implementation Checklist: Common Mistakes in Naive Event-Study Code

The 10 most common gaps found in Python (and other non-Stata) CAR
implementations, derived from auditing naive scripts against Kaspereit's
eventstudy2. Each item describes what goes wrong, why it matters, and the
correct approach.

---

## 1. Zero-Filling Missing Returns

**What goes wrong**: Missing event-window returns are replaced with 0.0 before
computing CARs. This is often done implicitly via `fillna(0)`, `np.nan_to_num`,
or `merge(..., fill_value=0)`.

**Why it matters**: A missing return means the stock did not trade. Setting it
to zero says "the stock had exactly zero return" — which biases CARs toward
zero for illiquid stocks. In emerging markets, this can bias 20-50% of the
sample.

**Correct approach**: Leave missing returns as NaN. When summing ARs to
compute CARs, use `np.nansum` but also count the number of valid (non-NaN)
ARs. If the count is less than the window length, flag the CAR (or set it to
NaN depending on strictness). The only exception is BHAR models, where
zero-filling reflects a "buy and hold" assumption.

---

## 2. No Thin-Trading Adjustment

**What goes wrong**: The regression treats every observed return as a
single-day return, ignoring the fact that returns following non-trading gaps
span multiple days.

**Why it matters**: Multi-period returns have higher variance than single-day
returns. Treating them identically produces heteroscedastic errors in OLS,
biasing coefficient standard errors and all downstream test statistics. For
thinly-traded markets, this can invalidate the entire analysis.

**Correct approach**: Apply the Maynes-Rumsey (1993) trade-to-trade
transformation. Compute `cum_periods` for each observation, divide all
variables by `sqrt(cum_periods)`, and run OLS with `nocons`. See
`references/thin_trading.md` for the complete procedure and Python code.

---

## 3. No Prediction Error Correction (STDF)

**What goes wrong**: Abnormal returns are standardized by dividing by
`sigma_hat` only (the OLS residual standard deviation), ignoring the
additional uncertainty from estimating the model coefficients.

**Why it matters**: The forecast variance at event time t is:

    Var(AR_t) = sigma^2 * (1 + x'_t (X'X)^{-1} x_t)

The second term (the Theil correction, or hat-value term) can be substantial
when event-window factor values deviate from estimation-window means. Ignoring
it understates the SAR denominator, inflating test statistics.

**Correct approach**: Compute STDF_it = sigma_hat * sqrt(1 + h_t) for each
event-window observation, where h_t = x'_t (X'X)^{-1} x_t. Use STDF (not
sigma_hat) as the denominator for standardized abnormal returns in the Patell
and BMP tests.

---

## 4. No IPO/Delisting Guard

**What goes wrong**: CARs are computed for stocks whose first observed return
date is after the event-window start, or whose last observed return date is
before the event-window end.

**Why it matters**: These stocks have incomplete coverage of the event window
due to IPO or delisting. Their CARs are based on a subset of the intended
window and suffer from survivorship/selection bias. Including them contaminates
the cross-sectional analysis.

**Correct approach**: For each firm, determine the first and last dates with
valid return observations (the IPO and delisting dates). If the IPO date
falls inside the event window (after evw_lb) or the delisting date falls
inside the event window (before evw_ub), exclude the firm-event and log the
exclusion reason.

---

## 5. No CAR Boundary Contamination Check

**What goes wrong**: When a stock doesn't trade on the first day of a CAR
window but did trade on a later day, the later day's return spans back into
the pre-window period. This multi-period return is included in the CAR without
checking whether it bleeds outside the intended window.

**Why it matters**: The CAR then contains price changes from dates outside the
window boundaries, making it a noisy estimate of the true within-window
abnormal return.

**Correct approach**: Check `cum_periods` at the first and last days of each
CAR window. If the first day has `cum_periods > 1`, the return spans before
the window start — set CAR = NaN. If the last day has a missing AR, the
window boundary is not covered — set CAR = NaN.

---

## 6. No Event-Date Shift Limit

**What goes wrong**: Event dates that fall on non-trading days (weekends,
holidays) are silently mapped to the next trading day, potentially many days
later.

**Why it matters**: If an event date of December 24 is shifted to January 2
(9 calendar days later), the event window is misaligned. The "event day"
return now captures 9 days of price changes, most of which are unrelated to
the event.

**Correct approach**: Set a maximum calendar-day shift (default: 3 days). If
the nearest trading day is more than `max_shift` calendar days after the event
date, exclude the event and log the reason. eventstudy2 uses `shift(3)` by
default.

---

## 7. No Exclusion Reason Logging

**What goes wrong**: Firm-events are silently dropped or have CARs set to NaN
without recording why.

**Why it matters**: Without exclusion reasons, it's impossible to diagnose
sample attrition, verify that exclusions are appropriate, or report them in
the paper's methodology section. Reviewers and replicators need to know why
N dropped from 1000 to 650.

**Correct approach**: For every firm-event, record the exclusion reason (or
"included") in a dedicated column. Categories: `insufficient_est_obs`,
`insufficient_evt_obs`, `ipo_in_window`, `delisting_in_window`,
`event_off_dateline`, `boundary_contamination`, `event_date_shift_exceeded`.
Produce a summary table of exclusion counts.

---

## 8. No Test Statistics

**What goes wrong**: CARs are computed and reported as means with standard
t-tests or no statistical tests at all.

**Why it matters**: A simple cross-sectional t-test on CARs assumes
homoscedasticity and independence, both of which are typically violated in
event studies. Without proper test statistics, inference is unreliable.
Published event studies require at minimum one parametric and one
nonparametric test.

**Correct approach**: Compute at least: Patell (1976) standardized residuals
test, BMP (Boehmer et al. 1991) for event-induced variance robustness,
Kolari-Pynnonen (2010) adjusted BMP for cross-correlation, and the generalized
sign test (Cowan 1992) as a nonparametric alternative. Report test statistics
and p-values for each CAR window.

---

## 9. Log vs. Simple Return Mismatch

**What goes wrong**: Stock returns are in log form (or simple form) while
factor returns are in the other form. Or returns are left in simple form when
the thin-trading transformation requires log returns.

**Why it matters**: Jensen's inequality: E[ln(1+R)] != ln(1+E[R]). Mixing log
and simple returns between LHS and RHS of the market model introduces a
systematic bias. The thin-trading transformation (summing returns over
non-trading gaps) is only valid for log returns, since log returns are
additive over time.

**Correct approach**: Convert all returns to log form via `r = ln(1 + R)`
before estimation (for RAW, COMEAN, MA, FM models). Ensure both stock and
factor returns use the same convention. The only exception is BHAR, which uses
simple returns for compounding: `product(1 + R_t)`.

---

## 10. No Dateline Construction

**What goes wrong**: Event windows are defined in calendar time (e.g., "5
business days before the event") using generic business-day calendars, rather
than the actual trading calendar derived from the returns data.

**Why it matters**: Generic business-day calendars don't account for
market-specific holidays, trading halts, or data gaps. An event window defined
as "[-5, +5] business days" might include days when the market was closed,
leading to misaligned windows and missing data that could have been avoided.

**Correct approach**: Build the dateline from the actual returns data: the set
of dates on which securities (or the market) have valid returns. Define all
windows in **dateline time** (relative trading days). Optionally filter the
dateline with a threshold to drop thinly-populated dates (e.g., holidays where
<20% of stocks traded). Map event dates to the dateline, not to a generic
calendar.

---

## Quick Self-Check

Before finalizing a CAR pipeline, verify:

- [ ] Missing event-window returns are NaN, not zero
- [ ] Thin-trading adjustment is applied (or consciously disabled for liquid samples)
- [ ] STDF (not just sigma_hat) is used in SAR denominators
- [ ] IPO/delisting dates are checked against the event window
- [ ] Boundary contamination (cum_periods > 1 at window edges) is guarded
- [ ] Event-date shift has a maximum (default: 3 calendar days)
- [ ] Every excluded firm-event has a logged reason
- [ ] At least 4 test statistics are computed (parametric + nonparametric)
- [ ] Stock and factor returns are both in log form (or both simple)
- [ ] The dateline is built from actual returns data, not a generic calendar
