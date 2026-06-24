#!/usr/bin/env python3
"""
eventstudy.py - generic, out-of-the-box event-study CAR engine.

A Python replication of Thomas Kaspereit's eventstudy2 (Stata, v3.2b). Computes
abnormal returns, cumulative abnormal returns over arbitrary windows, and a panel
of test statistics, faithfully reproducing eventstudy2's numerics:

  * dateline construction (market/factor file for FM/MA; security returns for RAW/COMEAN)
  * event-date mapping to the nearest dateline day on/after the event (shift guard)
  * per-security expansion over [IPO, DEL] on the dateline
  * Maynes-Rumsey (1993) trade-to-trade thin-trading transform (on by default)
  * factor-model OLS with nocons + 1/sqrt(cum_periods) intercept regressor
  * Theil (1971) standard deviation of forecast (STDF)
  * AR = return - predicted (no zero-fill of missing event-window returns)
  * CAR accumulation with eventstudy2's boundary-contamination guards
  * CAAR test statistics: cross-sectional t (Serra), Brown-Warner CDA, Patell,
    Patell+Kolari-Pynnonen, Boehmer (BMP), Kolari (BMP+KP), generalized sign,
    Wilcoxon

It is GENERIC: column names, model, windows, thin-trading, and log handling are
all CLI flags with sensible defaults (eventstudy2 defaults). No project-specific
paths, identifiers, or data are baked in.

Requires numpy, pandas, scipy. Parquet I/O additionally needs pyarrow or polars.

  python eventstudy.py --selftest                 # synthetic-data self-check, no inputs
  python eventstudy.py --returns r.csv --market m.csv --events e.csv \
      --car-windows "-1,1;-5,5;-10,10" --model FM --out-dir out/

Author of the replicated methodology: Thomas Kaspereit (eventstudy2). This file
is an independent reimplementation, not a port of his Stata/Mata source.
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # quiet pandas copy-on-write FutureWarnings

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist


# ----------------------------------------------------------------------------- IO
def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        try:
            import polars as pl
            return pl.read_parquet(p).to_pandas()
        except ImportError:
            return pd.read_parquet(p)
    return pd.read_csv(p)


def write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() in (".parquet", ".pq"):
        try:
            import polars as pl
            pl.from_pandas(df).write_parquet(path)
            return
        except ImportError:
            df.to_parquet(path, index=False)
            return
    df.to_csv(path, index=False)


# ------------------------------------------------------------------ core helpers
def cum_periods(ret: np.ndarray) -> np.ndarray:
    """eventstudy2 cum_periods over a security's contiguous dateline rows.

    thinvar_t = 1 if ret[t-1] missing and ret[t] present (first trade after a gap).
    cum[t] = 1; cum[t] += cum[t-1] if (ret[t] missing OR thinvar_t) AND ret[t-1] missing.
    Mirrors eventstudy2.ado lines 451-454.
    """
    n = len(ret)
    cp = np.ones(n)
    miss = ~np.isfinite(ret)
    for t in range(1, n):
        prev_missing = miss[t - 1]
        if not prev_missing:
            continue
        thinvar = (not miss[t])  # ret[t] present and prev missing
        if miss[t] or thinvar:
            cp[t] = cp[t] + cp[t - 1]
    return cp


def _ols(y: np.ndarray, X: np.ndarray):
    beta, _r, _rk, _sv = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    ssr = float(resid @ resid)
    rmse = np.sqrt(ssr / (n - k)) if n > k else np.nan
    XtXi = np.linalg.inv(X.T @ X)
    return beta, rmse, XtXi, ssr


# ------------------------------------------------------------------- the engine
class Result:
    def __init__(self, ar, car, teststats):
        self.ar = ar
        self.car = car
        self.teststats = teststats


def run_event_study(
    returns: pd.DataFrame,
    market: pd.DataFrame,
    events: pd.DataFrame,
    *,
    id_col="firm_id",
    date_col="date",
    ret_col="ret",
    event_id_col=None,
    event_date_col="event_date",
    mkt_col="mkt",
    factor_cols=(),
    model="FM",
    car_windows=((-1, 1), (-5, 5)),
    evwlb=None, evwub=None,
    eswlb=-270, eswub=-21,
    minesw=30, minevw=1,
    shift=3,
    thin_trading=True,
    log_returns=True,           # convert simple -> ln(1+r) (eventstudy2 default for RAW/MA/FM)
    arfillevent=False,          # eventstudy2 NEVER-zero-fill default is off
    nokolari=False,
) -> Result:
    factor_cols = list(factor_cols)
    df = 0 if model != "FM" else len(factor_cols)           # extra factors beyond market
    rhs_cols = ([mkt_col] + factor_cols) if model in ("FM", "MA") else []

    cw = [tuple(w) for w in car_windows]
    if evwlb is None:
        evwlb = min(w[0] for w in cw)
    if evwub is None:
        evwub = max(w[1] for w in cw)
    cw = [(max(lb, evwlb), min(ub, evwub)) for lb, ub in cw]

    # ---- dateline: from the market file for FM/MA, from security returns for
    #      RAW/COMEAN (eventstudy2.ado lines 199-206)
    if model in ("FM", "MA"):
        if market is None:
            raise ValueError(f"model {model} requires a market/factor file")
        md = market.copy()
        md[date_col] = pd.to_datetime(md[date_col])
        md = md.dropna(subset=[mkt_col]).sort_values(date_col).drop_duplicates(date_col)
        md = md.reset_index(drop=True)
        dateline = md[date_col].to_numpy()
        mkt_vals = {c: md[c].to_numpy(float) for c in ([mkt_col] + factor_cols) if c in md.columns}
    else:                                   # RAW, COMEAN: dateline from returns
        rd = returns[[date_col, ret_col]].copy()
        rd[date_col] = pd.to_datetime(rd[date_col])
        rd = rd.dropna(subset=[ret_col]).sort_values(date_col).drop_duplicates(date_col)
        dateline = rd[date_col].to_numpy()
        mkt_vals = {}
    D = len(dateline)
    didx = pd.Series(np.arange(D), index=pd.DatetimeIndex(dateline))
    start, end = dateline[0], dateline[-1]

    # ---- returns panel indexed by (firm -> per-date return)
    rr = returns[[id_col, date_col, ret_col]].copy()
    rr[id_col] = rr[id_col].astype(str)
    rr[date_col] = pd.to_datetime(rr[date_col])
    rr["di"] = rr[date_col].map(didx)
    rr = rr.dropna(subset=["di"])
    rr["di"] = rr["di"].astype(int)
    rr = rr.sort_values([id_col, "di"])
    ret_by_firm = {gv: (g["di"].to_numpy(), g[ret_col].to_numpy(float))
                   for gv, g in rr.groupby(id_col, sort=False)}

    # ---- events
    ev = events.copy()
    ev[id_col] = ev[id_col].astype(str)
    ev[event_date_col] = pd.to_datetime(ev[event_date_col])
    if event_id_col is None:
        ev = ev.reset_index(drop=True)
        ev["__eid"] = ev.groupby(id_col).cumcount()
        event_id_col = "__eid"

    ar_rows = []
    car_records = []
    # per-firm-event arrays kept for the test statistics, grouped by car window key
    keep = {w: {"car": [], "si": [], "T": [], "vit": [], "are": [], "stdfe": [],
                "mr_est": [], "mre": []} for w in cw}
    pooled_ar_est = []   # for the generalized-sign p_hat (pooled across firms)
    # estimation-window grids (aligned to [eswlb..eswub]) for the global KP ADJ
    est_days = np.arange(eswlb, eswub + 1)
    glob = {"ar_grid": [], "stdf_grid": [], "si": [], "T": []}

    for _, e in ev.iterrows():
        gv = e[id_col]
        eid = e[event_id_col]
        ev_date = e[event_date_col]
        rec = {id_col: gv, "event_id": eid, "event_date": ev_date,
               "exclusion_reason": "", "nobs": np.nan, "sigma_hat": np.nan}
        for w in cw:
            rec[f"CAR[{w[0]},{w[1]}]"] = np.nan
            rec[f"n_ar[{w[0]},{w[1]}]"] = 0

        # event-date mapping (nearest dateline day on/after, within shift cal. days)
        if ev_date < start or ev_date > end:
            rec["exclusion_reason"] = "event_off_dateline"
            car_records.append(rec); continue
        after = dateline[dateline >= np.datetime64(ev_date)]
        if len(after) == 0 or (after[0] - np.datetime64(ev_date)) / np.timedelta64(1, "D") > shift:
            rec["exclusion_reason"] = "event_off_dateline"
            car_records.append(rec); continue
        E = int(didx.loc[pd.Timestamp(after[0])])

        if gv not in ret_by_firm:
            rec["exclusion_reason"] = "no_returns"
            car_records.append(rec); continue
        fdi, fret = ret_by_firm[gv]
        # [IPO, DEL] = first/last dateline index with a non-missing return
        valid = np.isfinite(fret)
        if not valid.any():
            rec["exclusion_reason"] = "no_returns"
            car_records.append(rec); continue
        ipo, dele = fdi[valid].min(), fdi[valid].max()
        idx = np.arange(ipo, dele + 1)                  # contiguous dateline rows
        ret_full = np.full(len(idx), np.nan)
        pos = {d: k for k, d in enumerate(idx)}
        for d, r in zip(fdi, fret):
            if ipo <= d <= dele:
                ret_full[pos[d]] = r
        dif = idx - E

        rhs_full = {c: mkt_vals[c][idx] for c in rhs_cols}

        # log transform (eventstudy2 default for RAW/COMEAN/MA/FM); cum_periods is
        # computed on the pre-log returns (missingness), factor availability on the
        # post-log RHS, matching eventstudy2's ordering (lines 451-454 then 456-475).
        r_log = np.log1p(ret_full) if log_returns else ret_full
        rhs_log = {c: (np.log1p(rhs_full[c]) if log_returns else rhs_full[c]) for c in rhs_cols}

        factor_avail = np.ones(len(idx), bool)
        for c in rhs_cols:
            factor_avail &= np.isfinite(rhs_log[c])

        cp = cum_periods(ret_full) if thin_trading else np.ones(len(idx))
        sq = np.sqrt(cp)

        # cumulate factors over gaps, then divide by sqrt(cp); returns only divided
        def cumulate(v):
            vc = v.copy()
            for t in range(1, len(vc)):
                if cp[t] > 1 and np.isfinite(vc[t - 1]):
                    vc[t] = vc[t] + vc[t - 1]
            return vc
        rhs_cumlevel = {c: cumulate(rhs_log[c]) for c in rhs_cols}   # un-transformed level
        rhs_trans = {c: rhs_cumlevel[c] / sq for c in rhs_cols}
        r_trans = r_log / sq
        inter_trans = 1.0 / sq

        est = (dif >= eswlb) & (dif <= eswub) & np.isfinite(r_trans) & factor_avail
        evt = (dif >= evwlb) & (dif <= evwub)
        sec_ok = np.isfinite(r_log)
        mkt_ok = factor_avail if model in ("FM", "MA") else np.ones(len(idx), bool)
        n_est = int((est & sec_ok).sum())                      # count_est_obsWithSecAndMKT
        n_evt = int((evt & sec_ok & mkt_ok).sum())             # count_event_obsWithSecAndMKT

        # exclusions
        ipo_dif = dif[idx == ipo][0] if ipo in idx else None
        del_dif = dif[idx == dele][0] if dele in idx else None
        if n_est < minesw:
            rec["exclusion_reason"] = "insufficient_est_obs"
            car_records.append(rec); continue
        if n_evt < minevw:
            rec["exclusion_reason"] = "insufficient_evt_obs"
            car_records.append(rec); continue
        if (ipo - E) > evwlb or (dele - E) < evwub:
            rec["exclusion_reason"] = "ipo_or_delisting_in_window"
            car_records.append(rec); continue

        # ---- OLS on transformed estimation-window data (nocons)
        try:
            if model == "FM":
                Xcols = [inter_trans] + [rhs_trans[c] for c in rhs_cols]
                Xe = np.column_stack([c[est] for c in Xcols])
                beta, rmse, XtXi, _ = _ols(r_trans[est], Xe)
                x_unt = np.column_stack([np.ones(len(idx))] + [rhs_cumlevel[c] for c in rhs_cols])
                pred = x_unt @ beta
                lev = np.einsum("ij,jk,ik->i", x_unt, XtXi, x_unt)
                stdf = rmse * np.sqrt(1.0 + lev)
                ar = r_log - pred
                betas = beta
            elif model == "COMEAN":                      # constant mean
                Xe = (inter_trans[est])[:, None]
                beta, rmse, XtXi, _ = _ols(r_trans[est], Xe)
                x_unt = np.ones((len(idx), 1))
                pred = x_unt @ beta
                lev = np.einsum("ij,jk,ik->i", x_unt, XtXi, x_unt)
                stdf = rmse * np.sqrt(1.0 + lev)
                ar = r_log - pred
                betas = beta
            elif model == "MA":                          # market-adjusted: predicted = market
                # AR = r - market. eventstudy2 STDF = rmse of `reg ma_cum_returns zero`
                # (transformed market-adjusted residual), constant (no leverage term).
                ar = r_log - rhs_log[mkt_col]
                resid = (r_trans - rhs_trans[mkt_col])[est]
                resid = resid[np.isfinite(resid)]
                rmse_ma = np.sqrt(np.sum(resid ** 2) / len(resid)) if len(resid) else np.nan
                stdf = np.full(len(idx), rmse_ma)
                betas = np.array([0.0, 1.0])
            elif model == "RAW":                         # raw returns, no benchmark
                # eventstudy2 STDF = rmse of `reg cum_returns zero` -> resid = cum_returns
                ar = r_log.copy()
                resid = r_trans[est]
                resid = resid[np.isfinite(resid)]
                rmse_raw = np.sqrt(np.sum(resid ** 2) / len(resid)) if len(resid) else np.nan
                stdf = np.full(len(idx), rmse_raw)
                betas = np.array([0.0])
            else:
                raise ValueError(f"unknown model {model}")
        except np.linalg.LinAlgError:
            rec["exclusion_reason"] = "singular_regressors"
            car_records.append(rec); continue

        if arfillevent:
            fillm = evt & ~np.isfinite(ar)
            ar = ar.copy(); ar[fillm] = 0.0

        # sigma_hat (si) from un-transformed est-window AR, eventstudy2 convention
        ar_est_vals = ar[est & np.isfinite(ar)]
        T_i = len(ar_est_vals)
        si = np.sqrt(np.sum(ar_est_vals ** 2) / (T_i - 2 - df)) if T_i > (2 + df) else np.nan
        rec["nobs"] = T_i
        rec["sigma_hat"] = si
        for k, b in enumerate(betas):
            rec[f"beta{k}"] = b

        # AR rows (event-window, for the AR panel)
        for t in np.where(evt)[0]:
            ar_rows.append({id_col: gv, "event_id": eid, "dif": int(dif[t]),
                            "date": pd.Timestamp(dateline[idx[t]]),
                            "ret": ret_full[t], "AR": ar[t], "STDF": stdf[t],
                            "cum_periods": cp[t]})

        # market (log) returns aligned to dif, for the Serra/Theil cross-sectional t
        mkt_log_full = rhs_log[mkt_col] if mkt_col in rhs_log else np.zeros(len(idx))
        est_pos = np.where(est & np.isfinite(ar))[0]
        mr_est = mkt_log_full[est_pos]

        # ---- per-window CAR with boundary guards + collect test-stat inputs
        for w in cw:
            lb, ub = w
            wm = (dif >= lb) & (dif <= ub)
            wpos = np.where(wm)[0]
            if len(wpos) == 0:
                continue
            are_w = ar[wpos]
            cp_lb = cp[wpos[0]]            # cum_periods at first window day
            ar_ub = ar[wpos[-1]]          # AR at last window day
            n_valid = int(np.sum(np.isfinite(are_w)))
            rec[f"n_ar[{lb},{ub}]"] = n_valid
            if (not arfillevent) and ((not np.isfinite(ar_ub)) or cp_lb > 1):
                continue                  # boundary contamination -> CAR stays NaN
            car_val = float(np.nansum(are_w))
            rec[f"CAR[{lb},{ub}]"] = car_val
            # collect for test stats (only non-excluded firms with valid CAR)
            keep[w]["car"].append(car_val)
            keep[w]["si"].append(si)
            keep[w]["T"].append(T_i)
            keep[w]["are"].append(are_w)
            keep[w]["stdfe"].append(stdf[wpos])
            keep[w]["mre"].append(mkt_log_full[wpos])
            keep[w]["mr_est"].append(mr_est)

        # estimation-window grids aligned to [eswlb..eswub] (global, for KP ADJ)
        ar_grid = np.full(len(est_days), np.nan)
        stdf_grid = np.full(len(est_days), np.nan)
        em = est & np.isfinite(ar)
        gpos = (dif[em] - eswlb).astype(int)
        ok = (gpos >= 0) & (gpos < len(est_days))
        ar_grid[gpos[ok]] = ar[em][ok]
        stdf_grid[gpos[ok]] = stdf[em][ok]
        glob["ar_grid"].append(ar_grid)
        glob["stdf_grid"].append(stdf_grid)
        glob["si"].append(si)
        glob["T"].append(T_i)

        pooled_ar_est.append(ar_est_vals)
        car_records.append(rec)

    car = pd.DataFrame(car_records)
    ar = pd.DataFrame(ar_rows)

    # generalized-sign p_hat: pooled over all firms' estimation-window ARs
    if pooled_ar_est:
        allest = np.concatenate(pooled_ar_est)
        p_hat = np.sum(np.sign(allest)) / (2 * np.sum(np.isfinite(allest))) + 0.5
    else:
        p_hat = 0.5

    # global Kolari-Pynnonen ADJ from estimation-window standardized residuals
    adj = 1.0 if nokolari else _kolari_adj(glob["ar_grid"], glob["stdf_grid"],
                                           np.array(glob["si"]), df)
    teststats = _compute_teststats(cw, keep, df, p_hat, adj, len(est_days),
                                   model in ("RAW", "COMEAN"))
    return Result(ar, car, teststats)


def _compute_teststats(cw, keep, df, p_hat, adj, est_len, nomarket):
    dof_t = est_len - 2 - df                     # eventstudy2 ttail dof = rows(AAR)-(2+df)
    rows = []
    for w in cw:
        lb, ub = w
        K = keep[w]
        N = len(K["car"])
        row = {"window": f"[{lb},{ub}]", "N": N}
        if N < 2:
            rows.append(row); continue
        car = np.array(K["car"])
        si = np.array(K["si"])
        T = np.array(K["T"])
        caar = float(np.mean(car))
        row["CAAR"] = caar

        # Patell SARs (eventstudy2 v3.2b form): Vit = ARE / sqrt(si*STDFE);
        # WiL_i = sum_s Vit_is / sqrt(s) cumulated over present window days
        WiL = np.full(N, np.nan)
        for i in range(N):
            are = K["are"][i]; stdfe = K["stdfe"][i]
            m = np.isfinite(are) & np.isfinite(stdfe)
            vit = are[m] / np.sqrt(si[i] * stdfe[m])
            Ls = np.arange(1, len(vit) + 1)
            WiL[i] = np.sum(vit / np.sqrt(Ls))
        varadj = (T - 2 - df) / (T - 4 - df)

        # --- Patell + KP
        z_pat = np.nansum(WiL) / np.sqrt(np.nansum(varadj))
        row["Patell"] = z_pat
        row["p_Patell"] = 2 * norm.sf(abs(z_pat))
        row["PatellADJ"] = z_pat * adj
        row["p_PatellADJ"] = 2 * norm.sf(abs(z_pat * adj))

        # --- Boehmer (BMP) + Kolari (BMP*ADJ)
        bmp = np.mean(WiL) / (np.std(WiL, ddof=1) / np.sqrt(N))
        row["Boehmer"] = bmp
        row["p_Boehmer"] = 2 * t_dist.sf(abs(bmp), dof_t)
        row["Kolari"] = bmp * adj
        row["p_Kolari"] = 2 * t_dist.sf(abs(bmp * adj), dof_t)
        row["KP_ADJ"] = adj

        # --- cross-sectional t with Theil cumulative correction (Serra 2002),
        #     exactly as eventstudy2's NCAAREt_test (Mata lines 1698-1711):
        #     Citcum = L + L/N_t + [cumsum_s(MRE_s - L_s*Rmbar)]^2 / sum_tau(MR-Rmbar)^2
        len_w = ub - lb + 1
        ARE = np.column_stack([_pad(a, len_w) for a in K["are"]])     # (len_w, N)
        MRE = np.column_stack([_pad(m, len_w) for m in K["mre"]])
        ind = np.isfinite(ARE)
        Nday = ind.sum(axis=1)                                        # firms per day
        Lrun = np.cumsum(ind, axis=0)                                 # running count
        citcum = np.empty(N)
        for i in range(N):
            mr_est = K["mr_est"][i]
            rmbar = np.mean(mr_est)
            denom = np.sum((mr_est - rmbar) ** 2)
            if nomarket:                          # RAW/COMEAN: Citcum = window length only
                citcum[i] = Lrun[-1, i]
                continue
            fin = ind[:, i]                       # eventstudy2 L is missing where ARE missing,
            term = (MRE[:, i] - Lrun[:, i] * rmbar)[fin]   # and the cumsum skips those rows
            nomcum = (np.sum(term)) ** 2
            citcum[i] = Lrun[-1, i] + Lrun[-1, i] / Nday[-1] + (nomcum / denom if denom > 0 else 0.0)
        sicum2 = (si ** 2) * citcum
        caar_var = np.sum(sicum2) / (Nday[-1] ** 2)
        t_cs = caar / np.sqrt(caar_var)
        row["t_test"] = t_cs
        row["p_t_test"] = 2 * t_dist.sf(abs(t_cs), dof_t)

        # --- generalized sign (firm-level CAR) and Wilcoxon (pooled firm-days)
        row["GenSign"], row["p_GenSign"] = _gensign(car, N, p_hat)
        pooled = ARE[ind]                       # all non-missing firm-day ARs in window
        row["Wilcoxon"], row["p_Wilcoxon"] = _wilcoxon(pooled)
        rows.append(row)
    return pd.DataFrame(rows)


def _pad(a, n):
    if len(a) == n:
        return np.asarray(a, float)
    out = np.full(n, np.nan)
    out[:len(a)] = a
    return out


def _kolari_adj(ar_grids, stdf_grids, si, df):
    """KP (2010) ADJ from pairwise correlations of estimation-window standardized
    residuals, computed once globally. Mirrors KOLARI() in eventstudy2.ado.

    VitEst_it = AR_it / sqrt(si_i * STDF_it).  Pearson correlation is scale- and
    location-invariant, so the si scaling does not affect rbar, but the per-day
    1/sqrt(STDF_it) weighting does (it is part of eventstudy2's V). rbar = mean of
    off-diagonal pairwise correlations over days both firms trade.

        ADJ = sqrt((1 - rbar) / (1 + (N-1) * rbar))
    """
    N = len(ar_grids)
    if N < 2:
        return 1.0
    A = np.column_stack(ar_grids)          # (days x firms)
    S = np.column_stack(stdf_grids)
    si = np.asarray(si, float)
    V = A / np.sqrt(si[None, :] * S)       # standardized residuals (days x firms)
    corrs = []
    for i in range(N):
        for j in range(i + 1, N):
            m = np.isfinite(V[:, i]) & np.isfinite(V[:, j])
            if m.sum() < 3:
                continue
            a, b = V[m, i], V[m, j]
            if a.std() == 0 or b.std() == 0:
                continue
            corrs.append(np.corrcoef(a, b)[0, 1])
    if not corrs:
        return np.nan                       # KP adjustment undefined -> propagate
    rbar = float(np.mean(corrs))
    # eventstudy2 infers the effective firm count from the # of valid ordered pairs
    n_ordered = 2 * len(corrs)
    n_eff = round(0.5 + np.sqrt(0.25 + n_ordered))
    val = (1 - rbar) / (1 + (n_eff - 1) * rbar)
    return float(np.sqrt(val)) if val > 0 else np.nan


def _gensign(car, N, p_hat):
    n_nonneg = int(np.sum(car >= 0))
    z = (n_nonneg - N * p_hat) / np.sqrt(N * p_hat * (1 - p_hat))
    return z, 2 * norm.sf(abs(z))


def _wilcoxon(car):
    x = car[np.isfinite(car)]
    n = len(x)
    from scipy.stats import rankdata
    ranks = rankdata(np.abs(x))
    Sn = np.sum((x >= 0).astype(float) * ranks)
    ESn = n * (n + 1) / 4
    VarSn = n * (n + 1) * (2 * n + 1) / 24
    z = (Sn - ESn) / np.sqrt(VarSn)
    return z, 2 * norm.sf(abs(z))


# --------------------------------------------------------------------- selftest
def _selftest():
    """Synthetic-data self-check: known DGP, no external inputs.

    Build a liquid market + 30 firms with AR=alpha+beta*mkt+eps and inject a known
    +5% abnormal return on the event day for every firm; assert the engine recovers
    it (CAAR ~ +0.05 at [0,0], strongly significant) and that a no-event placebo is
    insignificant.
    """
    # thin-trading core: cum_periods must match eventstudy2.ado (lines 451-454)
    cp = cum_periods(np.array([0.01, np.nan, np.nan, 0.02, 0.03, np.nan, 0.04]))
    assert list(cp) == [1, 1, 2, 3, 1, 1, 2], f"cum_periods wrong: {list(cp)}"

    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2018-01-01", periods=400)
    mkt = pd.DataFrame({"date": dates, "mkt": rng.normal(0.0003, 0.01, len(dates))})
    E = 320
    rows = []
    for f in range(30):
        beta = rng.uniform(0.8, 1.3)
        eps = rng.normal(0, 0.012, len(dates))
        r = 0.0002 + beta * mkt["mkt"].to_numpy() + eps
        r[E] += 0.05                        # known +5% event-day abnormal return
        for d, ret in zip(dates, r):
            rows.append({"firm_id": f"F{f}", "date": d, "ret": ret})
    returns = pd.DataFrame(rows)
    events = pd.DataFrame({"firm_id": [f"F{f}" for f in range(30)],
                           "event_date": dates[E]})

    res = run_event_study(returns, mkt, events, id_col="firm_id", mkt_col="mkt",
                          model="FM", car_windows=[(0, 0), (-1, 1)],
                          eswlb=-250, eswub=-11, evwlb=-10, evwub=10,
                          minesw=100, thin_trading=False)
    ts = res.teststats.set_index("window")
    caar00 = ts.loc["[0,0]", "CAAR"]
    p00 = ts.loc["[0,0]", "p_Patell"]
    assert res.car["exclusion_reason"].eq("").all(), "no firm should be excluded"
    assert abs(caar00 - 0.05) < 0.01, f"CAAR[0,0] should recover ~0.05, got {caar00:.4f}"
    assert p00 < 1e-6, f"event day should be highly significant, p={p00:.2e}"
    # placebo at a non-event day -> insignificant
    res2 = run_event_study(returns, mkt, events.assign(event_date=dates[200]),
                           id_col="firm_id", mkt_col="mkt", model="FM",
                           car_windows=[(0, 0)], eswlb=-150, eswub=-11,
                           evwlb=-10, evwub=10, minesw=100, thin_trading=False)
    p_plac = res2.teststats.set_index("window").loc["[0,0]", "p_Patell"]
    assert p_plac > 0.10, f"placebo should be insignificant, p={p_plac:.3f}"
    print(f"SELFTEST OK  CAAR[0,0]={caar00:.4f} (true 0.05)  p_event={p00:.2e}  "
          f"p_placebo={p_plac:.3f}  firms={ts.loc['[0,0]','N']}")
    return True


# -------------------------------------------------------------------------- CLI
def _parse_windows(s):
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        lb, ub = part.split(",")
        out.append((int(lb), int(ub)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generic event-study CAR engine "
                                 "(Python replication of eventstudy2).")
    ap.add_argument("--selftest", action="store_true", help="run synthetic self-check and exit")
    ap.add_argument("--returns"); ap.add_argument("--market"); ap.add_argument("--events")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--id-col", default="firm_id")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--ret-col", default="ret")
    ap.add_argument("--event-id-col", default=None)
    ap.add_argument("--event-date-col", default="event_date")
    ap.add_argument("--mkt-col", default="mkt")
    ap.add_argument("--factor-cols", default="", help="comma-separated extra FM factors")
    ap.add_argument("--model", default="FM", choices=["FM", "MA", "COMEAN", "RAW"])
    ap.add_argument("--car-windows", default="-1,1;-5,5", help='e.g. "-1,1;-5,5;-10,10"')
    ap.add_argument("--evwlb", type=int, default=None)
    ap.add_argument("--evwub", type=int, default=None)
    ap.add_argument("--eswlb", type=int, default=-270)
    ap.add_argument("--eswub", type=int, default=-21)
    ap.add_argument("--minesw", type=int, default=30)
    ap.add_argument("--minevw", type=int, default=1)
    ap.add_argument("--shift", type=int, default=3)
    ap.add_argument("--no-thin-trading", action="store_true")
    ap.add_argument("--no-log-returns", action="store_true")
    ap.add_argument("--arfillevent", action="store_true")
    ap.add_argument("--nokolari", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0

    if not (args.returns and args.events):
        ap.error("--returns and --events are required (or use --selftest)")
    if args.model in ("FM", "MA") and not args.market:
        ap.error(f"--market is required for model {args.model}")

    res = run_event_study(
        read_table(args.returns),
        read_table(args.market) if args.market else None,
        read_table(args.events),
        id_col=args.id_col, date_col=args.date_col, ret_col=args.ret_col,
        event_id_col=args.event_id_col, event_date_col=args.event_date_col,
        mkt_col=args.mkt_col,
        factor_cols=[c for c in args.factor_cols.split(",") if c],
        model=args.model, car_windows=_parse_windows(args.car_windows),
        evwlb=args.evwlb, evwub=args.evwub, eswlb=args.eswlb, eswub=args.eswub,
        minesw=args.minesw, minevw=args.minevw, shift=args.shift,
        thin_trading=not args.no_thin_trading, log_returns=not args.no_log_returns,
        arfillevent=args.arfillevent, nokolari=args.nokolari,
    )
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_table(res.ar, out / "ar_panel.csv")
    write_table(res.car, out / "car_panel.csv")
    write_table(res.teststats, out / "test_statistics.csv")
    n_ok = (res.car["exclusion_reason"] == "").sum()
    print(f"events={len(res.car)} included={n_ok} excluded={len(res.car)-n_ok}")
    print(res.teststats.to_string(index=False))
    print(f"written: {out/'ar_panel.csv'}, {out/'car_panel.csv'}, {out/'test_statistics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
