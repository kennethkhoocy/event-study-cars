#!/usr/bin/env python3
"""
validate_cars.py
================
Post-hoc validation script for CAR output files.

Checks that a CAR output parquet file conforms to the output contract
defined in the event-study-cars skill. Fully generic — no hardcoded
project paths or column names.

Usage:
    python validate_cars.py --car-file output.parquet \
        --firm-col firm_id --event-col event_id \
        --car-cols CAR_mm_m1_p1,CAR_mm_m5_p5 \
        --exclusion-col exclusion_reason \
        --nar-cols n_ar_mm_m1_p1,n_ar_mm_m5_p5 \
        --window-lengths 3,11 \
        --test-stat-dir output/test_stats/

All arguments except --car-file are optional and have sensible auto-detection
behavior.
"""

import argparse
import sys
from pathlib import Path

try:
    import polars as pl

    USE_POLARS = True
except ImportError:
    import pandas as pd

    USE_POLARS = False


def read_parquet(path: str):
    """Read a parquet file using polars (preferred) or pandas."""
    if USE_POLARS:
        return pl.read_parquet(path)
    return pd.read_parquet(path)


def get_columns(df) -> list[str]:
    if USE_POLARS:
        return df.columns
    return list(df.columns)


def col_values(df, col):
    """Return column values as a numpy-like array."""
    if USE_POLARS:
        return df[col].to_numpy()
    return df[col].values


def count_non_null(df, col) -> int:
    if USE_POLARS:
        return df.filter(pl.col(col).is_not_null()).height
    return df[col].notna().sum()


def count_null(df, col) -> int:
    if USE_POLARS:
        return df.filter(pl.col(col).is_null()).height
    return df[col].isna().sum()


def n_rows(df) -> int:
    if USE_POLARS:
        return df.height
    return len(df)


def filter_both_non_null(df, col_a, col_b):
    """Return rows where col_a is not null AND col_b is not null."""
    if USE_POLARS:
        return df.filter(pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null())
    mask = df[col_a].notna() & df[col_b].notna()
    return df[mask]


def filter_non_null_and_check(df, car_col, nar_col, expected_length):
    """Return count of rows where CAR is valid but n_ar != expected_length."""
    if USE_POLARS:
        bad = df.filter(
            pl.col(car_col).is_not_null() & (pl.col(nar_col) != expected_length)
        )
        return bad.height
    mask = df[car_col].notna() & (df[nar_col] != expected_length)
    return mask.sum()


class ValidationResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.message = ""
        self.count_bad = 0
        self.count_total = 0

    def fail(self, msg: str, count_bad: int = 0, count_total: int = 0):
        self.passed = False
        self.message = msg
        self.count_bad = count_bad
        self.count_total = count_total

    def ok(self, msg: str, count_total: int = 0):
        self.passed = True
        self.message = msg
        self.count_total = count_total

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        detail = f" ({self.message})" if self.message else ""
        counts = ""
        if self.count_bad > 0:
            counts = f" [{self.count_bad}/{self.count_total} violations]"
        elif self.count_total > 0:
            counts = f" [{self.count_total} checked]"
        return f"  [{status}] {self.name}{detail}{counts}"


def auto_detect_columns(columns: list[str]) -> dict:
    """Try to auto-detect column roles from naming conventions."""
    detected = {}

    car_cols = [c for c in columns if c.upper().startswith("CAR")]
    if car_cols:
        detected["car_cols"] = car_cols

    nar_cols = [c for c in columns if c.lower().startswith("n_ar")]
    if nar_cols:
        detected["nar_cols"] = nar_cols

    excl_candidates = [
        c
        for c in columns
        if any(
            kw in c.lower()
            for kw in ["exclusion", "exclude", "reason", "drop", "flag"]
        )
    ]
    if excl_candidates:
        detected["exclusion_col"] = excl_candidates[0]

    for candidate in ["firm_id", "stockcode", "ticker", "security_id", "permno"]:
        if candidate in columns:
            detected["firm_col"] = candidate
            break

    for candidate in ["event_id", "eventid", "event_key"]:
        if candidate in columns:
            detected["event_col"] = candidate
            break

    return detected


def validate(args) -> list[ValidationResult]:
    results = []

    # ── Load data ──
    car_path = Path(args.car_file)
    if not car_path.exists():
        r = ValidationResult("File exists")
        r.fail(f"File not found: {car_path}")
        return [r]

    df = read_parquet(str(car_path))
    columns = get_columns(df)
    total_rows = n_rows(df)

    r = ValidationResult("File readable")
    r.ok(f"{total_rows} rows, {len(columns)} columns", total_rows)
    results.append(r)

    # ── Auto-detect columns if not specified ──
    detected = auto_detect_columns(columns)

    car_cols = (
        args.car_cols.split(",") if args.car_cols else detected.get("car_cols", [])
    )
    nar_cols = (
        args.nar_cols.split(",") if args.nar_cols else detected.get("nar_cols", [])
    )
    exclusion_col = args.exclusion_col or detected.get("exclusion_col")
    firm_col = args.firm_col or detected.get("firm_col")
    event_col = args.event_col or detected.get("event_col")
    window_lengths = (
        [int(x) for x in args.window_lengths.split(",")]
        if args.window_lengths
        else []
    )

    # ── Check 1: Required identifier columns ──
    for col_name, col_val in [("firm_id", firm_col), ("event_id", event_col)]:
        r = ValidationResult(f"Column present: {col_name}")
        if col_val and col_val in columns:
            r.ok(f"Found as '{col_val}'")
        elif col_val:
            r.fail(f"Column '{col_val}' not found in dataset")
        else:
            r.fail(f"No {col_name} column detected — specify via --{col_name.replace('_', '-')}-col")
        results.append(r)

    # ── Check 2: CAR columns exist ──
    r = ValidationResult("CAR columns present")
    if not car_cols:
        r.fail("No CAR columns detected — specify via --car-cols")
    else:
        missing_car = [c for c in car_cols if c not in columns]
        if missing_car:
            r.fail(f"Missing CAR columns: {missing_car}")
        else:
            r.ok(f"{len(car_cols)} CAR columns found")
    results.append(r)

    # ── Check 3: No CAR value where exclusion reason is non-null ──
    if exclusion_col and exclusion_col in columns and car_cols:
        for cc in car_cols:
            if cc not in columns:
                continue
            r = ValidationResult(f"Exclusion guard: {cc}")
            # Rows where exclusion is non-null AND CAR is also non-null = bad
            bad_df = filter_both_non_null(df, exclusion_col, cc)
            n_bad = n_rows(bad_df)
            if n_bad > 0:
                r.fail(
                    f"{n_bad} rows have both a CAR value and an exclusion reason",
                    count_bad=n_bad,
                    count_total=total_rows,
                )
            else:
                r.ok("No CARs where exclusion reason is set", total_rows)
            results.append(r)
    elif exclusion_col and exclusion_col not in columns:
        r = ValidationResult("Exclusion column present")
        r.fail(f"Column '{exclusion_col}' not found")
        results.append(r)

    # ── Check 4: Valid CARs have n_ar == expected window length ──
    if nar_cols and window_lengths and car_cols:
        if len(nar_cols) != len(car_cols) or len(window_lengths) != len(car_cols):
            r = ValidationResult("AR count consistency")
            r.fail(
                f"Mismatch: {len(car_cols)} CAR cols, {len(nar_cols)} n_ar cols, "
                f"{len(window_lengths)} window lengths — must be equal"
            )
            results.append(r)
        else:
            for cc, nc, wl in zip(car_cols, nar_cols, window_lengths):
                r = ValidationResult(f"AR count = {wl}: {cc}")
                if nc not in columns:
                    r.fail(f"n_ar column '{nc}' not found")
                elif cc not in columns:
                    r.fail(f"CAR column '{cc}' not found")
                else:
                    n_bad = filter_non_null_and_check(df, cc, nc, wl)
                    n_valid = count_non_null(df, cc)
                    if n_bad > 0:
                        r.fail(
                            f"{n_bad} valid CARs have n_ar != {wl}",
                            count_bad=n_bad,
                            count_total=n_valid,
                        )
                    else:
                        r.ok(f"All {n_valid} valid CARs have n_ar = {wl}", n_valid)
                results.append(r)

    # ── Check 5: Estimation diagnostics present ──
    diag_cols_expected = ["alpha", "beta", "nobs", "r2", "sigma_hat"]
    diag_cols_patterns = [
        "alpha",
        "beta",
        "nobs",
        "n_obs",
        "r2",
        "r_squared",
        "sigma",
        "rmse",
    ]
    found_diag = [
        c for c in columns if any(p in c.lower() for p in diag_cols_patterns)
    ]
    r = ValidationResult("Estimation diagnostics present")
    if found_diag:
        r.ok(f"Found {len(found_diag)} diagnostic columns: {found_diag[:5]}...")
    else:
        r.fail("No estimation diagnostic columns detected (alpha, beta, nobs, r2, sigma_hat)")
    results.append(r)

    # ── Check 6: Test statistic files exist alongside CAR panel ──
    if args.test_stat_dir:
        ts_dir = Path(args.test_stat_dir)
        r = ValidationResult("Test statistic files")
        if not ts_dir.exists():
            r.fail(f"Test stat directory not found: {ts_dir}")
        else:
            ts_files = list(ts_dir.glob("*.parquet")) + list(ts_dir.glob("*.csv"))
            if ts_files:
                r.ok(f"{len(ts_files)} test stat files found in {ts_dir}")
            else:
                r.fail(f"No .parquet or .csv files in {ts_dir}")
        results.append(r)
    else:
        # Check same directory as CAR file
        car_dir = car_path.parent
        ts_candidates = [
            f
            for f in car_dir.iterdir()
            if f.is_file()
            and any(
                kw in f.stem.lower()
                for kw in ["test_stat", "aar", "caar", "patell", "bmp"]
            )
        ]
        r = ValidationResult("Test statistic files (auto-detect)")
        if ts_candidates:
            r.ok(
                f"{len(ts_candidates)} test stat files found: "
                f"{[f.name for f in ts_candidates[:3]]}..."
            )
        else:
            r.fail(
                "No test statistic files found alongside CAR panel. "
                "Specify --test-stat-dir if they are elsewhere."
            )
        results.append(r)

    # ── Check 7: NaN summary for CARs ──
    for cc in car_cols:
        if cc not in columns:
            continue
        r = ValidationResult(f"NaN coverage: {cc}")
        n_valid = count_non_null(df, cc)
        n_nan = count_null(df, cc)
        pct_valid = 100 * n_valid / total_rows if total_rows > 0 else 0
        if pct_valid < 10:
            r.fail(
                f"Only {pct_valid:.1f}% valid ({n_valid}/{total_rows})",
                count_bad=n_nan,
                count_total=total_rows,
            )
        else:
            r.ok(
                f"{pct_valid:.1f}% valid ({n_valid}/{total_rows}), "
                f"{n_nan} NaN",
                total_rows,
            )
        results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate a CAR output parquet file against the output contract."
    )
    parser.add_argument(
        "--car-file", required=True, help="Path to the CAR output parquet file"
    )
    parser.add_argument(
        "--firm-col", default=None, help="Column name for firm identifier"
    )
    parser.add_argument(
        "--event-col", default=None, help="Column name for event identifier"
    )
    parser.add_argument(
        "--car-cols",
        default=None,
        help="Comma-separated CAR column names (e.g., CAR_m1_p1,CAR_m5_p5)",
    )
    parser.add_argument(
        "--exclusion-col",
        default=None,
        help="Column name for exclusion reason",
    )
    parser.add_argument(
        "--nar-cols",
        default=None,
        help="Comma-separated n_ar column names (parallel to --car-cols)",
    )
    parser.add_argument(
        "--window-lengths",
        default=None,
        help="Comma-separated expected window lengths (parallel to --car-cols)",
    )
    parser.add_argument(
        "--test-stat-dir",
        default=None,
        help="Directory containing test statistic output files",
    )

    args = parser.parse_args()
    results = validate(args)

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"  CAR Validation Report: {args.car_file}")
    print(f"{'='*60}\n")

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)

    for r in results:
        print(str(r))

    print(f"\n{'─'*60}")
    print(f"  {n_pass} passed, {n_fail} failed, {len(results)} total checks")
    print(f"{'─'*60}\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
