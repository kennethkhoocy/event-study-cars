# event-study-cars

A Claude Code skill that encodes a complete, publication-grade methodology for
computing **Cumulative Abnormal Returns (CARs)** and their test statistics. It
is, in essence, a **language-agnostic (Python-first) replication of
`eventstudy2`** — the Stata package by **Thomas Kaspereit** — so that the same
methodology can be implemented and audited outside Stata, in any market, asset
class, or event type.

## Motivation

This skill exists to replicate, in Python (or any language), the methodology of
**`eventstudy2` (v3.2b), the Stata event-study package written by Thomas
Kaspereit.** `eventstudy2` is the reference implementation that gets the hard
parts right; all credit for the underlying methodology and its careful
correctness belongs to Kaspereit and to the econometrics literature his package
operationalizes (see [References](#references)).

The skill ships a **complete, runnable Python engine** (`scripts/eventstudy.py`)
that reproduces `eventstudy2` independently of Stata, plus the full methodology
in `references/` and a post-hoc output checker. The engine is not a line-by-line
port of the Stata/Mata code; it is an independent reimplementation of the same
method — the 8-step pipeline, the exact correction formulas, and the test
statistics — written from the documented specification. Its fidelity is
**verified against Stata `eventstudy2` itself**: on a generic CRSP sample,
abnormal returns match to 1e-8, CARs to 6e-8, and CAAR plus all seven implemented
test statistics (cross-sectional t, Patell, Patell-KP, Boehmer/BMP, Kolari/BMP-KP,
generalized sign, Wilcoxon) to ~1e-7.

> **Credit.** The methodology, defaults, and correctness guarantees mirrored
> here originate with Thomas Kaspereit's `eventstudy2` for Stata. If you use
> this skill in research, cite `eventstudy2` and the original methodological
> papers, not this README.

The motivation for a faithful port is that event studies look deceptively
simple — subtract a benchmark return from a realized return, sum over a window,
run a t-test — yet a naive implementation gets the answer subtly and confidently
wrong, with errors that all push significance in convenient directions:

- **Zero-filling missing returns** biases CARs toward zero for exactly the
  illiquid stocks where abnormal returns are largest.
- **Ignoring thin trading** leaves the heteroscedasticity of multi-period
  returns uncorrected, so the standard errors are wrong and the t-stats are
  inflated.
- **IPO / delisting inside the window** silently injects survivorship bias.
- **Boundary contamination** lets a return that spans outside the intended
  window leak into the CAR.
- **Reporting only a Patell or plain t-test** ignores event-induced variance
  and cross-sectional correlation, both of which over-reject the null.

Kaspereit's `eventstudy2` handles all of this correctly, but it is Stata-only
and its mechanics live inside the package implementation. This skill lifts that
same methodology into a reusable, language-agnostic reference: the full 8-step
pipeline, the exact correction formulas, all 13 test statistics, the ten most
common mistakes, and a generic validation script — so any CAR pipeline (Python,
R, Julia, or Stata) can be built, audited, or upgraded to the `eventstudy2`
standard. The goal is that a result is correct because the method is correct,
verifiably the same method Kaspereit implemented.

## What it covers

The skill documents the full pipeline end to end:

1. **Dateline construction** — a master trading calendar in relative trading
   time, robust to heterogeneous holidays in international samples.
2. **Event-date mapping** — nearest valid trading day, with a `max_shift` guard
   that excludes (rather than silently relocates) far-off events.
3. **Estimation / event windows** — defined in trading time, with a gap to
   prevent event contamination and an IPO/delisting survivorship guard.
4. **Thin-trading adjustment** — the Maynes-Rumsey (1993) trade-to-trade
   transformation (a GLS correction), on by default.
5. **OLS + STDF** — benchmark estimation with the Theil (1971) prediction-error
   correction so standard errors reflect coefficient uncertainty.
6. **Abnormal returns** — with the strict no-zero-fill rule.
7. **CAR accumulation** — with boundary-contamination guards and valid-AR
   counts.
8. **Test statistics** — Patell, BMP, Kolari-Pynnonen adjusted BMP,
   generalized sign, Wilcoxon, GRANK-T, and more (13 total), at both the AAR and
   CAAR levels.

It supports the RAW, COMEAN, market-adjusted, factor-model (market model /
FF3 / FF5 / Carhart), and BHAR benchmark models.

## Repository layout

```
event-study-cars/
├── SKILL.md                              # entry point: 8-step pipeline, rules, defaults
├── CLAUDE.md                             # maintainer rules (ship a generic engine)
├── references/
│   ├── estimation_models.md              # RAW, COMEAN, MA, FM, BHAR specifications
│   ├── thin_trading.md                   # Maynes-Rumsey (1993) trade-to-trade transform
│   ├── test_statistics.md                # all 13 test statistics with formulas
│   ├── kolari_pynnonen.md                # cross-correlation adjustment + GRANK-T
│   └── implementation_checklist.md       # the 10 most common mistakes
├── scripts/
│   ├── eventstudy.py                     # the runnable engine + CLI + self-test
│   └── validate_cars.py                  # post-hoc output-integrity checker
├── LICENSE                               # MIT
└── README.md
```

## Install

This is a [Claude Code](https://claude.com/claude-code) skill. Install it by
placing the folder in your skills directory; Claude discovers it automatically
and activates it from its description (no manual loading needed).

**Personal skill** (available in every project):

```bash
# macOS / Linux
git clone https://github.com/kennethkhoocy/event-study-cars.git \
    ~/.claude/skills/event-study-cars
```

```powershell
# Windows (PowerShell)
git clone https://github.com/kennethkhoocy/event-study-cars.git `
    "$env:USERPROFILE\.claude\skills\event-study-cars"
```

**Project-scoped skill** (checked into one repo, shared with collaborators):

```bash
git clone https://github.com/kennethkhoocy/event-study-cars.git \
    .claude/skills/event-study-cars
```

Either way, the final path must end in `.../skills/event-study-cars/` with
`SKILL.md` directly inside it. Restart Claude Code, or run `/reload-plugins`, to
pick up a newly added skill mid-session.

To confirm it loaded, ask Claude something like *"compute CARs with Patell and
BMP tests"* — the skill should trigger.

## Running the engine out of the box

`scripts/eventstudy.py` is self-contained and needs only `numpy`, `pandas`, and
`scipy` (`pyarrow` or `polars` additionally for Parquet I/O). No Stata required.

Verify the install on synthetic data, with zero external inputs:

```bash
python scripts/eventstudy.py --selftest
# SELFTEST OK  CAAR[0,0]=0.0497 (true 0.05)  p_event=2.10e-111  p_placebo=0.680  firms=30
```

Run it on your own data — three files (CSV or Parquet): a returns panel
(`id, date, ret`), a market/factor panel (`date, mkt[, factors]`), and an events
list (`id, event_date`):

```bash
python scripts/eventstudy.py \
    --returns returns.csv --market market.csv --events events.csv \
    --id-col permno --date-col date --ret-col ret \
    --event-date-col event_date --mkt-col vwretd \
    --model FM --car-windows "-1,1;-5,5;-10,10" \
    --eswlb -250 --eswub -30 --evwlb -10 --evwub 10 \
    --out-dir out/
```

It writes `ar_panel.csv` (abnormal returns), `car_panel.csv` (per firm-event
CARs, betas, exclusion reasons), and `test_statistics.csv` (CAAR + tests per
window). Defaults follow `eventstudy2` (estimation window, thin-trading on,
log-return conversion, no zero-fill); every choice is a flag — see
`--help`. Add `--factor-cols smb,hml` for Fama-French, `--model MA` for
market-adjusted, `--no-thin-trading` for highly liquid samples.

### Fidelity check vs Stata

The engine was validated against Stata `eventstudy2` (v3.2b) on a generic CRSP
sample of 40 firms at a placebo event date, with identical settings, across all
four benchmark models. Maximum absolute Python-vs-Stata difference:

| Model | Abnormal returns | CAR (all windows) | CAAR | test statistics |
|-------|------------------|-------------------|------|-----------------|
| FM (market model) | 1.4e-08 | 5.7e-08 | 1e-9 | ≤1.4e-7 |
| COMEAN | 2.8e-08 | 3.7e-08 | 7e-10 | ≤7.8e-8 |
| MA (market-adjusted) | 8.9e-09 | 1.4e-08 | 2e-9 | ≤2.9e-7 |
| RAW | 1.7e-08 | 3.7e-08 | 8e-10 | ≤1.8e-7 |

Test statistics compared: cross-sectional t (Serra), Patell, Boehmer/BMP, Kolari
(BMP-KP), generalized sign, Wilcoxon. In other words, the Python output is the
`eventstudy2` output to floating-point precision. (The Corrado-Cowan and
Zivney-Cowan rank tests and GRANK-T are documented in `references/` but not yet in
the engine; the recommended minimum set — Patell, BMP, Kolari-Pynnonen BMP,
generalized sign — is fully implemented and validated.)

## Using the validation script standalone

`scripts/validate_cars.py` runs on its own and needs only `polars` (preferred)
or `pandas`:

```bash
pip install polars        # or: pip install pandas

python scripts/validate_cars.py --car-file output.parquet \
    --firm-col firm_id --event-col event_id \
    --car-cols CAR_mm_m1_p1,CAR_mm_m5_p5 \
    --exclusion-col exclusion_reason \
    --nar-cols n_ar_mm_m1_p1,n_ar_mm_m5_p5 \
    --window-lengths 3,11 \
    --test-stat-dir output/test_stats/
```

Only `--car-file` is required; everything else auto-detects. The script verifies
that no CARs exist where an exclusion reason is set, that valid-AR counts match
the declared window lengths, that required columns are present, and that test
statistic files accompany the CAR panel. It exits non-zero on any failure, so it
drops straight into CI or a Makefile.

## How to use the skill

The fastest path is to run the shipped engine directly (see *Running the engine
out of the box* above) — it works on any returns data without writing new code.

For methodology questions or audits, describe the task to Claude — *"compute CARs
for these returns with FF3 and BMP/Kolari-Pynnonen tests"* or *"audit my existing
event-study code"*. Claude runs `scripts/eventstudy.py` for the computation and
reads `references/` for the formulas. The reference files also stand alone as a
methodology desk reference, independent of Claude.

## References

- Boehmer, Musumeci & Poulsen (1991) — event-induced variance (BMP test)
- Cowan (1992) — generalized sign test
- Kolari & Pynnonen (2010, 2011) — cross-correlation adjustment, GRANK-T
- Lyon, Barber & Tsai (1999) — skewness-adjusted BHAR bootstrap
- Maynes & Rumsey (1993) — trade-to-trade returns for thin trading
- Patell (1976) — standardized residual test
- Theil (1971) — prediction error / standard deviation of forecast

## License

MIT — see [LICENSE](LICENSE). Note this covers the reimplementation only; the
`eventstudy2` methodology is Thomas Kaspereit's, and research use should cite his
package and the original papers (see *Credit* above).
