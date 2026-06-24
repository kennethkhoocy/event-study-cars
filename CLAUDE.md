# event-study-cars — maintainer rules

## Ship a generic, out-of-the-box engine. Do NOT rebuild scripts per use.

This skill must be usable **out of the box** by any user who has Python on PATH.
The skill ships a complete, generic, runnable event-study engine in `scripts/`.
Claude Code must NOT re-author CAR/test-statistic code for each project — point
the user at the shipped CLI and let them run it on their own data.

Concretely:
- `scripts/eventstudy.py` is the engine + CLI + self-test. It is **generic**:
  column names, model, windows, thin-trading, and log-return handling are all
  CLI flags with sensible defaults. No project-specific paths, identifiers, or
  data are baked in.
- Inputs are plain files (CSV or Parquet): a returns panel, a market/factor
  panel, and an events list. Outputs are an AR panel, a per-window CAR panel,
  and a test-statistics table.
- Requires only `numpy`, `pandas`, `scipy` (Parquet I/O also needs `pyarrow`
  or `polars`). `python scripts/eventstudy.py --selftest` runs on synthetic
  data with zero external inputs.

When a user asks to compute CARs: run the shipped CLI, do not write new scripts.
Only extend `scripts/eventstudy.py` itself if a genuinely missing capability is
needed — keep it generic, and keep the self-test green.

## Fidelity contract

The engine replicates Thomas Kaspereit's `eventstudy2` (Stata, v3.2b). The
authoritative source is `eventstudy2.ado` (install in Stata via
`ssc install eventstudy2`; it lives under the Stata `ado/plus/e/` tree). AR and
CAR values must match eventstudy2 to numerical precision on identical
inputs/settings; the engine was validated against Stata output on a generic CRSP
sample across models FM/COMEAN/MA/RAW. The references/ files transcribe the
methodology and the Mata test-statistic formulas.
