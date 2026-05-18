"""Phase 9 evaluation runner.

run_eval() orchestrates all three modes (baseline, rag_llm, agent) against a
list of tickers, judges each report with judge_report(), and returns a
long-format DataFrame with per-row scores, latency, and cost.

Outputs written to disk:
  data/eval/results.csv   — full long-format DataFrame
  data/eval/summary.md    — mean scores and costs per mode, rendered as markdown

Cost/latency are extracted from each mode's existing stdout log lines:
  "Wall time: X.XXs"  and  "Total cost: $X.XXXXX"
All three modes already print these consistently, so we capture stdout rather
than refactoring the mode APIs.

CLI: python -m yoda.eval.runner [TICKER [TICKER ...]]
Default tickers: CURATED_TICKERS (50 major US-listed stocks, ~$15/run, 2-3 hours).
Quick test: python -m yoda.eval.runner NFLX (~$0.30, 3-5 min).
Custom override: python -m yoda.eval.runner AAPL AMZN JPM PANW NFLX COIN
"""

import contextlib
import io
import re
import time

import pandas as pd

from yoda.eval.judge import cache_key_for, judge_report, prune_judge_cache, write_run_manifest
from yoda.eval.rubric import JudgeScores
from yoda.ingest.chunker import chunk_filing
from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.baseline import run_baseline
from yoda.modes.personality_panel import run_personality_panel
from yoda.schema import EarningsReport


# Curated 10-ticker universe covering 9 distinct sectors.
# Chosen for variety over breadth: one representative per sector/archetype
# so evaluation captures diverse filing structures and business narratives.
CURATED_TICKERS = [
    "AAPL",   # Tech — hardware + services
    "NVDA",   # Tech — semiconductors / AI
    "AMZN",   # Tech — cloud + e-commerce
    "JPM",    # Financials — banking
    "JNJ",    # Healthcare — pharma + medtech
    "XOM",    # Energy
    "KO",     # Consumer Staples
    "NFLX",   # Media / Streaming
    "CAT",    # Industrials
    "PANW",   # Cybersecurity / SaaS
]


# ---------------------------------------------------------------------------
# Regex patterns for parsing cost and latency from captured stdout
# ---------------------------------------------------------------------------

_RE_WALL_TIME = re.compile(r"Wall time:\s+([\d.]+)s")
_RE_COST      = re.compile(r"Total cost:\s+\$([\d.]+)")


# ---------------------------------------------------------------------------
# Baseline excerpt builder
# (Duplicated from baseline.py __main__ because that block is not a function.
# Keeping it local avoids importing __main__-level code from another module.)
# ---------------------------------------------------------------------------

def _build_baseline_excerpt(filing: dict) -> str:
    # Build a ~5000-char excerpt preferring MD&A then Financial Statements,
    # matching the same logic used in baseline.py's __main__ smoke test.
    chunks = chunk_filing(filing["clean_text"], filing["raw_html"])
    preferred = ["MD&A", "Financial Statements"]
    parts: list[str] = []
    total = 0

    for section_name in preferred:
        for chunk in chunks:
            if chunk.section == section_name and total < 5000:
                parts.append(chunk.text)
                total += len(chunk.text)
        if total >= 5000:
            break

    return " ".join(parts)[:5000]


# ---------------------------------------------------------------------------
# Stdout-capture helpers
# ---------------------------------------------------------------------------

def _capture(fn, *args) -> tuple[EarningsReport, float, float, str]:
    # Run fn(*args) with stdout captured. Returns (report, latency_s, cost_usd, log).
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        report = fn(*args)
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()

    wall_match = _RE_WALL_TIME.search(log)
    cost_match  = _RE_COST.search(log)
    latency = float(wall_match.group(1)) if wall_match else elapsed
    cost    = float(cost_match.group(1))  if cost_match  else 0.0

    return report, latency, cost, log


def _capture_panel(ticker: str) -> tuple[EarningsReport, float, float, str]:
    # Variant of _capture for run_personality_panel which returns
    # (report, personality_results, critique_messages). Extracts the report
    # and discards the trace details (the eval rubric scores the report).
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        report, _personality_results, _critique = run_personality_panel(ticker)
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()

    wall_match = _RE_WALL_TIME.search(log)
    cost_match  = _RE_COST.search(log)
    latency = float(wall_match.group(1)) if wall_match else elapsed
    cost    = float(cost_match.group(1))  if cost_match  else 0.0

    return report, latency, cost, log


# ---------------------------------------------------------------------------
# Per-mode dispatch
# ---------------------------------------------------------------------------

def _run_mode(
    mode: str, ticker: str, filing: dict
) -> tuple[EarningsReport, float, float, str]:
    """Run one mode for one ticker. Returns (report, latency_s, cost_usd, log)."""
    if mode == "baseline":
        excerpt = _build_baseline_excerpt(filing)
        return _capture(run_baseline, ticker, excerpt)
    elif mode == "yoda":
        return _capture_panel(ticker)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Score extraction helper
# ---------------------------------------------------------------------------

def _scores_to_dict(scores: JudgeScores) -> dict:
    # Flatten the five DimensionScore instances into plain int columns.
    return {
        "extraction_completeness": scores.extraction_completeness.score,
        "accuracy":                scores.accuracy.score,
        "source_traceability":     scores.source_traceability.score,
        "relevance":               scores.relevance.score,
        "usefulness":              scores.usefulness.score,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eval(
    tickers: list[str],
    modes: list[str] = ("baseline", "yoda"),
) -> pd.DataFrame:
    """Run each (ticker, mode) pair, judge it, return a long-format DataFrame.

    DataFrame columns:
        ticker, mode, extraction_completeness, accuracy, source_traceability,
        relevance, usefulness, latency_seconds, cost_usd

    Also writes data/eval/results.csv and data/eval/summary.md.
    """
    import pathlib
    out_dir = pathlib.Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"

    # Remove any partial results file from a previous interrupted run.
    results_path.unlink(missing_ok=True)

    rows = []
    # Keys of every judge_report() call this run; written to a manifest at the end.
    run_cache_keys: set[str] = set()

    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\n=== {ticker} ===")

        # Fetch the filing once per ticker; all three modes share it.
        print(f"  fetching filing for {ticker}...")
        filing = fetch_latest_filing(ticker)
        filing_text = filing["clean_text"][:30000]

        for mode in modes:
            print(f"  running mode={mode}...")
            try:
                report, latency, cost, log = _run_mode(mode, ticker, filing)
            except Exception as exc:
                print(f"  ERROR in mode={mode}: {exc}")
                continue

            # Save report JSON — overwrites previous run, keeping 1 per (mode, ticker).
            (out_dir / f"{mode}_{ticker}.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8"
            )

            # Judge the report against the filing text.
            scores = judge_report(report, filing_text)
            run_cache_keys.add(cache_key_for(report))

            row = {
                "ticker": ticker,
                "mode": mode,
                "latency_seconds": round(latency, 2),
                "cost_usd": round(cost, 5),
                **_scores_to_dict(scores),
            }
            rows.append(row)

            # Write this row immediately so progress survives a crash.
            pd.DataFrame([row]).to_csv(
                results_path, mode="a", header=len(rows) == 1, index=False
            )
            print(f"  mode={mode} done — latency={latency:.1f}s cost=${cost:.4f}")

    df = pd.DataFrame(rows)

    # Overwrite with the complete, clean DataFrame (removes any partial-write artifacts).
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Build and write the summary markdown.
    _write_summary(df, out_dir / "summary.md")
    print(f"Summary saved to {out_dir / 'summary.md'}")

    # Generate comparison chart as a top-level output.
    chart_path = out_dir / "comparison.png"
    _write_chart(df, chart_path)
    print(f"Chart saved to {chart_path}")

    # Write a manifest for this run and prune cache to the last 2 runs.
    write_run_manifest(run_cache_keys)
    prune_judge_cache(keep_runs=2)

    # Delete report JSONs from modes not in this run (e.g. rag_llm, agent, panel_fast).
    active_prefixes = tuple(f"{m}_" for m in modes)
    for f in out_dir.glob("*.json"):
        if not any(f.name.startswith(p) for p in active_prefixes):
            f.unlink()

    return df


# ---------------------------------------------------------------------------
# Summary markdown builder
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame) -> str:
    # Render a DataFrame as a GitHub-flavored markdown table without tabulate.
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep    = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    rows   = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(df.index, df.values)
    ]
    return "\n".join([header, sep] + rows)


def _write_summary(df: pd.DataFrame, path) -> None:
    # Compute mean scores and operational metrics grouped by mode.
    score_cols = [
        "extraction_completeness", "accuracy",
        "source_traceability", "relevance", "usefulness",
    ]
    ops_cols = ["latency_seconds", "cost_usd"]

    mode_means = df.groupby("mode")[score_cols + ops_cols].mean().round(2)

    lines = ["# Phase 9 Evaluation Summary", ""]
    lines.append("## Mean scores by mode")
    lines.append("")
    lines.append(_df_to_markdown(mode_means))
    lines.append("")

    # Per-ticker mean if more than one ticker was evaluated.
    if df["ticker"].nunique() > 1:
        ticker_means = df.groupby("ticker")[score_cols].mean().round(2)
        lines.append("## Mean scores by ticker")
        lines.append("")
        lines.append(_df_to_markdown(ticker_means))
        lines.append("")

    import pathlib
    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Comparison chart — grouped bar chart, baseline vs Yoda variants across 5 metrics
# ---------------------------------------------------------------------------

# Display-name mapping for the chart legend / title. Keep the underlying
# mode keys intact in the DataFrame / CSV / summary.md — only the chart
# rebrands the modes for slide presentation.
_MODE_DISPLAY = {
    "baseline": "Baseline",
    "yoda":     "Yoda",
}


def _display(mode: str) -> str:
    return _MODE_DISPLAY.get(mode, mode)


def _write_chart(df: pd.DataFrame, path) -> None:
    # Grouped bar chart with one bar per mode within each metric group.
    # Designed for slide embedding: 10x5 figure, value labels above bars.
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        "extraction_completeness", "accuracy",
        "source_traceability", "relevance", "usefulness",
    ]
    mode_means = df.groupby("mode")[metrics].mean()

    # X positions for the metric groups; one bar offset per mode.
    x = np.arange(len(metrics))
    n_modes = len(mode_means.index)
    width = 0.8 / max(n_modes, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Render one bar series per mode, offset so they sit side-by-side.
    for i, mode in enumerate(mode_means.index):
        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, mode_means.loc[mode], width, label=_display(mode))
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    # Labels, title, axis limits, and grid.
    ax.set_ylabel("Mean Score (1-5)")
    ax.set_title("Baseline vs Yoda — Mean Rubric Scores")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylim(0, 5)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point — run with: python -m yoda.eval.runner [TICKER ...]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Default to the curated 50-ticker universe for comprehensive evaluation.
    # Override with CLI args: python -m yoda.eval.runner NFLX AAPL (for quick tests)
    tickers = sys.argv[1:] if len(sys.argv) > 1 else CURATED_TICKERS
    df = run_eval(tickers)
    print(df.to_string(index=False))
