"""Sequential ticker-queue processor (Phase 10).

process_queue() runs the personality panel against a list of tickers, saves
each PDF to disk as it completes, and bundles everything into a ZIP at the
end. Failures don't abort the queue — they're logged and the run continues.

The personality panel is already parallel internally (6 personalities in
parallel inside one ticker), so running tickers in parallel here would
mostly fight for OpenAI rate-limit headroom. Sequential per-ticker keeps
the API spend even and the progress easy to display in the Streamlit UI.
"""

import pathlib
import time
import zipfile
from datetime import datetime
from typing import Callable

from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.personality_panel import run_personality_panel
from yoda.report.pdf import report_to_pdf


# ---------------------------------------------------------------------------
# Output directories — created lazily on first run
# ---------------------------------------------------------------------------

_REPORTS_DIR = pathlib.Path("data/reports")
_ZIPS_DIR    = pathlib.Path("data/queue_zips")


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _prune_reports(ticker: str) -> None:
    # Delete all but the most recent PDF for this ticker in data/reports/.
    pdfs = sorted(_REPORTS_DIR.glob(f"{ticker}_*.pdf"))  # ascending = oldest first
    for old in pdfs[:-1]:
        old.unlink()


def _prune_zips(keep: int = 2) -> None:
    # Delete all but the most recent `keep` ZIPs in data/queue_zips/.
    zips = sorted(_ZIPS_DIR.glob("yoda_queue_*.zip"))  # ascending = oldest first
    for old in zips[:-keep]:
        old.unlink()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_queue(
    tickers: list[str],
    on_progress: Callable[[str, str], None] | None = None,
) -> tuple[pathlib.Path, list[dict]]:
    """Run the personality panel sequentially for each ticker.

    Each PDF is saved to data/reports/{ticker}_{timestamp}.pdf as it's
    generated. After all tickers finish, the PDFs are bundled into a single
    ZIP at data/queue_zips/yoda_queue_{timestamp}.zip.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process. Upper-cased and stripped per-ticker.
    on_progress : optional callback
        Called with (ticker, status) where status is one of:
          "running"    — about to invoke the panel
          "complete"   — succeeded; PDF available
          "failed: <msg>" — exception during this ticker

    Returns
    -------
    (zip_path, results)
        zip_path: pathlib.Path to the ZIP bundle.
        results:  list[dict] with one entry per ticker. Each entry has:
            ticker, status ("complete" | "failed"), pdf_path | None,
            error | None, seconds, cost_usd
    """
    # Normalize input: upper-case, strip, drop blanks, dedupe while preserving
    # order so the user sees results in the order they typed them.
    seen: set[str] = set()
    normalized: list[str] = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            normalized.append(t)
            seen.add(t)

    # Ensure output directories exist before we try to write into them.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _ZIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Each ticker gets its own result record so the UI can render a table.
    results: list[dict] = []
    pdf_paths: list[pathlib.Path] = []

    for ticker in normalized:
        # Notify the UI we're starting this ticker.
        if on_progress is not None:
            on_progress(ticker, "running")

        started = time.perf_counter()
        try:
            # Run the panel — the actual heavy lifting.
            report, _personality_results, _critique = run_personality_panel(ticker)

            # Fetch the filing once more so we can pass the URL to the PDF
            # for inline citation hyperlinks. fetch_latest_filing is cached
            # on disk so this is fast after the panel already ran it.
            filing = fetch_latest_filing(ticker)

            # Build the PDF filename with a timestamp so re-runs of the same
            # ticker don't overwrite previous outputs.
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            pdf_path = _REPORTS_DIR / f"{ticker}_{ts}.pdf"
            report_to_pdf(report, str(pdf_path), filing_url=filing["url"])
            pdf_paths.append(pdf_path)
            _prune_reports(ticker)

            elapsed = time.perf_counter() - started
            results.append({
                "ticker":   ticker,
                "status":   "complete",
                "pdf_path": pdf_path,
                "error":    None,
                "seconds":  elapsed,
                # cost_usd is approximate — derived from the panel's own
                # accounting; we don't re-run the calculation here.
                "cost_usd": None,
            })

            if on_progress is not None:
                on_progress(ticker, "complete")

        except Exception as exc:
            elapsed = time.perf_counter() - started
            # Failure is non-fatal — log the error and move on so a single
            # bad ticker doesn't abort a batch of 20.
            results.append({
                "ticker":   ticker,
                "status":   "failed",
                "pdf_path": None,
                "error":    str(exc),
                "seconds":  elapsed,
                "cost_usd": None,
            })

            if on_progress is not None:
                on_progress(ticker, f"failed: {exc}")

    # Build the ZIP bundle. We always create one even if every ticker failed
    # so the caller has a consistent return type; an empty ZIP is fine.
    zip_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path = _ZIPS_DIR / f"yoda_queue_{zip_ts}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf in pdf_paths:
            # Store under just the basename so users extracting the ZIP get
            # files in their current directory, not nested under data/reports/.
            zf.write(pdf, arcname=pdf.name)

    _prune_zips()
    return zip_path, results


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.queue.processor [TICKER ...]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Default to a small set so a manual run isn't expensive.
    cli_tickers = sys.argv[1:] if len(sys.argv) > 1 else ["NFLX"]

    def _print_progress(ticker: str, status: str) -> None:
        # Simple stdout callback — the Streamlit UI wires up its own version.
        print(f"  [queue] {ticker}: {status}")

    print(f"Processing queue: {cli_tickers}\n")
    zp, res = process_queue(cli_tickers, on_progress=_print_progress)

    print(f"\nZIP created at: {zp}")
    print("Results:")
    for r in res:
        print(f"  {r['ticker']:6s}: {r['status']:8s} ({r['seconds']:.1f}s)"
              f"  {r['pdf_path'] or r['error']}")
