# Yoda — Pre-Earnings Research Assistant

> **Status:** Phase 10 (multi-agent personality panel) complete. Latest evaluation pits **Baseline** against **Yoda** across a 10-ticker sector-diverse universe. Numbers and a comparison chart are in [`data/eval/summary.md`](data/eval/summary.md) and [`data/eval/comparison.png`](data/eval/comparison.png).

---
## 1. Context, User, and Problem

**Who:** A buy-side or sell-side analyst who covers 10–50 tickers and prepares for earnings calls individually, often the night before.

**The workflow being improved:** Pre-earnings research. Before every earnings call, an analyst reads the most recent 10-Q or 10-K, pulls consensus estimates, scans recent news, and assembles a one-page dossier: key metrics, segment trends, forward guidance, risk flags, and what to watch. This typically takes 60–90 minutes per ticker and is done manually, tab by tab.

**Why it matters:** Earnings surprise is one of the highest-signal events in equity analysis, and analyst preparation directly affects how quickly they can react. The bottleneck is not judgment — it is assembly time: fetching the right SEC filing section, locating a specific guidance quote, cross-checking segment revenue against prior quarters. A system that automates citation-backed assembly frees the analyst to focus on the 10% that is genuinely interpretive.

**The failure mode we are solving for:** Generic LLM summaries of earnings filings hallucinate figures and drop citations. An analyst cannot use a report that might be fabricated. Every claim in Yoda must carry a source citation (section name + chunk ID from the filing, or API name + timestamp for external data); anything that cannot be cited goes into a visible `data_gaps` list rather than being silently invented.

---

## 2. Solution and Design

Yoda fetches the most recent 10-Q (and, when available within the same 92-day window, the supplemental 10-K) from SEC EDGAR for a given ticker, chunks each filing by section, embeds and indexes them in a local ChromaDB vector store using OpenAI `text-embedding-3-small` (1536-dim, ~$0.004/filing), enriches with analyst consensus and news, and generates a structured `EarningsReport` via a multi-agent personality panel. The report downloads as a PDF; a Queue tab batches multiple tickers and bundles the PDFs into a ZIP.

### How Yoda Works

Yoda runs a six-personality panel via `yoda/modes/personality_panel.py`. Six personalities (Optimist, Pessimist, Conservative, Dreamer, Contrarian, Quant) each run parallel tool-use loops (6 tool calls, 45s wall-clock), followed by a cross-critique phase where each personality emits SUPPORTS / CHALLENGES / EXTENDS messages on its peers' hypotheses. A deterministic filter retains/contests/drops hypotheses, then gpt-4o synthesizes a final report that surfaces contested items inside the watchlist. Typical latency: ~90s per ticker.

Each personality uses three tools: `retrieve_filing` (semantic search across both the 10-Q and the supplemental 10-K when present), `search_news` (Tavily web search — all URLs accumulate into a shared news pool), and `lookup_peer` (fetch + chunk + search a competitor's filing on demand). The synthesizer then writes a structured `EarningsReport` whose watchlist items each carry 0–3 starting-point URLs drawn from that shared pool, validated against it so synthesis cannot invent links.

### Key Design Choices

| Choice | Rationale |
|---|---|
| `gpt-4o` for synthesis, `gpt-4o-mini` for personality loops + critique | Accuracy where it counts; cost control in the 6× parallel loops |
| Six-personality panel + typed cross-critique | Forces the report to consider divergent lenses (optimism, skepticism, balance-sheet, long-horizon, contrarian, quant) before synthesizing one analyst voice; cross-critique flags contested claims for the watchlist |
| 10-Q primary + 10-K supplemental ingest | The 10-Q has the freshest quarterly data for pre-earnings; the 10-K supplies annual narrative when both are within the 92-day window |
| Finnhub consensus with FMP backup | Finnhub free tier is sparse; `yoda/tools/consensus.py` falls back to FMP when Finnhub returns nothing |
| ChromaDB persistent client (`filings_openai` collection) | No native deps on Windows; cosine-distance index over 1536-dim OpenAI vectors |
| `reportlab` for PDF (not weasyprint) | weasyprint requires GTK3/Pango/Cairo native libs on Windows; reportlab is pure Python |
| Cite-or-skip rule + URL-pool validation | Any uncitable fact goes to `data_gaps`; every news URL and watchlist URL is validated against the investigation's news pool so synthesis cannot fabricate links |
| Filing-only `source_citation` enforcement | `key_metrics`, `revenue_segments`, `key_risks`, and `forward_guidance` source citations are restricted to filing section labels — a post-synthesis scrubber replaces any leaked news URLs / outlet names with a bare section fallback, and the chunk-heading extractor rejects mid-word fragments so the judge can verify traceability |
| `claude-sonnet-4-6` as eval judge | Different model family from the OpenAI-based generation system; prevents same-model bias in scoring |

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Yoda App                                    │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ Single Ticker    │  │ Queue (batch)    │  │ Eval Harness         │    │
│  │ Streamlit tab    │  │ Streamlit tab    │  │ yoda/eval/runner.py  │    │
│  │ Yoda             │  │ many → ZIP       │  │ + judge.py           │    │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘    │
│           │                     │                       │                │
│           ▼                     ▼                       │                │
│  ┌────────────────────────────────────────────────┐     │                │
│  │              Ingest + Retrieval                │     │                │
│  │  edgar.py     chunker.py    embeddings.py      │     │                │
│  │  (10-Q + 10-K) (section-aware) (ChromaDB)      │     │                │
│  └──────────────────────┬─────────────────────────┘     │                │
│                         │                               │                │
│                         ▼                               │                │
│  ┌────────────────────────────────────────────────┐     │                │
│  │           Personality Panel (Phase 10)         │     │                │
│  │  Optimist   Pessimist   Conservative           │     │                │
│  │  Dreamer    Contrarian  Quant                  │     │                │
│  │  (gpt-4o-mini · parallel tool-use loops)       │     │                │
│  │                                                │     │                │
│  │  Tools: retrieve_filing · search_news ·        │     │                │
│  │         lookup_peer                            │     │                │
│  └──────────────────────┬─────────────────────────┘     │                │
│                         │                               │                │
│                         ▼                               │                │
│  ┌────────────────────────────────────────────────┐     │                │
│  │  Cross-Critique + Synthesis                    │     │                │
│  │  gpt-4o  →  EarningsReport (Pydantic)          │     │                │
│  └──────────┬─────────────────────┬───────────────┘     │                │
│             │                     │                     │                │
│             ▼                     ▼                     ▼                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ PDF (reportlab)  │  │ data/reports/    │  │ Judge                │    │
│  │ data/reports/    │  │ data/filings/    │  │ claude-sonnet-4-6    │    │
│  │   {TICKER}.pdf   │  │ data/eval/       │  │ data/eval/judge_*    │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────┐
        │  External APIs                                   │
        │  OpenAI (gpt-4o, gpt-4o-mini, embeddings)        │
        │  Anthropic (judge)   SEC EDGAR (filings)         │
        │  Finnhub + FMP (consensus)   Tavily (news)       │
        └──────────────────────────────────────────────────┘
```

```
app.py (Streamlit — Single ticker + Queue tabs)
    ├── yoda/ingest/edgar.py             # SEC EDGAR fetch + disk cache (10-Q primary + 10-K supplemental)
    ├── yoda/ingest/chunker.py           # section-aware HTML → text chunks
    ├── yoda/retrieval/                  # text-embedding-3-small + ChromaDB
    ├── yoda/tools/consensus.py          # Finnhub + FMP backup
    ├── yoda/tools/news.py               # Tavily search
    ├── yoda/modes/personality_panel.py  # Phase 10: 6-personality panel
    ├── yoda/modes/tools.py              # tool registry shared across personalities
    ├── yoda/modes/rag_llm.py            # Legacy mode (kept as smoke test)
    ├── yoda/modes/agent.py              # Legacy mode (kept as smoke test)
    ├── yoda/modes/baseline.py           # Prompt-only baseline (eval lower bound)
    ├── yoda/queue/processor.py          # Batch processor: many tickers → ZIP
    ├── yoda/schema.py                   # EarningsReport + WatchItem Pydantic models
    ├── yoda/report/pdf.py               # EarningsReport → PDF (reportlab)
    └── yoda/eval/                       # Model-as-judge eval harness
```

---

## 3. Evaluation and Results

### Evaluation Scope

By default, `run_eval()` evaluates against a **curated 10-ticker universe** chosen for sector variety — one representative per sector/archetype so the eval captures diverse filing structures and business narratives:

| Ticker | Sector |
|---|---|
| AAPL | Tech — hardware + services |
| NVDA | Tech — semiconductors / AI |
| AMZN | Tech — cloud + e-commerce |
| JPM | Financials — banking |
| JNJ | Healthcare — pharma + medtech |
| XOM | Energy |
| KO | Consumer Staples |
| NFLX | Media / Streaming |
| CAT | Industrials |
| PANW | Cybersecurity / SaaS |

Each ticker runs against two modes: `baseline` and `yoda`. To evaluate a custom set:
```bash
python -m yoda.eval.runner NFLX AAPL JPM  # Custom tickers
python -m yoda.eval.runner NFLX            # Single ticker (quick test, ~$0.30, 3-5 min)
```

**Cost / Latency**: 10 tickers × 2 modes ≈ $1.50 per full run, 25–30 min. Single ticker is ~$0.15 and 2–4 min for verification.

### Modes Compared

- **`baseline`** — Prompt-only (`yoda/modes/baseline.py`): a single `gpt-4o` call with a manually sliced ~5000-character excerpt from the filing (MD&A preferred, Financial Statements fallback). No RAG, no agent loop. This represents the "just give the LLM a chunk of the filing" approach and sets the lower bound.
- **`yoda`** — Full personality panel with OpenAI `text-embedding-3-small` for retrieval. Vectors stored in the `filings_openai` Chroma collection.

### Rubric

Reports were scored by `claude-sonnet-4-6` through `yoda/eval/judge.py` (cross-family to prevent self-grading bias). Judge outputs are cached under `data/eval/judge_cache/` so re-runs are free; a manifest written per run lets `prune_judge_cache(keep_runs=2)` keep only the two most recent runs' cached files. Each report is scored on five dimensions, rated 1–5:

| Dimension | What it measures |
|---|---|
| **Extraction completeness** | Did the report extract the key facts actually available in the filing? |
| **Accuracy** | Are the figures correct relative to the source filing? |
| **Source traceability** | Do citations resolve to real sections or chunks? |
| **Relevance** | Is the content focused on pre-earnings analysis? |
| **Usefulness** | Would a sell-side analyst find this actionable before an earnings call? |

### Test Set

Ten US-listed stocks, one per sector/archetype (see the table above), all with at least 3 years of public trading history. See [`CURATED_TICKERS` in `yoda/eval/runner.py`](yoda/eval/runner.py) for the source list.

### Results

See [`data/eval/summary.md`](data/eval/summary.md) and [`data/eval/comparison.png`](data/eval/comparison.png) for the latest mean scores across baseline and yoda. The chart compares both modes on the five rubric dimensions.

![Mean rubric scores by mode — Baseline vs Yoda](data/eval/comparison.png)


Full per-ticker results are in [`data/eval/results.csv`](data/eval/results.csv) and [`data/eval/summary.md`](data/eval/summary.md).

---

## 4. Artifact Snapshot

### UI Flow

The app has two tabs: **Single ticker** (interactive, one report at a time) and **Queue (batch)** (paste many tickers, generate overnight, download all PDFs as a ZIP).

```
┌─────────────────────────────────────────────────┐
│  Yoda — Pre-Earnings Research Assistant          │
│  [ Single ticker ] [ Queue (batch) ]             │
│                                                  │
│  Ticker  [  NFLX                  ]              │
│                                                  │
│  [  Generate Report  ]                           │
└─────────────────────────────────────────────────┘
           ↓ (on click)
┌─────────────────────────────────────────────────┐
│  ⏳ Generating report for NFLX...                │
│  ┌───────────────────────────────────────────┐  │
│  │ [panel] Fetching filing for NFLX...       │  │
│  │ [panel] Filing: 10-Q 2026-04-18           │  │
│  │ [panel] Supplemental: 10-K 2026-...       │  │
│  │ [panel] Phase 2: 6 personalities...       │  │
│  │ [Optimist] iter 1: retrieve_filing(...)   │  │
│  │ [Pessimist] iter 1: search_news(...)      │  │
│  │ [Quant]    iter 2: lookup_peer('DIS',...) │  │
│  │ [panel] Phase 4: synthesizing...          │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
           ↓ (complete)
┌─────────────────────────────────────────────────┐
│  NFLX — Netflix, Inc.                            │
│  10-Q filed 2026-04-18 · 10-K filed 2026-01-29   │
│                                                  │
│  [  Download PDF Report  ]                       │
│                                                  │
│  Pre-Earnings Watchlist                          │
│   **Ad-tier ARPU:** ...                          │
│   -> Monitor ad-tier ARPU disclosure             │
│   - https://reuters.com/...                      │
│   - https://wsj.com/...                          │
│                                                  │
│  ▶ Bull Case / Bear Case                         │
│  ▶ Recent News (8)                               │
│  ▶ Data Gaps (2)                                 │
└─────────────────────────────────────────────────┘
```

### Sample Report Fields (NFLX)

```json
{
  "ticker": "NFLX",
  "company_name": "Netflix, Inc.",
  "filing_type": "10-Q",
  "filing_date": "2026-04-18",
  "supplemental_filing_type": "10-K",
  "supplemental_filing_date": "2026-01-29",
  "key_metrics": [
    {
      "name": "Total Revenue",
      "value": "$12,249,757K",
      "unit": "",
      "source_citation": "Financial Statements — Consolidated Statements of Operations"
    }
  ],
  "forward_guidance": {
    "text": "Management expects continued margin expansion driven by advertising tier growth...",
    "source_citation": "MD&A — Outlook"
  },
  "what_to_watch": [
    {
      "text": "**Ad-tier ARPU:** Advertising-tier ARPU grew 18% YoY against a backdrop of cooling subscriber adds. However, mix-shift toward lower-priced ad-supported plans could mask underlying pricing softness.\n\n-> Monitor ad-tier ARPU disclosure and ad-impression growth commentary.",
      "relevant_urls": [
        "https://www.reuters.com/business/media-telecom/netflix-ad-tier-growth-2026-04-18",
        "https://www.wsj.com/articles/netflix-advertising-arpu-2026"
      ]
    }
  ],
  "data_gaps": [
    "Paid net additions not disclosed — subscriber metric absent from MD&A this quarter"
  ]
}
```

### PDF Output Structure

The downloaded PDF contains: cover page (ticker, company name, primary filing date — and the supplemental 10-K date when both filings are within the 92-day window), the Pre-Earnings Watchlist (each entry followed by clickable "Sources:" URLs when the synthesizer found relevant news in the pool), key metrics table, revenue segments table, forward guidance blockquote, key risks (new risks flagged in red), analyst consensus, recent news with hyperlinks, bull/bear bullets, and data gaps in amber.

---

## 5. Setup and Usage

### Prerequisites

- Python 3.11+
- Conda (recommended) or virtualenv
- API keys for OpenAI, Anthropic, Finnhub, FMP, and Tavily (all have free tiers sufficient for testing)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd Yoda

# Create and activate a conda environment (or use virtualenv)
conda create -n yoda python=3.11
conda activate yoda

# Install dependencies
pip install -r requirements.txt

# Copy the env template and fill in your keys
cp .env.example .env
```

Edit `.env` with your keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
FINNHUB_API_KEY=...
FMP_API_KEY=...
TAVILY_API_KEY=tvly-...
SEC_USER_AGENT="Your Name your@email.com"
```

### Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter a ticker (e.g. `NFLX`) and click **Generate Report**. For batch runs, switch to the **Queue (batch)** tab, paste multiple tickers (one per line or comma-separated), and run them sequentially — PDFs save to `data/reports/` as they complete and bundle into a downloadable ZIP at the end.

The first run for a ticker fetches and indexes the SEC filings — both the 10-Q (primary) and the 10-K (supplemental) when both are within the 92-day freshness window. Subsequent runs for the same ticker are fast because the filings and ChromaDB index are cached to disk.

### Run the Evaluation Harness

```bash
# Single ticker (cheap verification, ~$0.30, 3–5 min)
python -m yoda.eval.runner NFLX

# Full curated 10-ticker run (~$3, 15–20 min)
python -m yoda.eval.runner
```

Outputs `data/eval/results.csv`, `data/eval/summary.md`, and `data/eval/comparison.png`. Judge results are cached in `data/eval/judge_cache/` so repeated runs are free; only the two most recent runs' cached files are kept on disk.

Cache cleanup at the end of every run:
- `data/eval/{mode}_{TICKER}.json` — one report JSON per `(mode, ticker)` (overwrites older copies)
- `data/eval/judge_cache/` — JSON cache files from the two most recent runs only (older are pruned via run manifests)
- `data/reports/{TICKER}_{ts}.pdf` — one PDF per ticker (older are deleted)
- `data/queue_zips/yoda_queue_{ts}.zip` — at most the two most recent ZIP bundles

### Run Individual Mode Smoke Tests

```bash
# Multi-agent personality panel — current production mode
python -m yoda.modes.personality_panel NFLX

# Legacy modes (kept as harnesses; not used by the Streamlit app)
python -m yoda.modes.baseline
python -m yoda.modes.rag_llm
python -m yoda.modes.agent

# PDF generation from a saved report JSON
python -m yoda.report.pdf NFLX
```

---

## Required API Keys

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM generation (`gpt-4o`) and cheap iterative steps (`gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Yes | Model-as-judge in the evaluation harness (`claude-sonnet-4-6`) |
| `FINNHUB_API_KEY` | Yes | Analyst consensus estimates (primary source) |
| `FMP_API_KEY` | Yes | Financial Modeling Prep — backup source when Finnhub returns no consensus |
| `TAVILY_API_KEY` | Yes | News and web search |
| `SEC_USER_AGENT` | Yes | SEC EDGAR requires a contact string, e.g. `"Name email@example.com"` |

---

## Repo Structure

```
yoda/
├── app.py                              # Streamlit entry point (Single ticker + Queue tabs)
├── requirements.txt
├── .env.example
├── yoda/
│   ├── config.py                       # env vars, model names, constants
│   ├── ingest/
│   │   ├── edgar.py                    # ticker -> SEC EDGAR fetch (10-Q primary + 10-K supplemental)
│   │   └── chunker.py                  # section-aware chunking
│   ├── retrieval/
│   │   ├── embeddings.py               # OpenAI text-embedding-3-small
│   │   └── vector_store.py             # ChromaDB wrapper — filings_openai collection
│   ├── tools/
│   │   ├── consensus.py                # Finnhub + FMP backup
│   │   └── news.py                     # Tavily wrapper
│   ├── modes/
│   │   ├── personality_panel.py        # Phase 10: 6-personality panel
│   │   ├── tools.py                    # Tool registry (retrieve_filing, search_news, lookup_peer)
│   │   ├── baseline.py                 # Prompt-only baseline (eval lower bound)
│   │   ├── rag_llm.py                  # Legacy Mode 1 (kept as smoke test)
│   │   └── agent.py                    # Legacy Mode 2 (kept as smoke test)
│   ├── queue/
│   │   └── processor.py                # Batch queue processor + ZIP bundler
│   ├── schema.py                       # EarningsReport + WatchItem Pydantic models
│   ├── report/
│   │   └── pdf.py                      # EarningsReport -> PDF (reportlab)
│   └── eval/
│       ├── rubric.py                   # rubric Pydantic models
│       ├── judge.py                    # claude-sonnet-4-6 model-as-judge
│       └── runner.py                   # batch eval over test tickers
├── data/
│   ├── filings/{TICKER}/               # cached primary + supplemental HTML + latest.json (gitignored)
│   ├── chroma/                         # ChromaDB persistence — filings_openai collection (gitignored)
│   ├── reports/                        # PDFs written by the Queue processor, 1 per ticker (gitignored)
│   ├── queue_zips/                     # ZIP bundles from queue runs, max 2 most recent (gitignored)
│   └── eval/                           # eval outputs: results.csv, summary.md, comparison.png, judge_cache/ (last 2 runs)
├── .claude/worktrees/                  # working trees from /worktree development (gitignored)
└── tests/
    └── test_smoke.py
```

---

## APIs and Keys

API keys are stored as environment variables (`.env`) and excluded from the repo via `.gitignore`. See `.env.example` for the full template.
