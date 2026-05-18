"""Tool registry for the multi-agent personality panel.

Three tools, each a plain Python function, registered with an OpenAI
function-calling schema so personality agents can choose to call them inside
their tool-use loop. The agents see the same tool surface regardless of which
personality they embody — what differs is the system prompt that shapes how
they use the tools.

Tools:
  retrieve_filing(query, k)         — semantic search on the primary ticker's chunks
  search_news(query, max_results)   — Tavily web search
  lookup_peer(peer_ticker, query, k) — fetch+chunk+search a competitor filing

A `ToolContext` instance carries the per-run state that the tool functions
need (primary ticker, accession number, shared ChromaStore, news pool, and a
lock for peer-ingestion serialization). The personality_panel module builds
the context once per ticker and passes it into every tool invocation.
"""

import threading
import time

from yoda.ingest.edgar import fetch_latest_filing
from yoda.ingest.chunker import chunk_filing, Chunk
from yoda.retrieval.embeddings import embed_texts
from yoda.retrieval.vector_store import ChromaStore
from yoda.tools.news import search_news as _tavily_search


# ---------------------------------------------------------------------------
# Tool context — per-run state shared with each tool invocation
# ---------------------------------------------------------------------------

class ToolContext:
    # Holds the state every tool needs to do its job. One instance is built
    # in Phase 1 of personality_panel.run() and passed unchanged into every
    # tool invocation across all personality threads.
    def __init__(
        self,
        primary_ticker: str,
        primary_accession: str,
        store: ChromaStore,
        supplemental_accession: str | None = None,
    ) -> None:
        self.primary_ticker = primary_ticker
        self.primary_accession = primary_accession
        self.supplemental_accession = supplemental_accession  # 10-K when primary is 10-Q
        self.store = store

        # Tracks which peer tickers have already been ingested into the store
        # so a second lookup_peer call for the same peer skips re-ingestion.
        # Protected by _peer_lock so concurrent personalities don't race on
        # the same peer's first-call ingestion.
        self._ingested_peers: dict[str, str] = {}   # peer_ticker -> accession
        self._peer_lock = threading.Lock()

        # Accumulator for news items collected across all personalities so the
        # synthesis step has the full pool with URLs intact. Personality threads
        # append; the synthesizer reads after Phase 2 completes.
        self.news_pool: list[dict] = []
        self._news_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Tool 1 — retrieve_filing
# ---------------------------------------------------------------------------

def retrieve_filing(ctx: ToolContext, query: str, k: int = 5) -> list[Chunk]:
    """Semantic search over the primary ticker's filing chunks.

    Queries both the primary filing (10-Q) and the supplemental filing (10-K)
    when both are available, returning up to 2k chunks so personalities see
    the freshest quarterly data alongside annual context.
    """
    # ChromaStore.query is safe to call concurrently from multiple personality
    # threads since reads don't mutate the on-disk index.
    chunks = ctx.store.query(ctx.primary_accession, query, k=k)
    if ctx.supplemental_accession:
        chunks += ctx.store.query(ctx.supplemental_accession, query, k=k)
    return chunks


# ---------------------------------------------------------------------------
# Tool 2 — search_news
# ---------------------------------------------------------------------------

def search_news(ctx: ToolContext, query: str, max_results: int = 3) -> list[dict]:
    """Tavily news search. Results are also appended to the shared news_pool
    so the synthesizer can see every URL any personality discovered.
    """
    # Delegate to the existing Tavily wrapper (yoda.tools.news.search_news).
    results = _tavily_search(query, max_results=max_results)

    # Append to the shared pool under a lock so parallel personalities don't
    # corrupt the list. We don't dedupe here — the synthesizer handles that.
    with ctx._news_lock:
        ctx.news_pool.extend(results)

    return results


# ---------------------------------------------------------------------------
# Tool 3 — lookup_peer
# ---------------------------------------------------------------------------

def lookup_peer(ctx: ToolContext, peer_ticker: str, query: str, k: int = 3) -> list[Chunk]:
    """Fetch + chunk + embed a peer company's filing (cached on disk after
    first call), then return the k chunks most relevant to *query*.

    Uses an internal lock to ensure two personalities asking for the same
    peer simultaneously don't both trigger the SEC download.
    """
    peer_ticker = peer_ticker.upper().strip()

    # First check: is this peer already ingested into our ChromaStore?
    # We use the per-context lock so the check + ingest pair is atomic.
    with ctx._peer_lock:
        if peer_ticker not in ctx._ingested_peers:
            # Fetch is cached on disk by yoda.ingest.edgar so repeat calls
            # for the same peer across runs avoid the SEC round-trip too.
            peer_filing = fetch_latest_filing(peer_ticker)
            peer_chunks = chunk_filing(peer_filing["clean_text"], peer_filing["raw_html"])
            peer_texts = [c.text for c in peer_chunks]
            peer_embeds = embed_texts(peer_texts)
            ctx.store.upsert(peer_filing["accession_number"], peer_chunks, peer_embeds)
            ctx._ingested_peers[peer_ticker] = peer_filing["accession_number"]

        peer_accession = ctx._ingested_peers[peer_ticker]

    # Query is read-only, safe outside the lock.
    return ctx.store.query(peer_accession, query, k=k)


# ---------------------------------------------------------------------------
# OpenAI tool schemas — passed to chat.completions.create(tools=[...])
# ---------------------------------------------------------------------------

# Each entry follows OpenAI's function-calling schema. The "name" must match
# a key in TOOL_DISPATCH below so the loop can route the model's tool_call
# back to the right Python function.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_filing",
            "description": (
                "Semantic search over the primary company's most recent 10-Q "
                "or 10-K. Returns chunks of filing text relevant to your query. "
                "Use this for facts directly stated in the filing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in natural language.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many chunks to return (default 5, max 10).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search the web for recent news about the company or its industry. "
                "Use this for sentiment, competitor moves, regulatory developments, "
                "or events that happened after the filing date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in natural language.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many articles to return (default 3, max 5).",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_peer",
            "description": (
                "Fetch a competitor's most recent SEC filing and search it for "
                "evidence on a topic. Use this to compare the primary company "
                "against a named peer (e.g., DIS for NFLX, BX for COIN)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "peer_ticker": {
                        "type": "string",
                        "description": "The peer company's ticker symbol (e.g., 'DIS').",
                    },
                    "query": {
                        "type": "string",
                        "description": "What to search for in the peer's filing.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many chunks to return (default 3, max 5).",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": ["peer_ticker", "query"],
                "additionalProperties": False,
            },
        },
    },
]


# Maps the tool name (as returned in the model's tool_call) to the Python
# function that implements it. The dispatcher in personality_panel.py looks
# up the function here and passes (ctx, **args) into it.
TOOL_DISPATCH = {
    "retrieve_filing": retrieve_filing,
    "search_news":     search_news,
    "lookup_peer":     lookup_peer,
}


# ---------------------------------------------------------------------------
# Helper: normalize a tool call for repetition detection
# ---------------------------------------------------------------------------

def normalize_call(tool_name: str, args: dict) -> str:
    # Build a stable string key for repetition detection. We only consider the
    # tool name + main query so callers asking the same question with k=3 vs
    # k=5 are still recognised as duplicates.
    if tool_name == "retrieve_filing":
        return f"retrieve_filing:{args.get('query', '').strip().lower()}"
    if tool_name == "search_news":
        return f"search_news:{args.get('query', '').strip().lower()}"
    if tool_name == "lookup_peer":
        peer = args.get("peer_ticker", "").upper().strip()
        return f"lookup_peer:{peer}:{args.get('query', '').strip().lower()}"
    # Unknown tool — return a non-empty key so duplicate-unknowns are caught.
    return f"{tool_name}:{repr(sorted(args.items()))}"


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.tools [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()

    # Build a real context: fetch + chunk + embed + upsert the primary filing.
    print(f"[tools-smoke] Ingesting {ticker} so tools have something to query...")
    filing = fetch_latest_filing(ticker)
    chunks = chunk_filing(filing["clean_text"], filing["raw_html"])
    embeds = embed_texts([c.text for c in chunks])

    store = ChromaStore()
    store.upsert(filing["accession_number"], chunks, embeds)

    ctx = ToolContext(
        primary_ticker=ticker,
        primary_accession=filing["accession_number"],
        store=store,
    )
    print(f"[tools-smoke] Indexed {len(chunks)} chunks for {ticker} "
          f"({filing['filing_type']} {filing['filing_date']})")

    # Exercise each tool once and print a one-line summary.
    print("\n[tools-smoke] retrieve_filing('revenue segment breakdown'):")
    t0 = time.perf_counter()
    rf = retrieve_filing(ctx, "revenue segment breakdown", k=3)
    print(f"  -> {len(rf)} chunks in {time.perf_counter() - t0:.2f}s; "
          f"first section: {rf[0].section if rf else '(none)'}")

    print("\n[tools-smoke] search_news('Netflix earnings'):")
    t0 = time.perf_counter()
    sn = search_news(ctx, f"{ticker} earnings", max_results=2)
    print(f"  -> {len(sn)} articles in {time.perf_counter() - t0:.2f}s")
    for r in sn:
        print(f"    {r['title'][:80]}  ({r['url']})")

    # Pick a sensible peer for the smoke test by ticker. DIS is a reasonable
    # default for media tickers; HOOD pairs with COIN; we fall back to MSFT.
    peer = {"NFLX": "DIS", "COIN": "HOOD", "PANW": "CRWD"}.get(ticker, "MSFT")
    print(f"\n[tools-smoke] lookup_peer('{peer}', 'revenue trends'):")
    t0 = time.perf_counter()
    lp = lookup_peer(ctx, peer, "revenue trends", k=2)
    print(f"  -> {len(lp)} chunks from {peer} in {time.perf_counter() - t0:.2f}s")

    # Verify the news_pool was populated by the search_news call.
    print(f"\n[tools-smoke] ctx.news_pool has {len(ctx.news_pool)} items collected")

    # Verify the repetition normalizer.
    key1 = normalize_call("retrieve_filing", {"query": "Revenue Segment Breakdown"})
    key2 = normalize_call("retrieve_filing", {"query": "revenue segment breakdown"})
    assert key1 == key2, "normalize_call must be case-insensitive"
    print(f"[tools-smoke] normalize_call deduplicates case + whitespace: OK")
