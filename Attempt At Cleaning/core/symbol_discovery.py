"""News-driven symbol discovery (Stage 6) — the "find a promising stock" front end.

The loop: grab recent news (<= as_of) -> extract candidate tickers -> rank ->
hand the top symbol(s) to the swarm. This module is the extraction + ranking
core. The pure logic (text -> ranked tickers) is testable offline; the news
fetch is INJECTED (`search_fn`) so historical runs stay point-in-time (we drop
any item published after as_of).

Correctness focus — AVOID false-positive tickers. All-caps tokens like CEO, USA,
AI, ETF, GDP, CEO are NOT tickers. We only accept a candidate from:
  1. a cashtag           ($NVDA)
  2. an explicit exchange tag   (NASDAQ: NVDA)
  3. a bare all-caps token that is in a provided `universe` (tradable set)
  4. a company name resolved via a provided name->ticker map (Walmart -> WMT)
A stoplist rejects common non-ticker acronyms even if a sloppy universe includes
them. Without a `universe`, bare tokens are NOT accepted (too risky).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional


# Common all-caps tokens that look like tickers but aren't. Rejected even if a
# caller's universe accidentally contains them.
_STOPLIST: frozenset[str] = frozenset({
    "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "ETFS", "GDP", "CPI", "PPI",
    "USA", "US", "USD", "EUR", "GBP", "EU", "UK", "UN", "AI", "ML", "EV",
    "PE", "EPS", "ROI", "ROE", "SEC", "FDA", "FED", "FOMC", "IRS", "DOJ",
    "Q1", "Q2", "Q3", "Q4", "FY", "YOY", "QOQ", "TTM", "EBIT", "EBITDA",
    "NYSE", "NASDAQ", "AMEX", "OTC", "IPOS", "M&A", "API", "CES", "GAAP",
    "NEW", "NOW", "ALL", "FOR", "ARE", "CAN", "USA", "TV", "PC", "OS",
})

_CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
_EXCHANGE = re.compile(
    r"\b(?:NYSE|NASDAQ|Nasdaq|NYSEARCA|AMEX|OTC)\s*[:\-]?\s*([A-Z]{1,5})\b")
_BARE = re.compile(r"\b([A-Z]{1,5})\b")


@dataclass
class Candidate:
    ticker: str
    score: float = 0.0
    mentions: int = 0
    sources: set = field(default_factory=set)


def _item_fields(item) -> tuple[str, Optional[date], str]:
    """Normalise a news item (dict or NewsItem-like) -> (title, published, source)."""
    if isinstance(item, dict):
        return (str(item.get("title", "")),
                item.get("published"),
                str(item.get("source", "")))
    return (str(getattr(item, "title", "")),
            getattr(item, "published", None),
            str(getattr(item, "source", "")))


def _recency_weight(published: Optional[date], ref: Optional[date]) -> float:
    """1.0 base; up to +0.5 for items within ~30 days of the reference date."""
    if published is None or ref is None:
        return 1.0
    days = (ref - published).days
    if days < 0:
        return 1.0  # future items are dropped upstream; be safe here too
    return 1.0 + max(0.0, 1.0 - days / 30.0) * 0.5


def discover_from_news(
    items,
    *,
    universe: Optional[set] = None,
    name_to_ticker: Optional[dict] = None,
    recency_ref: Optional[date] = None,
    top_k: int = 5,
) -> list[Candidate]:
    """Extract + rank ticker candidates from a list of news items (pure).

    `universe` (uppercase tickers) gates bare-token acceptance and filters all
    candidates to the tradable set when provided. `name_to_ticker` maps lowercased
    company names/aliases to tickers.
    """
    uni = {t.upper() for t in universe} if universe else None
    name_map = {k.lower(): v.upper() for k, v in (name_to_ticker or {}).items()}
    name_patterns = [
        (re.compile(r"\b" + re.escape(name) + r"\b"), tic)
        for name, tic in name_map.items()
    ]

    cands: dict[str, Candidate] = {}

    def _add(tic: str, weight: float, source: str) -> None:
        tic = tic.upper()
        if tic in _STOPLIST:
            return
        if uni is not None and tic not in uni:
            return
        c = cands.get(tic) or Candidate(ticker=tic)
        c.mentions += 1
        c.score += weight
        if source:
            c.sources.add(source)
        cands[tic] = c

    for item in items:
        title, published, source = _item_fields(item)
        if not title:
            continue
        w = _recency_weight(published, recency_ref)
        low = title.lower()

        # 1. cashtags
        for m in _CASHTAG.finditer(title):
            _add(m.group(1), w, source)
        # 2. explicit exchange tags
        for m in _EXCHANGE.finditer(title):
            _add(m.group(1), w, source)
        # 3. company-name resolution
        for pat, tic in name_patterns:
            if pat.search(low):
                _add(tic, w, source)
        # 4. bare all-caps tokens — ONLY when validated by a universe
        if uni is not None:
            for m in _BARE.finditer(title):
                tok = m.group(1)
                if tok in uni and tok not in _STOPLIST:
                    _add(tok, w, source)

    ranked = sorted(cands.values(),
                    key=lambda c: (c.score, c.mentions, c.ticker),
                    reverse=True)
    return ranked[:top_k]


def discover_symbols(
    as_of: date,
    search_fn: Callable[[str], list],
    *,
    universe: Optional[set] = None,
    name_to_ticker: Optional[dict] = None,
    queries: Optional[list[str]] = None,
    top_k: int = 5,
) -> list[str]:
    """Fetch recent news via `search_fn` and return ranked candidate tickers.

    `search_fn(query) -> list[items]` is the INJECTED network boundary (DDG
    adapter, or a frozen reader for backtests). Every item published AFTER as_of
    is dropped (point-in-time). Returns ticker strings, most promising first.
    """
    queries = queries or [
        "stocks to watch today", "biggest stock movers", "analyst upgrades",
        "earnings beat", "stock surges",
    ]
    items: list = []
    for q in queries:
        try:
            results = search_fn(q) or []
        except Exception:
            results = []
        for it in results:
            _, published, _ = _item_fields(it)
            if published is not None and published > as_of:
                continue  # point-in-time: never see the future
            items.append(it)

    candidates = discover_from_news(
        items, universe=universe, name_to_ticker=name_to_ticker,
        recency_ref=as_of, top_k=top_k)
    return [c.ticker for c in candidates]
