"""Stock data layer — point-in-time snapshots, lenses, look-ahead guard.

This module is the swarm's ground-truth interface. THE single biggest
correctness risk in the whole Stock Swarm is look-ahead bias: a run as of date
T must never see data dated after T. `assert_no_lookahead()` is the loud guard
for that; treat a failure as a stop-the-line defect, not a warning.

What's REAL here (do not water down):
  * Snapshot dataclass + per-field provenance dating
  * LENS_FIELDS partitioning (the diversity engine for a single ticker)
  * assert_no_lookahead() — the point-in-time guard
  * FrozenSnapshotProvider — reads a frozen JSON snapshot; this is what the
    historical-DB backtest uses, and it is fully functional + tested.

What's SCAFFOLDED (TODO, see STOCK_SWARM_POC_PROMPT.md Stage 2):
  * YFinanceProvider._fetch_raw — the live yfinance network call. Stage 0 must
    confirm yfinance is not deprecated before this is fleshed out. The clamp +
    assertion wrapper around it is already real.

The grader-only "reality" lookup (realized forward return) deliberately does
NOT live here — it is in eval/ground_truth.py so no agent can import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


class LookaheadError(AssertionError):
    """Raised when a snapshot field is dated after its as_of date."""


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

# Lens -> the snapshot fields a scout assigned to that lens may see. This is the
# stock analog of disjoint corpus partitions: each scout conditions on ONE
# lens's facts only. The lens name becomes the scout's partition_id.
LENS_FIELDS: dict[str, list[str]] = {
    "valuation":          ["price", "pe", "fwd_pe", "peg", "ps", "ev_ebitda",
                           "analyst_target", "market_cap"],
    "growth":             ["rev_growth_yoy", "eps_growth_yoy", "eps",
                           "gross_margin", "operating_margin", "net_margin"],
    "technical":          ["price", "sma50", "sma200", "week52_high",
                           "week52_low"],
    "news_sentiment":     ["news"],
    "risk_balance_sheet": ["debt_to_equity", "fcf_margin", "div_yield",
                           "market_cap"],
}

LENSES: tuple[str, ...] = tuple(LENS_FIELDS.keys())

# Numeric metric fields (everything the DataValidator can verify).
_METRIC_FIELDS = (
    "price", "market_cap", "pe", "fwd_pe", "peg", "ps", "ev_ebitda", "eps",
    "rev_growth_yoy", "eps_growth_yoy", "gross_margin", "operating_margin",
    "net_margin", "fcf_margin", "div_yield", "week52_high", "week52_low",
    "sma50", "sma200", "analyst_target",
)


@dataclass(frozen=True)
class NewsItem:
    title: str
    published: date
    url: str = ""
    source: str = ""


@dataclass
class Snapshot:
    """Point-in-time fact sheet for one ticker as of `as_of`.

    All numeric fields use the units in core.stock_verify.METRIC_CLASS:
    ratios plain, percents in percentage-points, prices/market_cap in dollars.
    `provenance` maps a field name -> the date that value is effective. Every
    provenance date and every news.published MUST be <= as_of.
    """
    ticker: str
    as_of: date
    # numeric facts (None = unavailable for this ticker/date)
    price: Optional[float] = None
    market_cap: Optional[float] = None
    pe: Optional[float] = None
    fwd_pe: Optional[float] = None
    peg: Optional[float] = None
    ps: Optional[float] = None
    ev_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    eps: Optional[float] = None
    rev_growth_yoy: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    fcf_margin: Optional[float] = None
    div_yield: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    sma50: Optional[float] = None
    sma200: Optional[float] = None
    analyst_target: Optional[float] = None
    news: list[NewsItem] = field(default_factory=list)
    provenance: dict[str, date] = field(default_factory=dict)

    # ---- ground-truth lookup for the DataValidator ----------------------
    def get(self, metric: str) -> Optional[float]:
        """Return the ground-truth value for a metric, or None if unavailable."""
        if metric not in _METRIC_FIELDS:
            return None
        return getattr(self, metric, None)

    def present_metrics(self) -> list[str]:
        return [m for m in _METRIC_FIELDS if getattr(self, m, None) is not None]

    # ---- lens partitioning (prompt rendering for one scout) -------------
    def lens_facts(self, lens: str) -> str:
        """Render only the facts belonging to `lens`, as scout-prompt text.

        Enforces lens partitioning at the data layer: a valuation scout never
        sees growth fields, etc. Numbers carry an [as-of] stamp so the agent
        habit of dating every figure is grounded in what it's shown.
        """
        names = LENS_FIELDS.get(lens, [])
        lines: list[str] = []
        for name in names:
            if name == "news":
                for it in self.news[:6]:
                    lines.append(f"- [{it.published.isoformat()}] {it.title}")
                continue
            val = getattr(self, name, None)
            if val is None:
                continue
            lines.append(f"- {name} = {_fmt(name, val)}")
        if not lines:
            return f"(no {lens} facts available for {self.ticker} as of {self.as_of.isoformat()})"
        header = f"{self.ticker} — {lens} facts as of {self.as_of.isoformat()}:"
        return header + "\n" + "\n".join(lines)


def _fmt(metric: str, val: float) -> str:
    from core.stock_verify import METRIC_CLASS
    cls = METRIC_CLASS.get(metric, "ratio")
    if cls == "percent":
        return f"{val:.2f}%"
    if cls == "money":
        return f"${val:,.0f}"
    if cls == "price":
        return f"${val:,.2f}"
    return f"{val:.2f}"


# ---------------------------------------------------------------------------
# Look-ahead guard (the existential-risk assertion)
# ---------------------------------------------------------------------------

def assert_no_lookahead(snap: Snapshot) -> None:
    """Fail loudly if any field in `snap` is dated after `snap.as_of`.

    Checks every provenance date and every news.published. This is the guard
    that keeps backtest results honest — if it fires, a data source leaked
    future information into a historical run.
    """
    asof = snap.as_of
    for fld, d in snap.provenance.items():
        if d is not None and d > asof:
            raise LookaheadError(
                f"[lookahead] {snap.ticker} field {fld!r} dated {d} > as_of {asof}"
            )
    for it in snap.news:
        if it.published > asof:
            raise LookaheadError(
                f"[lookahead] {snap.ticker} news {it.title[:40]!r} "
                f"published {it.published} > as_of {asof}"
            )


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StockDataProvider(Protocol):
    def get_snapshot(self, symbol: str, as_of: date) -> Snapshot: ...
    def discover_symbols(self, as_of: date, universe: Optional[str] = None) -> list[str]: ...


# ---------------------------------------------------------------------------
# FrozenSnapshotProvider — REAL. Reads point-in-time JSON (the historical DB).
# ---------------------------------------------------------------------------

class FrozenSnapshotProvider:
    """Reads frozen snapshots from a directory of JSON files.

    Layout (the historical DB the project owner builds — Stage 4):
        <root>/<TICKER>/<YYYY-MM-DD>.json   # one snapshot per (ticker, as_of)
        <root>/_discovery/<YYYY-MM-DD>.json # {"symbols": ["NVDA", ...]} per date

    Every snapshot is validated by assert_no_lookahead() on load, so a
    mis-dated DB row fails fast rather than silently inflating the backtest.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get_snapshot(self, symbol: str, as_of: date) -> Snapshot:
        path = self.root / symbol.upper() / f"{as_of.isoformat()}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no frozen snapshot for {symbol} @ {as_of} at {path}"
            )
        snap = snapshot_from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert_no_lookahead(snap)
        return snap

    def discover_symbols(self, as_of: date, universe: Optional[str] = None) -> list[str]:
        path = self.root / "_discovery" / f"{as_of.isoformat()}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("symbols", []))


def snapshot_from_dict(d: dict) -> Snapshot:
    """Build a Snapshot from a JSON-decoded dict (dates as ISO strings)."""
    as_of = _parse_date(d["as_of"])
    news = [
        NewsItem(
            title=n["title"],
            published=_parse_date(n["published"]),
            url=n.get("url", ""),
            source=n.get("source", ""),
        )
        for n in d.get("news", [])
    ]
    provenance = {k: _parse_date(v) for k, v in d.get("provenance", {}).items()}
    kwargs = {"ticker": d["ticker"].upper(), "as_of": as_of,
              "news": news, "provenance": provenance}
    for m in _METRIC_FIELDS:
        if m in d and d[m] is not None:
            kwargs[m] = float(d[m])
    return Snapshot(**kwargs)


def _parse_date(v) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# YFinanceProvider — SCAFFOLD. Live fetch is Stage 2 (after Stage 0 confirms
# yfinance is not deprecated). The clamp + assert wrapper is already real so a
# fleshed-out _fetch_raw inherits point-in-time safety for free.
# ---------------------------------------------------------------------------

class YFinanceProvider:
    """Live yfinance-backed provider.

    Use only when `as_of == today` (yfinance fundamentals/news are 'as of now',
    so they are look-ahead-unsafe for historical T — historical runs MUST use
    FrozenSnapshotProvider). get_snapshot() enforces this and still runs
    assert_no_lookahead() as a backstop.
    """

    def __init__(self, cache_dir: str | Path = "stock_cache"):
        self.cache_dir = Path(cache_dir)

    def get_snapshot(self, symbol: str, as_of: date) -> Snapshot:
        if as_of != date.today():
            raise ValueError(
                f"YFinanceProvider is live-only; historical as_of {as_of} must "
                f"use FrozenSnapshotProvider to avoid look-ahead bias."
            )
        cached = self._read_cache(symbol, as_of)
        if cached is not None:
            assert_no_lookahead(cached)
            return cached
        raw = self._fetch_raw(symbol, as_of)            # <-- scaffold
        snap = self._raw_to_snapshot(symbol, as_of, raw)
        assert_no_lookahead(snap)
        self._write_cache(snap)
        return snap

    def discover_symbols(self, as_of: date, universe=None) -> list[str]:
        """Live news-driven discovery via the search layer (best-effort).

        Only for as_of==today (live): web results carry no reliable publish
        dates, so historical discovery MUST use FrozenSnapshotProvider (which
        reads dated `_discovery/<as_of>.json`). Without a `universe` set, only
        cashtags/exchange-tags/known names are accepted (the safe default for
        noisy web text — see core/symbol_discovery.py)."""
        if as_of != date.today():
            return []
        try:
            from core.search_tool import search as _search
            from core.symbol_discovery import discover_symbols as _disc
        except Exception:
            return []

        def search_fn(q):
            try:
                chunks = _search(q, max_results=8, task_type="stock") or []
            except Exception:
                return []
            out = []
            for c in chunks:
                title = ((getattr(c, "source_tag", "") or "") + " "
                         + (getattr(c, "text", "") or ""))[:500]
                out.append({"title": title, "published": None})
            return out

        uni = universe if isinstance(universe, (set, frozenset)) else None
        return _disc(as_of, search_fn, universe=uni, top_k=8)

    # ---- live fetch (Stage 0 finding: Yahoo rate-limits hard — go via cache,
    #      use a browser session, back off; never hammer) ------------------
    def _fetch_raw(self, symbol: str, as_of: date) -> dict:
        import time
        import random
        import yfinance as yf

        sess = _make_session()
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                tk = yf.Ticker(symbol, session=sess) if sess else yf.Ticker(symbol)
                info = dict(tk.info or {})
                fast: dict = {}
                try:
                    f = tk.fast_info
                    for k in ("last_price", "market_cap", "year_high", "year_low",
                              "fifty_day_average", "two_hundred_day_average"):
                        try:
                            fast[k] = getattr(f, k)
                        except Exception:
                            pass
                except Exception:
                    pass
                news: list[dict] = []
                try:
                    for n in (tk.news or [])[:8]:
                        news.append(_normalise_news_item(n))
                except Exception:
                    pass
                return {"info": info, "fast_info": fast, "news": news}
            except Exception as exc:  # noqa: BLE001 — classify by message below
                last_exc = exc
                name = type(exc).__name__
                if "RateLimit" in name or "429" in str(exc) or "Too Many" in str(exc):
                    time.sleep(min(30.0, 3.0 * (2 ** attempt)) + random.uniform(0, 1))
                    continue
                break
        raise RuntimeError(
            f"yfinance fetch failed for {symbol} (last: {type(last_exc).__name__}: "
            f"{str(last_exc)[:120]}). Yahoo rate-limits aggressively — install "
            f"curl_cffi and ingest slowly into the frozen DB rather than fetching "
            f"live per run."
        )

    def _raw_to_snapshot(self, symbol: str, as_of: date, raw: dict) -> Snapshot:
        """Map raw yfinance fields -> Snapshot with CORRECT UNITS.

        Unit contract (the silent-error trap — tested in
        tests/test_stock_yf_mapping.py):
          * margins/growth: yfinance gives FRACTIONS (0.45) -> ×100 to pp.
          * debtToEquity:   yfinance gives a PERCENT-like number (150.0) -> ÷100
                            to a ratio (1.5).
          * dividend yield: prefer dividendRate/price×100 (unambiguous); else
                            the dividendYield field, which has flip-flopped
                            between fraction and percent across yfinance
                            versions — treat <=1 as fraction, >1 as already-pp.
          * ratios (PE etc.): plain. prices/target/SMAs: dollars.
        Provenance is stamped as_of (live path requires as_of==today, so this is
        look-ahead-safe); news is dropped if undated or dated after as_of.
        """
        info = raw.get("info", {}) or {}
        fast = raw.get("fast_info", {}) or {}

        def g(*keys):
            for k in keys:
                v = info.get(k)
                if v is None:
                    v = fast.get(k)
                if v is not None:
                    return v
            return None

        def num(v) -> Optional[float]:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if f == f else None  # drop NaN

        def pct(v) -> Optional[float]:
            x = num(v)
            return x * 100.0 if x is not None else None

        price = num(g("currentPrice", "last_price", "regularMarketPrice"))

        # dividend yield — compute from rate/price when possible (unambiguous)
        div_rate = num(info.get("dividendRate"))
        if div_rate is not None and price:
            div_yield = div_rate / price * 100.0
        else:
            raw_dy = num(info.get("dividendYield"))
            div_yield = None if raw_dy is None else (raw_dy * 100.0 if raw_dy <= 1.0 else raw_dy)

        de = num(info.get("debtToEquity"))
        debt_to_equity = de / 100.0 if de is not None else None

        fcf = num(info.get("freeCashflow"))
        rev = num(info.get("totalRevenue"))
        fcf_margin = (fcf / rev * 100.0) if (fcf is not None and rev) else None

        fields: dict = {
            "price": price,
            "market_cap": num(g("marketCap", "market_cap")),
            "pe": num(info.get("trailingPE")),
            "fwd_pe": num(info.get("forwardPE")),
            "peg": num(info.get("trailingPegRatio") or info.get("pegRatio")),
            "ps": num(info.get("priceToSalesTrailing12Months")),
            "ev_ebitda": num(info.get("enterpriseToEbitda")),
            "debt_to_equity": debt_to_equity,
            "eps": num(info.get("trailingEps")),
            "rev_growth_yoy": pct(info.get("revenueGrowth")),
            "eps_growth_yoy": pct(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")),
            "gross_margin": pct(info.get("grossMargins")),
            "operating_margin": pct(info.get("operatingMargins")),
            "net_margin": pct(info.get("profitMargins")),
            "fcf_margin": fcf_margin,
            "div_yield": div_yield,
            "week52_high": num(g("fiftyTwoWeekHigh", "year_high")),
            "week52_low": num(g("fiftyTwoWeekLow", "year_low")),
            "sma50": num(g("fiftyDayAverage", "fifty_day_average")),
            "sma200": num(g("twoHundredDayAverage", "two_hundred_day_average")),
            "analyst_target": num(info.get("targetMeanPrice")),
        }
        present = {k: v for k, v in fields.items() if v is not None}

        news: list[NewsItem] = []
        for n in raw.get("news", []):
            d = n.get("published")
            if not isinstance(d, date) or d > as_of:   # drop undated/future
                continue
            news.append(NewsItem(title=str(n.get("title", ""))[:200], published=d,
                                 url=n.get("url", ""), source=n.get("source", "")))

        provenance = {k: as_of for k in present}
        return Snapshot(ticker=symbol.upper(), as_of=as_of, news=news,
                        provenance=provenance, **present)

    # ---- cache (real) ---------------------------------------------------
    def _cache_path(self, symbol: str, as_of: date) -> Path:
        return self.cache_dir / symbol.upper() / f"{as_of.isoformat()}.json"

    def _read_cache(self, symbol: str, as_of: date) -> Optional[Snapshot]:
        p = self._cache_path(symbol, as_of)
        if not p.exists():
            return None
        try:
            return snapshot_from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return None

    def _write_cache(self, snap: Snapshot) -> None:
        p = self._cache_path(snap.ticker, snap.as_of)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snapshot_to_dict(snap), indent=2), encoding="utf-8")


def snapshot_to_dict(snap: Snapshot) -> dict:
    """Serialise a Snapshot to a JSON-able dict (dates as ISO strings)."""
    out: dict = {"ticker": snap.ticker, "as_of": snap.as_of.isoformat()}
    for m in _METRIC_FIELDS:
        v = getattr(snap, m, None)
        if v is not None:
            out[m] = v
    out["news"] = [
        {"title": n.title, "published": n.published.isoformat(),
         "url": n.url, "source": n.source}
        for n in snap.news
    ]
    out["provenance"] = {k: v.isoformat() for k, v in snap.provenance.items()}
    return out


# ---------------------------------------------------------------------------
# Live-fetch helpers (used only by YFinanceProvider; import-light)
# ---------------------------------------------------------------------------

def _make_session():
    """Browser-impersonating session to dodge Yahoo's UA-based rate limiting.

    Prefers curl_cffi (chrome impersonation — the de-facto yfinance workaround);
    falls back to a requests session with a desktop UA; None if neither is
    installed (yfinance then uses its default, which Stage 0 showed gets 429'd).
    """
    try:
        from curl_cffi import requests as _creq  # type: ignore
        return _creq.Session(impersonate="chrome")
    except Exception:
        pass
    try:
        import requests  # type: ignore
        s = requests.Session()
        s.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"),
        })
        return s
    except Exception:
        return None


def _normalise_news_item(n: dict) -> dict:
    """Normalise a yfinance news item across schema versions.

    Old schema: flat {title, providerPublishTime (epoch), link, publisher}.
    New schema (0.2.5x): nested under 'content' {title, pubDate (ISO),
    canonicalUrl{url}, provider{displayName}}. Returns
    {title, published: date|None, url, source}; published=None if it can't be
    dated (such items are dropped downstream — undated == not point-in-time safe).
    """
    c = n.get("content") if isinstance(n.get("content"), dict) else None
    title = (c or {}).get("title") or n.get("title") or ""
    url = ""
    src = ""
    if c is not None:
        cu = c.get("canonicalUrl")
        url = cu.get("url", "") if isinstance(cu, dict) else ""
        pv = c.get("provider")
        src = pv.get("displayName", "") if isinstance(pv, dict) else ""
    else:
        url = n.get("link", "")
        src = n.get("publisher", "")

    published: Optional[date] = None
    epoch = n.get("providerPublishTime")
    iso = (c or {}).get("pubDate") or (c or {}).get("displayTime")
    if epoch is not None:
        try:
            published = datetime.utcfromtimestamp(int(epoch)).date()
        except (TypeError, ValueError, OSError):
            published = None
    elif iso:
        try:
            published = datetime.strptime(str(iso)[:10], "%Y-%m-%d").date()
        except ValueError:
            published = None
    return {"title": title, "published": published, "url": url, "source": src}
