"""Numeric-claim verification for the Stock Swarm (the D4 fix).

This is the correctness-critical core that makes `verification_score` meaningful
for stock claims. In the debate run (`outputs/anothergroq.txt`) verification was
~0 because the validator did soft text-matching against Wikipedia. Here we
verify a claim's *number* against ground-truth market data.

Pipeline:
    text claim  -> extract_numeric_claims()  -> [(metric, value, kind)]
                -> resolve against a Snapshot's ground-truth value
                -> closeness()  -> strength in [0, 1]

`verify_claim()` returns a continuous strength (not {0, 0.5, 1.0}) so the
projection's `verification_score` reflects *how close* the swarm's number was.

Design notes for future edits:
  * Units are normalised per metric CLASS (ratio / percent / price / money) so
    "22%" is compared in percentage points and "$1.2B" in dollars. Get this
    wrong and verification silently passes/fails — there are tests in
    tests/test_stock_verify.py; run them after any change here.
  * No network, no LLM, no heavy imports — pure, fast, deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Metric vocabulary
# ---------------------------------------------------------------------------
# Canonical metric keys. Snapshot stores values in the unit noted per CLASS.
#   ratio   : plain multiple (P/E = 34.0)
#   percent : percentage points (rev growth 22% -> 22.0; gross margin 45% -> 45.0)
#   price   : dollars per share (price 450.0)
#   money   : absolute dollars (market cap 1.2e12)
METRIC_CLASS: dict[str, str] = {
    "pe": "ratio",
    "fwd_pe": "ratio",
    "peg": "ratio",
    "ps": "ratio",
    "ev_ebitda": "ratio",
    "debt_to_equity": "ratio",
    "rev_growth_yoy": "percent",
    "eps_growth_yoy": "percent",
    "gross_margin": "percent",
    "operating_margin": "percent",
    "net_margin": "percent",
    "fcf_margin": "percent",
    "div_yield": "percent",
    "price": "price",
    "sma50": "price",
    "sma200": "price",
    "week52_high": "price",
    "week52_low": "price",
    "analyst_target": "price",
    "eps": "price",  # dollars per share
    "market_cap": "money",
}

# Ordered (most specific first) alias patterns. The FIRST alias found in the
# text whose metric is present in the snapshot wins. Order matters: "forward
# p/e" must beat "p/e", "fcf margin" must beat "margin".
_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"forward\s+p/?e|fwd\.?\s*p/?e|forward\s+earnings\s+multiple|forward\s+earnings"), "fwd_pe"),
    (re.compile(r"\bpeg\b|price/?earnings[- ]to[- ]growth"), "peg"),
    (re.compile(r"ev/?\s*ebitda|enterprise\s+value\s*/?\s*ebitda"), "ev_ebitda"),
    (re.compile(r"price[- ]to[- ]sales|p/?s\s+ratio|\bp/?s\b"), "ps"),
    (re.compile(r"(?:trailing\s+)?p/?e(?:\s+ratio)?\b|price[- ]to[- ]earnings|times\s+earnings|earnings\s+multiple"), "pe"),
    (re.compile(r"debt[- ]to[- ]equity|debt/?equity|\bd/?e\s+ratio\b|\bd/?e\b"), "debt_to_equity"),
    (re.compile(r"revenue\s+(?:grew|growth|increased|rose|up|expansion)|sales\s+(?:grew|growth|increased|rose)|rev(?:enue)?\s+growth|top[- ]line\s+grow"), "rev_growth_yoy"),
    (re.compile(r"eps\s+(?:grew|growth|increased|rose)|earnings\s+per\s+share\s+grow|eps\s+growth"), "eps_growth_yoy"),
    (re.compile(r"gross\s+margin"), "gross_margin"),
    (re.compile(r"operating\s+margin|op\.?\s+margin|ebit\s+margin"), "operating_margin"),
    (re.compile(r"fcf\s+margin|free\s+cash\s+flow\s+margin"), "fcf_margin"),
    (re.compile(r"net\s+margin|profit\s+margin|net\s+profit\s+margin"), "net_margin"),
    (re.compile(r"dividend\s+yield|div\.?\s+yield"), "div_yield"),
    (re.compile(r"analyst\s+(?:price\s+)?target|price\s+target|mean\s+target|consensus\s+target"), "analyst_target"),
    (re.compile(r"market\s+cap(?:italization)?|mkt\s+cap"), "market_cap"),
    (re.compile(r"200[- ]day(?:\s+moving\s+average)?|sma\s*200|200\s*dma"), "sma200"),
    (re.compile(r"50[- ]day(?:\s+moving\s+average)?|sma\s*50|50\s*dma"), "sma50"),
    (re.compile(r"52[- ]week\s+high|52w\s+high"), "week52_high"),
    (re.compile(r"52[- ]week\s+low|52w\s+low"), "week52_low"),
    (re.compile(r"\beps\b|earnings\s+per\s+share"), "eps"),
    (re.compile(r"share\s+price|stock\s+price|\bprice\b|trades?\s+at|trading\s+at"), "price"),
]

# Tolerances per metric class: (err_full, err_zero).
#   strength = 1.0           when err <= err_full
#   strength = 0.0           when err >= err_zero
#   linear in between.
# For ratio/price/money err is RELATIVE (|c-a|/|a|); for percent err is
# ABSOLUTE in percentage points (|c-a|), because relative error on a small
# percent (e.g. 1% vs 2%) is misleadingly huge.
_TOLERANCE: dict[str, tuple[float, float]] = {
    "ratio":   (0.03, 0.30),   # within 3% rel -> 1.0 ; 0 by 30% rel
    "price":   (0.01, 0.15),   # within 1% rel -> 1.0 ; 0 by 15% rel
    "money":   (0.05, 0.40),   # within 5% rel -> 1.0 ; 0 by 40% rel
    "percent": (1.0, 10.0),    # within 1pp   -> 1.0 ; 0 by 10pp
}

# When a numeric claim cannot be resolved to a snapshot metric, the validator
# deposits a neutral VERIFICATION (mirrors coding TestValidator's 0.5 "could
# not construct" outcome) rather than penalising the claim.
UNRESOLVED_STRENGTH = 0.5


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericToken:
    raw: str
    value: float          # normalised to the token's own unit family
    kind: str             # 'percent' | 'multiple' | 'money' | 'plain'
    start: int            # char offset in the (lowercased) text
    end: int


@dataclass(frozen=True)
class NumericClaim:
    metric: str
    value: float          # normalised into the metric's snapshot unit
    kind: str
    raw: str


_MONEY_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

# A number with optional leading $, thousands commas, decimal, and an optional
# trailing %, x, or magnitude suffix (k/m/b/t). Captured groups:
#   1: optional $    2: the numeric core    3: optional %/x/suffix
_NUM_RE = re.compile(
    r"(\$)?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)\s*"
    r"(%|x|×|[kmbt]\b)?",
    re.IGNORECASE,
)


def _tokenize_numbers(text: str) -> list[NumericToken]:
    """Find all numeric tokens with their unit kind and char span."""
    out: list[NumericToken] = []
    for m in _NUM_RE.finditer(text):
        dollar, core, suffix = m.group(1), m.group(2), m.group(3)
        try:
            val = float(core.replace(",", ""))
        except ValueError:
            continue
        suffix = (suffix or "").lower().replace("×", "x")
        if suffix == "%":
            kind = "percent"
        elif suffix == "x":
            kind = "multiple"
        elif suffix in _MONEY_SUFFIX:
            val *= _MONEY_SUFFIX[suffix]
            kind = "money"
        elif dollar:
            kind = "money"
        else:
            kind = "plain"
        out.append(NumericToken(raw=m.group(0).strip(), value=val, kind=kind,
                                start=m.start(2), end=m.end()))
    return out


def _normalise_for_metric(token: NumericToken, metric: str) -> float:
    """Convert a token's value into the unit the snapshot stores for `metric`.

    The tricky cases:
      * debt_to_equity is a ratio but is often quoted as a percent ("50%")
        when the snapshot stores 0.5 — divide by 100.
      * percent-class metrics quoted bare ("margin of 45") are already in pp.
    """
    cls = METRIC_CLASS.get(metric, "ratio")
    v = token.value
    if metric == "debt_to_equity" and token.kind == "percent":
        return v / 100.0
    if cls == "percent":
        # token may be "45%" (percent kind, already pp) or bare "45" — both pp.
        return v
    # ratio / price / money: percent/multiple/plain/money all map to the number
    # as-is (money suffix already applied in tokenizer).
    return v


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    """Extract (metric, value) claims from free-text.

    For each metric alias found, associate the numeric token nearest to the
    alias match. Returns claims in alias-priority order (most specific metric
    first), de-duplicated by metric.
    """
    if not text:
        return []
    low = text.lower()
    tokens = _tokenize_numbers(low)
    if not tokens:
        return []

    claims: list[NumericClaim] = []
    seen: set[str] = set()
    for pat, metric in _ALIASES:
        if metric in seen:
            continue
        m = pat.search(low)
        if not m:
            continue
        anchor = (m.start() + m.end()) // 2
        # nearest numeric token by char distance to the alias span
        nearest = min(
            tokens,
            key=lambda t: min(abs(t.start - anchor), abs(t.end - anchor)),
        )
        # Guard: don't associate a wildly distant number (> 60 chars away)
        dist = min(abs(nearest.start - anchor), abs(nearest.end - anchor))
        if dist > 60:
            continue
        claims.append(NumericClaim(
            metric=metric,
            value=_normalise_for_metric(nearest, metric),
            kind=nearest.kind,
            raw=nearest.raw,
        ))
        seen.add(metric)
    return claims


# ---------------------------------------------------------------------------
# Closeness / strength
# ---------------------------------------------------------------------------

def closeness(claimed: float, actual: float, metric: str) -> float:
    """Strength in [0, 1] for how well `claimed` matches `actual`.

    Relative error for ratio/price/money classes; absolute (percentage-point)
    error for the percent class.
    """
    cls = METRIC_CLASS.get(metric, "ratio")
    err_full, err_zero = _TOLERANCE[cls]
    if cls == "percent":
        err = abs(claimed - actual)
    else:
        denom = abs(actual)
        if denom < 1e-9:
            # actual is ~0: fall back to absolute error vs the full-tol band.
            err = abs(claimed - actual)
            err_full, err_zero = 0.0, max(err_zero, 1e-6)
        else:
            err = abs(claimed - actual) / denom
    if err <= err_full:
        return 1.0
    if err >= err_zero:
        return 0.0
    return round(1.0 - (err - err_full) / (err_zero - err_full), 4)


# ---------------------------------------------------------------------------
# Top-level verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationResult:
    metric: Optional[str]
    claimed: Optional[float]
    actual: Optional[float]
    strength: float
    note: str

    def as_atom(self) -> dict:
        """Shape stored in VERIFICATION.metadata['atoms'] for the genome/atom
        pipeline (projection._build_atoms reads this)."""
        return {
            "metric": self.metric,
            "claimed": self.claimed,
            "actual": self.actual,
            "strength": self.strength,
            "text": self.note,
        }


def verify_claim(text: str, snapshot) -> VerificationResult:
    """Verify the primary numeric claim in `text` against `snapshot`.

    `snapshot` is any object exposing `get(metric) -> Optional[float]`
    (core.stock_data.Snapshot does). Picks the first extracted claim whose
    metric the snapshot can resolve; deposits UNRESOLVED_STRENGTH when none
    resolve (e.g. a purely qualitative claim).
    """
    claims = extract_numeric_claims(text)
    if not claims:
        return VerificationResult(None, None, None, UNRESOLVED_STRENGTH,
                                  "no numeric claim found")
    for c in claims:
        actual = None
        try:
            actual = snapshot.get(c.metric)
        except Exception:
            actual = None
        if actual is None:
            continue
        s = closeness(c.value, float(actual), c.metric)
        note = (f"{c.metric}: claimed {c.value:g} vs actual {float(actual):g} "
                f"-> strength {s:.2f}")
        return VerificationResult(c.metric, c.value, float(actual), s, note)
    # Claims existed but none resolved against the snapshot.
    return VerificationResult(claims[0].metric, claims[0].value, None,
                              UNRESOLVED_STRENGTH,
                              f"{claims[0].metric} not in snapshot")
