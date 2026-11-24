"""Sector Dip Barometer.

Pulls daily history for the 11 SPDR sector ETFs plus SPY, scores how unusual
each sector's current dip is against its own history, and writes
docs/data.json for the dashboard.

Usage:
    python barometer.py            # live data from Yahoo Finance
    python barometer.py --sample   # synthetic data, for testing the dashboard
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = {
    "XLK": "Technology",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLB": "Materials",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}
BENCHMARK = "SPY"
# Where money goes when it leaves stocks
SAFE = {   # ticker: (name, group)
    "SPY": ("US large caps (S&P 500)", "Stocks"),
    "IWM": ("US small caps (Russell 2000)", "Stocks"),
    "VEA": ("Developed markets (Europe, Japan)", "Stocks"),
    "EEM": ("Emerging markets (China, India...)", "Stocks"),
    "BIL": ("Cash (T-bills, money markets)", "Safe"),
    "IEF": ("Treasuries (7-10 yr)", "Safe"),
    "TLT": ("Treasuries (20+ yr)", "Safe"),
    "GLD": ("Gold", "Safe"),
    "UUP": ("US dollar", "Safe"),
    "HYG": ("Junk bonds", "Other"),
    "DBC": ("Commodities (oil, metals, crops)", "Other"),
    "IBIT": ("Bitcoin", "Other"),
}
OUT = Path(__file__).parent / "docs" / "data.json"

# Label thresholds (drawdown from 52-week high, as a fraction)
PULLBACK = -0.05
CORRECTION = -0.10


# ---------------------------------------------------------------- data

def fetch_live() -> tuple[pd.DataFrame, dict]:
    import yfinance as yf

    tickers = list(dict.fromkeys(list(SECTORS) + [BENCHMARK] + list(SAFE)))
    closes = None
    for attempt in range(3):
        df = yf.download(tickers, period="max", auto_adjust=True,
                         progress=False, threads=False)
        closes = df["Close"] if "Close" in df else None
        if closes is not None and closes.dropna(how="all").shape[0] > 250:
            missing = [t for t in tickers if closes[t].dropna().empty]
            if not missing:
                break
        time.sleep(10 * (attempt + 1))
    if closes is None:
        raise RuntimeError("Could not download price data")

    pe = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            pe[t] = info.get("trailingPE")
        except Exception:
            pe[t] = None
    return closes, pe


def fetch_sample() -> tuple[pd.DataFrame, dict]:
    """Random-walk prices with a few built-in dips, so every label shows up."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * 12)
    n = len(idx)
    market = rng.normal(0.0004, 0.009, n)
    market[-60:] = rng.normal(0.0008, 0.006, 60)   # market near highs
    cols = {}
    # last-90-day shocks per ticker to exercise each label
    shocks = {"XLK": -0.0022, "XLE": -0.0018, "XLRE": -0.0009,
              "XLY": -0.0006}
    for t in dict.fromkeys(list(SECTORS) + [BENCHMARK] + list(SAFE)):
        beta = 1.0 if t == BENCHMARK else rng.uniform(0.6, 1.3)
        safe_beta = {"BIL": 0.0, "IEF": -0.1, "TLT": -0.25, "GLD": 0.05, "UUP": -0.1,
                     "DBC": 0.3, "IBIT": 1.8}
        if t in safe_beta:
            beta = safe_beta[t]
        noise = {"BIL": 0.0002, "IEF": 0.003, "TLT": 0.008, "GLD": 0.008, "UUP": 0.004,
                 "DBC": 0.01, "IBIT": 0.03}.get(t, 0.004)
        r = beta * market + rng.normal(0, noise, n)
        if t == "BIL":
            r += 0.00016
        # slow sector-specific trends so rotation has something to show
        r += 0.0006 * np.sin(np.arange(n) / rng.uniform(60, 160) + rng.uniform(0, 6))
        s = shocks.get(t, 0.0)
        if s:
            r[-90:] += s
            if t == "XLK":            # sharp drop, then a bounce: "On Sale"
                r[-90:-15] += s
                r[-15:] += 0.005
        cols[t] = 100 * np.exp(np.cumsum(r))
    closes = pd.DataFrame(cols, index=idx)
    closes.loc[: idx[252 * 5], "XLC"] = np.nan   # shorter history, like real XLC
    pe = {t: round(float(rng.uniform(12, 32)), 1) for t in closes}
    return closes, pe


# ---------------------------------------------------------------- metrics

def rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def analyze(s: pd.Series, spy: pd.Series) -> dict:
    s = s.dropna()
    price = float(s.iloc[-1])

    hi52 = s.rolling(252, min_periods=20).max()
    dd52_series = s / hi52 - 1
    dd52 = float(dd52_series.iloc[-1])
    ath = float(s.max())
    dd_ath = price / ath - 1

    # How rare is today's drawdown? % of past days with a shallower drawdown.
    hist = dd52_series.dropna().iloc[:-1]
    rarity = float((hist > dd52).mean() * 100) if len(hist) else 0.0

    # Speed: how fast it got here
    peak_date = s.iloc[-252:].idxmax()
    days_since_peak = int((s.index[-1] - peak_date).days)
    chg_1m = float(s.iloc[-1] / s.iloc[-22] - 1)
    chg_3m = float(s.iloc[-1] / s.iloc[-64] - 1)

    # Trend
    ma200 = s.rolling(200).mean()
    vs_200 = float(price / ma200.iloc[-1] - 1)
    ma200_slope = float(ma200.iloc[-1] / ma200.iloc[-21] - 1)
    low20 = float(s.iloc[-20:].min())
    off_low = price / low20 - 1
    new_low_recent = bool(s.iloc[-5:].min() <= low20 * 1.0001)
    r = rsi(s)

    # Is it the sector or the whole market? Sector/SPY ratio drawdown.
    spy_a = spy.reindex(s.index).ffill()
    ratio = s / spy_a
    rel_dd = float(ratio.iloc[-1] / ratio.rolling(252, min_periods=20).max().iloc[-1] - 1)
    rel_3m = float(chg_3m - (spy_a.iloc[-1] / spy_a.iloc[-64] - 1))

    # ---- label
    if dd52 > PULLBACK:
        label = "Normal"
        why = "Within 5% of its 52-week high."
    elif dd52 > CORRECTION:
        label = "Pullback"
        why = f"Down {abs(dd52):.0%} from its high, an ordinary dip."
    else:
        knife = (vs_200 < 0 and ma200_slope < 0 and new_low_recent and r < 45)
        if knife:
            label = "Falling Knife"
            why = ("Still making new lows below a falling 200-day average. "
                   "Wait for selling to slow.")
        else:
            label = "On Sale"
            why = (f"Down {abs(dd52):.0%}, deeper than {rarity:.0f}% of its history, "
                   f"and {off_low:.0%} off its recent low. Selling has slowed.")

    if label != "Normal":
        if rel_dd < -0.05:
            why += " Weaker than the market, so the dip is sector-specific."
        elif rel_3m > -0.02:
            why += " Moving with the market, so this is mostly a broad dip."

    spark = s.iloc[-252:]
    spark = spark.iloc[::3]   # ~84 points

    return {
        "price": round(price, 2),
        "dd52": round(dd52, 4),
        "dd_ath": round(dd_ath, 4),
        "rarity": round(rarity, 1),
        "days_since_peak": days_since_peak,
        "chg_1m": round(chg_1m, 4),
        "chg_3m": round(chg_3m, 4),
        "vs_200": round(vs_200, 4),
        "rsi": round(r, 1),
        "rel_dd": round(rel_dd, 4),
        "rel_3m": round(rel_3m, 4),
        "history_start": s.index[0].strftime("%Y-%m-%d"),
        "label": label,
        "why": why,
        "spark": [round(float(v), 2) for v in spark.values],
    }


def rotation(s: pd.Series, spy: pd.Series, weeks: int = 12) -> dict:
    """How the sector did vs SPY: last 4 weeks, the 4 weeks before, and week by week."""
    wk = s.resample("W-FRI").last().dropna()
    sw = spy.resample("W-FRI").last().reindex(wk.index).ffill()
    rel_w = (wk.pct_change() - sw.pct_change()).dropna().iloc[-weeks:]

    def rel(n0, n1):  # sector return minus SPY return between n0 and n1 weeks ago
        return float((wk.iloc[-1 - n1] / wk.iloc[-1 - n0] - 1) - (sw.iloc[-1 - n1] / sw.iloc[-1 - n0] - 1))

    now, before = rel(4, 0), rel(8, 4)
    if now > 0.005 and now > before:
        flow = "in"
    elif now < -0.005 and now < before:
        flow = "out"
    else:
        flow = "flat"
    return {
        "rel_4w": round(now, 4),
        "rel_prev_4w": round(before, 4),
        "flow": flow,
        "weekly": [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in rel_w.items()],
    }


def safe_havens(closes: pd.DataFrame) -> dict:
    """4-week returns for stocks vs bonds, cash, gold, and a plain verdict."""
    def ret(t, n0, n1):
        x = closes[t].dropna()
        return float(x.iloc[-1 - n1] / x.iloc[-1 - n0] - 1)

    rows = []
    for t, (name, group) in SAFE.items():
        if t not in closes or closes[t].dropna().shape[0] < 45:
            continue
        rows.append({"ticker": t, "name": name, "group": group,
                     "ret_4w": round(ret(t, 20, 0), 4),
                     "ret_prev_4w": round(ret(t, 40, 20), 4)})
    r = {x["ticker"]: x["ret_4w"] for x in rows}
    stocks, cash = r.get("SPY", 0), r.get("BIL", 0)
    bonds = max(r.get("IEF", -1), r.get("TLT", -1))
    gold = r.get("GLD", -1)
    credit = r.get("HYG", 0) - r.get("IEF", 0)   # junk beating Treasuries = appetite for risk

    if stocks > cash and stocks > bonds:
        verdict, mood = "Risk-on: money is choosing stocks over bonds and cash.", "on"
    elif stocks < cash and max(bonds, gold, cash) > stocks:
        best = max([("cash", cash), ("bonds", bonds), ("gold", gold)], key=lambda z: z[1])[0]
        verdict, mood = f"Risk-off: stocks are trailing cash. Safe money is favoring {best}.", "off"
    else:
        verdict, mood = "Mixed: stocks are up, but bonds or gold are doing better.", "mixed"
    if credit < -0.01:
        verdict += " Junk bonds are lagging Treasuries, an early stress sign."
    elif credit > 0.01:
        verdict += " Junk bonds are beating Treasuries, so credit markets are calm."
    return {"assets": rows, "verdict": verdict, "mood": mood, "credit_gap": round(credit, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    closes, pe = fetch_sample() if args.sample else fetch_live()
    spy = closes[BENCHMARK].dropna()

    rows = []
    for t, name in SECTORS.items():
        m = analyze(closes[t], spy)
        m.update(ticker=t, name=name, pe=pe.get(t))
        m["rotation"] = rotation(closes[t].dropna(), spy)
        rows.append(m)
    bench = analyze(spy, spy)
    bench.update(ticker=BENCHMARK, name="S&P 500", pe=pe.get(BENCHMARK))

    flows_in = sorted([r for r in rows if r["rotation"]["flow"] == "in"],
                      key=lambda r: -r["rotation"]["rel_4w"])
    flows_out = sorted([r for r in rows if r["rotation"]["flow"] == "out"],
                       key=lambda r: r["rotation"]["rel_4w"])
    rows.sort(key=lambda r: r["rarity"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    out = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "as_of": closes.index[-1].strftime("%Y-%m-%d"),
        "sample": bool(args.sample),
        "safe_havens": safe_havens(closes),
        "benchmark": bench,
        "sectors": rows,
        "rotating_in": [r["ticker"] for r in flows_in],
        "rotating_out": [r["ticker"] for r in flows_out],
        "counts": {lab: sum(r["label"] == lab for r in rows)
                   for lab in ["On Sale", "Falling Knife", "Pullback", "Normal"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"Wrote {OUT} ({out['as_of']})")
    for r in rows:
        print(f"{r['rank']:>2} {r['ticker']:<5} {r['label']:<14} "
              f"dd52 {r['dd52']:+.1%}  rarity {r['rarity']:.0f}%  "
              f"4w vs SPY {r['rotation']['rel_4w']:+.1%} (before {r['rotation']['rel_prev_4w']:+.1%}) {r['rotation']['flow']}")
    print("Rotating in: ", out["rotating_in"])
    print("Rotating out:", out["rotating_out"])


if __name__ == "__main__":
    main()
