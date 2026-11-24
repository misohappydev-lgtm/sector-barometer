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


PERIODS = {"1d": 1, "1w": 5, "4w": 20}          # trading days
FLOW_MIN = {"1d": 0.002, "1w": 0.004, "4w": 0.005}  # gap vs SPY needed to call a flow
PERIOD_WORDS = {"1d": "today", "1w": "this week", "4w": "over the last 4 weeks"}


def _ret(x: pd.Series, n0: int, n1: int) -> float:
    """Return from n0 trading days ago to n1 trading days ago."""
    x = x.dropna()
    return float(x.iloc[-1 - n1] / x.iloc[-1 - n0] - 1)


def rotation(s: pd.Series, spy: pd.Series, weeks: int = 12) -> dict:
    """Sector vs SPY over 1 day, 1 week, 4 weeks (and the same span before), plus weekly history."""
    s = s.dropna()
    spy_a = spy.reindex(s.index).ffill()
    out = {}
    for key, n in PERIODS.items():
        now = _ret(s, n, 0) - _ret(spy_a, n, 0)
        before = _ret(s, 2 * n, n) - _ret(spy_a, 2 * n, n)
        thr = FLOW_MIN[key]
        flow = "in" if (now > thr and now > before) else "out" if (now < -thr and now < before) else "flat"
        out[f"ret_{key}"] = round(_ret(s, n, 0), 4)
        out[f"rel_{key}"] = round(now, 4)
        out[f"rel_prev_{key}"] = round(before, 4)
        out[f"flow_{key}"] = flow
    out["flow"] = out["flow_4w"]

    # 12 back-to-back blocks per period (12 days, 12 weeks, 12 four-week spans), newest last
    out["blocks"] = {}
    idx = s.index
    for key, n in PERIODS.items():
        rows = []
        for k in range(weeks):
            e, b = len(idx) - 1 - k * n, len(idx) - 1 - (k + 1) * n
            if b < 0:
                break
            rel = (s.iloc[e] / s.iloc[b] - 1) - (spy_a.iloc[e] / spy_a.iloc[b] - 1)
            rows.append([idx[e].strftime("%Y-%m-%d"), round(float(rel), 4)])
        out["blocks"][key] = rows[::-1]
    out["weekly"] = out["blocks"]["1w"]
    return out


def _verdict(r: dict, period: str) -> tuple[str, str]:
    words = PERIOD_WORDS[period]
    stocks, cash = r.get("SPY", 0), r.get("BIL", 0)
    bonds = max(r.get("IEF", -1), r.get("TLT", -1))
    gold = r.get("GLD", -1)
    credit = r.get("HYG", 0) - r.get("IEF", 0)
    if stocks > cash and stocks > bonds:
        v, mood = f"Risk-on {words}: money is choosing stocks over bonds and cash.", "on"
    elif stocks < cash and max(bonds, gold, cash) > stocks:
        best = max([("cash", cash), ("bonds", bonds), ("gold", gold)], key=lambda z: z[1])[0]
        v, mood = f"Risk-off {words}: stocks are trailing cash, and safe money is favoring {best}.", "off"
    else:
        v, mood = f"Mixed {words}: stocks are up, but bonds or gold are doing better.", "mixed"
    lim = {"1d": 0.003, "1w": 0.006, "4w": 0.01}[period]
    if credit < -lim:
        v += " Junk bonds are lagging Treasuries, an early stress sign."
    elif credit > lim:
        v += " Junk bonds are beating Treasuries, so credit markets are calm."
    return v, mood


def safe_havens(closes: pd.DataFrame) -> dict:
    """Returns over 1d / 1w / 4w for stocks, bonds, cash, gold, etc., with a verdict per period."""
    rows = []
    for t, (name, group) in SAFE.items():
        if t not in closes or closes[t].dropna().shape[0] < 45:
            continue
        row = {"ticker": t, "name": name, "group": group}
        for key, n in PERIODS.items():
            row[f"ret_{key}"] = round(_ret(closes[t], n, 0), 4)
            row[f"ret_prev_{key}"] = round(_ret(closes[t], 2 * n, n), 4)
        rows.append(row)
    out = {"assets": rows}
    for key in PERIODS:
        v, mood = _verdict({x["ticker"]: x[f"ret_{key}"] for x in rows}, key)
        out[f"verdict_{key}"], out[f"mood_{key}"] = v, mood
    out["verdict"], out["mood"] = out["verdict_4w"], out["mood_4w"]
    return out


# ---------------------------------------------------------------- money market funds (ICI, weekly)

MMF_HISTORY = Path(__file__).parent / "dash" / "mmf_history.json"
ICI_URL = "https://www.ici.org/research/stats/mmf"


def fetch_mmf() -> list[dict]:
    """Scrape ICI's latest weekly release and add it to a running history file.

    ICI publishes total money market fund assets every Thursday for the week
    ended Wednesday. The weekly change in assets is mostly new money flowing in
    or out, so it is the best free read on money moving to cash.
    """
    import re
    import urllib.request

    hist = json.loads(MMF_HISTORY.read_text()) if MMF_HISTORY.exists() else []
    try:
        req = urllib.request.Request(ICI_URL, headers={"User-Agent": "Mozilla/5.0 (sector-barometer)"})
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        m = re.search(r"Total money market fund assets\W*\d?\s*(increased|decreased) by \$([\d,.]+) (billion|million) "
                      r"to \$([\d,.]+) trillion for the week ended Wednesday, (\w+ \d{1,2})", text)
        if not m:
            print("ICI: release text not found; keeping history as is")
            return hist
        sign = 1 if m.group(1) == "increased" else -1
        chg = float(m.group(2).replace(",", "")) * (1 if m.group(3) == "billion" else 0.001) * sign
        total = float(m.group(4).replace(",", "")) * 1000
        today = datetime.now(timezone.utc).date()
        week_end = None
        for yr in (today.year, today.year - 1):
            try:
                d = datetime.strptime(f"{m.group(5)} {yr}", "%B %d %Y").date()
            except ValueError:
                continue
            if d <= today:
                week_end = d.isoformat()
                break
        if not week_end:
            return hist
        hist = [h for h in hist if h["week_end"] != week_end]
        hist.append({"week_end": week_end, "total_bn": round(total, 1), "change_bn": round(chg, 2)})
        hist.sort(key=lambda h: h["week_end"])
        hist = hist[-104:]
        MMF_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        MMF_HISTORY.write_text(json.dumps(hist, indent=1))
        print(f"ICI: week ended {week_end}: {chg:+.1f}B to ${total/1000:.2f}T")
    except Exception as e:   # never let this break the daily run
        print(f"ICI fetch failed: {e}")
    return hist


# ---------------------------------------------------------------- dashboard files

DASH = Path(__file__).parent / "dash"
SHORT = {"XLK": "Tech", "XLC": "Communication", "XLY": "Discretionary", "XLF": "Financials",
         "XLV": "Health Care", "XLI": "Industrials", "XLE": "Energy", "XLB": "Materials",
         "XLP": "Staples", "XLRE": "Real Estate", "XLU": "Utilities"}


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def write_dash(out: dict, mmf: list[dict]) -> None:
    DASH.mkdir(exist_ok=True)
    secs = out["sectors"]
    sh = out["safe_havens"]

    summary = {"as_of": out["as_of"], "sample": out["sample"],
               "spy_dd52": out["benchmark"]["dd52"], "spy_label": out["benchmark"]["label"],
               "on_sale": out["counts"]["On Sale"], "falling_knife": out["counts"]["Falling Knife"],
               "pullback": out["counts"]["Pullback"], "normal": out["counts"]["Normal"]}
    spy_rows = {a["ticker"]: a for a in sh["assets"]}
    for key in PERIODS:
        summary[f"spy_ret_{key}"] = spy_rows["SPY"][f"ret_{key}"] if "SPY" in spy_rows else None
    for key in PERIODS:
        ins = sorted([r for r in secs if r["rotation"][f"flow_{key}"] == "in"], key=lambda r: -r["rotation"][f"rel_{key}"])
        outs = sorted([r for r in secs if r["rotation"][f"flow_{key}"] == "out"], key=lambda r: r["rotation"][f"rel_{key}"])
        i = _join([SHORT[r["ticker"]] for r in ins[:3]])
        o = _join([SHORT[r["ticker"]] for r in outs[:3]])
        w = PERIOD_WORDS[key]
        if i and o:
            h = f"Money is moving into {i} and out of {o} {w}."
        elif i:
            h = f"Money is moving into {i} {w}. Nothing is clearly losing out."
        elif o:
            h = f"Money is moving out of {o} {w}. Nothing is clearly gaining."
        else:
            h = f"No clear rotation {w}. Sectors are moving roughly with the market."
        summary[f"rotation_{key}"] = h
        summary[f"verdict_{key}"] = sh[f"verdict_{key}"]
        summary[f"mood_{key}"] = sh[f"mood_{key}"]
    if mmf:
        last = mmf[-1]
        summary["mmf_week_end"] = last["week_end"]
        summary["mmf_total_bn"] = last["total_bn"]
        summary["mmf_change_1w_bn"] = last["change_bn"]
        known = [h for h in mmf if h.get("total_bn") is not None]
        summary["mmf_change_4w_bn"] = round(known[-1]["total_bn"] - known[-5]["total_bn"], 1) if len(known) >= 5 else None
        summary["mmf_weeks"] = len(mmf)

    sector_rows, week_rows = [], []
    for r in secs:
        row = {"ticker": r["ticker"], "name": SHORT[r["ticker"]], "label": r["label"], "why": r["why"],
               "dd52": r["dd52"], "rarity": r["rarity"], "since": r["history_start"][:4],
               "chg_1m": r["chg_1m"], "rank": r["rank"]}
        for key in PERIODS:
            row[f"ret_{key}"] = r["rotation"][f"ret_{key}"]
            row[f"rel_{key}"] = r["rotation"][f"rel_{key}"]
            row[f"rel_prev_{key}"] = r["rotation"][f"rel_prev_{key}"]
            row[f"flow_{key}"] = r["rotation"][f"flow_{key}"]
        sector_rows.append(row)
        for key, blocks in r["rotation"]["blocks"].items():
            for end, v in blocks:
                week_rows.append({"ticker": r["ticker"], "name": SHORT[r["ticker"]], "period": key, "end": end, "rel": v})

    def dump(name, rows):
        (DASH / name).write_text(json.dumps(rows, indent=0))
    dump("summary.json", [summary])
    dump("sectors.json", sector_rows)
    dump("sector_weeks.json", week_rows)
    dump("assets.json", sh["assets"])
    dump("mmf.json", mmf)
    print(f"Wrote dashboard files to {DASH}")


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

    mmf = [] if args.sample else fetch_mmf()
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
    write_dash(out, mmf)
    print(f"Wrote {OUT} ({out['as_of']})")
    for r in rows:
        print(f"{r['rank']:>2} {r['ticker']:<5} {r['label']:<14} "
              f"dd52 {r['dd52']:+.1%}  rarity {r['rarity']:.0f}%  "
              f"4w vs SPY {r['rotation']['rel_4w']:+.1%} (before {r['rotation']['rel_prev_4w']:+.1%}) {r['rotation']['flow']}")
    print("Rotating in: ", out["rotating_in"])
    print("Rotating out:", out["rotating_out"])


if __name__ == "__main__":
    main()
