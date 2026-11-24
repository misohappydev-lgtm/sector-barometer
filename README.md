# Sector Dip Barometer

A daily Dip Barometer for the 11 S&P 500 sectors (SPDR ETFs), plus a sector rotation chart.

- **Dip by sector**: how far each sector is from its 52-week high, how rare that drawdown is in its own history, and a label: Normal, Pullback, On Sale or Falling Knife.
- **Where the money is moving**: a one-line summary, a bar chart of each sector vs the S&P 500 over the last 4 weeks (with the 4 weeks before for comparison), and a 12-week grid showing the shift week by week.

## Setup

1. Create a repo (e.g. `sector-barometer`) and push these files.
2. Settings → Pages → Deploy from branch → `main`, folder `/docs`.
3. Actions → "Update sector barometer" → Run workflow, to replace the sample data with live data.

After that it runs every weekday after the close. No API keys needed (Yahoo Finance via `yfinance`).

## Local

```
pip install -r requirements.txt
python barometer.py            # live data
python barometer.py --sample   # fake data for testing the page
cd docs && python -m http.server
```

## Tuning

Thresholds live at the top of `barometer.py` (`PULLBACK`, `CORRECTION`). Falling Knife = 10%+ off the high, below a falling 200-day average, a new 20-day low within the last week, and RSI under 45.
