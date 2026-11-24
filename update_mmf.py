"""Add one week of ICI money market fund data and refresh the dashboard files.

Usage: python update_mmf.py WEEK_END TOTAL_BN CHANGE_BN
  e.g. python update_mmf.py 2026-11-25 7990.12 13.68

ICI blocks GitHub's servers, so the daily scheduled Claude task reads ICI's
weekly release (https://www.ici.org/research/stats/mmf) and calls this.
"""
import json
import sys
from pathlib import Path

DASH = Path(__file__).parent / "dash"
HIST = DASH / "mmf_history.json"


def main(week_end: str, total_bn: float, change_bn: float) -> None:
    hist = json.loads(HIST.read_text()) if HIST.exists() else []
    hist = [h for h in hist if h["week_end"] != week_end]
    hist.append({"week_end": week_end, "total_bn": round(total_bn, 1), "change_bn": round(change_bn, 2)})
    hist.sort(key=lambda h: h["week_end"])
    hist = hist[-104:]
    HIST.write_text(json.dumps(hist, indent=1))
    (DASH / "mmf.json").write_text(json.dumps(hist, indent=0))

    summ_path = DASH / "summary.json"
    summ = json.loads(summ_path.read_text())
    last = hist[-1]
    known = [h for h in hist if h.get("total_bn") is not None]
    summ[0].update(
        mmf_week_end=last["week_end"],
        mmf_total_bn=last["total_bn"],
        mmf_change_1w_bn=last["change_bn"],
        mmf_change_4w_bn=round(known[-1]["total_bn"] - known[-5]["total_bn"], 1) if len(known) >= 5 else None,
        mmf_weeks=len(hist),
    )
    summ_path.write_text(json.dumps(summ, indent=0))
    print(f"Money market funds: week ended {week_end}, ${total_bn:,.1f}B ({change_bn:+.2f}B)")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
