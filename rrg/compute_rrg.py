"""
Weekly sector rotation data for the Monday Rotation Report.

Downloads about two years of daily prices from Yahoo Finance for the 11 Select Sector SPDR ETFs
and SPY, converts them to weekly (Friday) closes, and computes an
approximation of the JdK RS-Ratio and RS-Momentum used in Relative Rotation
Graphs. The real RRG formula is proprietary; this is the commonly published
approximation, so positions will be close to, but not identical to,
StockCharts.

Outputs (in data/):
  rrg_latest.json  - last TAIL weeks per sector, plus quadrant, for the report
  rrg_history.csv  - every computed week, for later backtesting
  prices_weekly.csv - the weekly closes used
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK = "SPY"
SECTORS = {
    "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}
# Calibrated Nov 10, 2026 against a StockCharts RRG screenshot (week ending Nov 6):
# all 11 sectors within about 1-2.5 points; quadrants matched for 8 of 11, the
# misses were sectors sitting close to a quadrant line.
SHORT = 10    # weeks, short EMA of relative strength
LONG = 40     # weeks, long EMA of relative strength
MOM = 10      # weeks, EMA of RS-Ratio used for RS-Momentum
TAIL = 6      # points on the chart tail (latest week + 5 prior), matches StockCharts' 5-week tail
OUT = Path(__file__).resolve().parent.parent / "data"


# ---------- price download ----------
def from_yahoo(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                     progress=False, threads=False)
    closes = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    return closes.dropna(how="all")


def get_prices() -> tuple[pd.DataFrame, str]:
    tickers = list(SECTORS) + [BENCHMARK]
    errors = []
    for name, fn in (("Yahoo Finance", from_yahoo),):
        for attempt in range(4):
            try:
                px = fn(tickers)
                missing = [t for t in tickers if t not in px or px[t].dropna().empty]
                if missing:
                    raise ValueError(f"missing tickers {missing}")
                return px, name
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name} attempt {attempt + 1}: {e}")
                time.sleep(30 * (attempt + 1))
    sys.exit("Could not download prices:\n" + "\n".join(errors))


# ---------- calculation ----------
def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    weekly = daily.resample("W-FRI").last().dropna(how="all")
    # drop a partial week if the job runs mid-week
    last_trading_day = daily.index.max()
    if weekly.index[-1] > last_trading_day and last_trading_day.weekday() < 4:
        weekly = weekly.iloc[:-1]
    return weekly


def rrg(weekly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """RS-Ratio and RS-Momentum approximation, both centred on 100.

    RS-Ratio    = 100 * EMA(SHORT) of relative strength / EMA(LONG) of relative strength
    RS-Momentum = 100 * RS-Ratio / EMA(MOM) of RS-Ratio
    """
    out = {}
    bench = weekly[BENCHMARK]
    for t in SECTORS:
        rs = weekly[t] / bench
        ratio = 100 * rs.ewm(span=SHORT, adjust=False).mean() / rs.ewm(span=LONG, adjust=False).mean()
        mom = 100 * ratio / ratio.ewm(span=MOM, adjust=False).mean()
        df = pd.DataFrame({"rs_ratio": ratio, "rs_momentum": mom})
        out[t] = df.iloc[LONG:]  # skip the warm-up period
    return out


def quadrant(ratio: float, mom: float) -> str:
    if ratio >= 100:
        return "Leading" if mom >= 100 else "Weakening"
    return "Improving" if mom >= 100 else "Lagging"


ARROWS = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
CROSS_HORIZON = 8  # weeks; crossings projected further out than this are reported as "none soon"


def direction(tail: pd.DataFrame) -> dict:
    """Heading, speed, distance and projected next quadrant crossing.

    Heading uses the average of the last two weekly steps to damp noise.
    Speed is the length of the last weekly step. Distance is from the 100/100 centre.
    The crossing is a straight-line projection at the current heading and pace,
    so it is only a rough guide: rotations curve.
    """
    x, y = float(tail.rs_ratio.iloc[-1]), float(tail.rs_momentum.iloc[-1])
    dx = (x - float(tail.rs_ratio.iloc[-3])) / 2
    dy = (y - float(tail.rs_momentum.iloc[-3])) / 2
    angle = math.degrees(math.atan2(dy, dx)) % 360
    speed = math.hypot(x - tail.rs_ratio.iloc[-2], y - tail.rs_momentum.iloc[-2])
    candidates = []
    if dx and (100 - x) / dx > 0:
        candidates.append(((100 - x) / dx, "x"))
    if dy and (100 - y) / dy > 0:
        candidates.append(((100 - y) / dy, "y"))
    crossing = None
    if candidates:
        weeks, axis = min(candidates)
        if weeks <= CROSS_HORIZON:
            nx = 200 - x if axis == "x" else x   # mirror across the crossed line
            ny = 200 - y if axis == "y" else y
            crossing = {"weeks": round(weeks, 1), "to": quadrant(nx, ny)}
    return {
        "heading_deg": round(angle),
        "heading_arrow": ARROWS[int(((angle + 22.5) % 360) // 45)],
        "speed": round(float(speed), 2),
        "distance": round(math.hypot(x - 100, y - 100), 2),
        "next_crossing": crossing,
    }


def already_current() -> bool:
    """True if rrg_latest.json already covers the most recent completed Friday."""
    f = OUT / "rrg_latest.json"
    if os.environ.get("FORCE") == "1" or not f.exists():
        return False
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    # most recent Friday whose close has happened (Saturday onward counts the same week)
    last_friday = today - pd.Timedelta(days=(today.weekday() - 4) % 7)
    if today.weekday() == 4:
        last_friday -= pd.Timedelta(days=7)
    return json.loads(f.read_text()).get("week_ending", "") >= last_friday.strftime("%Y-%m-%d")


def main() -> None:
    if already_current():
        print("Data already current; nothing to do.")
        return
    daily, source = get_prices()
    weekly = to_weekly(daily)
    result = rrg(weekly)

    OUT.mkdir(parents=True, exist_ok=True)
    weekly.round(4).to_csv(OUT / "prices_weekly.csv", index_label="week_ending")

    hist = pd.concat({t: df for t, df in result.items()}, names=["ticker", "week_ending"])
    hist.round(3).reset_index().to_csv(OUT / "rrg_history.csv", index=False)

    as_of = min(df.index[-1] for df in result.values())
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "week_ending": as_of.strftime("%Y-%m-%d"),
        "price_source": source,
        "benchmark": BENCHMARK,
        "settings_weeks": {"short": SHORT, "long": LONG, "momentum": MOM},
        "method": "Approximation of JdK RS-Ratio / RS-Momentum (not the proprietary RRG formula)",
        "sectors": [],
    }
    for t, df in result.items():
        tail = df.loc[:as_of].tail(TAIL)
        last = tail.iloc[-1]
        close = weekly[t].loc[:as_of]
        payload["sectors"].append({
            "ticker": t,
            "name": SECTORS[t],
            "price": round(float(close.iloc[-1]), 2),
            "tail_change_pct": round(float((close.iloc[-1] / close.iloc[-TAIL] - 1) * 100), 2),
            "rs_ratio": round(float(last.rs_ratio), 2),
            "rs_momentum": round(float(last.rs_momentum), 2),
            "quadrant": quadrant(last.rs_ratio, last.rs_momentum),
            **direction(tail),
            "tail": [
                {"week": d.strftime("%Y-%m-%d"),
                 "rs_ratio": round(float(r.rs_ratio), 2),
                 "rs_momentum": round(float(r.rs_momentum), 2)}
                for d, r in tail.iterrows()
            ],
        })
    (OUT / "rrg_latest.json").write_text(json.dumps(payload, indent=2))
    print(f"Week ending {payload['week_ending']} from {source}")
    for s in payload["sectors"]:
        print(f"  {s['ticker']:5} {s['rs_ratio']:7.2f} {s['rs_momentum']:7.2f}  {s['quadrant']}")


if __name__ == "__main__":
    main()
