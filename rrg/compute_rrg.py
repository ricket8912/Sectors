"""
Weekly sector rotation data for the Monday Rotation Report.

Downloads about two years of daily prices for the 11 Select Sector SPDR ETFs
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

import io
import json
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
WINDOW = 14   # weeks used to normalize RS and its momentum
TAIL = 10     # weeks of tail shown on the chart
OUT = Path(__file__).resolve().parent.parent / "data"


# ---------- price download ----------
def from_yahoo(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                     progress=False, threads=False)
    closes = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    return closes.dropna(how="all")


def from_stooq(tickers: list[str]) -> pd.DataFrame:
    import urllib.request
    frames = {}
    for t in tickers:
        url = f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
        with urllib.request.urlopen(url, timeout=30) as r:
            text = r.read().decode()
        d = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
        frames[t] = d["Close"]
        time.sleep(1)
    df = pd.DataFrame(frames)
    return df[df.index >= df.index.max() - pd.Timedelta(days=730)]


def get_prices() -> tuple[pd.DataFrame, str]:
    tickers = list(SECTORS) + [BENCHMARK]
    errors = []
    for name, fn in (("Yahoo Finance", from_yahoo), ("Stooq", from_stooq)):
        for attempt in range(3):
            try:
                px = fn(tickers)
                missing = [t for t in tickers if t not in px or px[t].dropna().empty]
                if missing:
                    raise ValueError(f"missing tickers {missing}")
                return px, name
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name} attempt {attempt + 1}: {e}")
                time.sleep(5 * (attempt + 1))
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


def rrg(weekly: pd.DataFrame, window: int = WINDOW) -> dict[str, pd.DataFrame]:
    """RS-Ratio and RS-Momentum approximation, both centred on 100."""
    out = {}
    bench = weekly[BENCHMARK]
    for t in SECTORS:
        rs = 100 * weekly[t] / bench
        ratio = 100 + (rs - rs.rolling(window).mean()) / rs.rolling(window).std(ddof=0)
        roc = 100 * (ratio / ratio.shift(1) - 1)
        mom = 100 + (roc - roc.rolling(window).mean()) / roc.rolling(window).std(ddof=0)
        out[t] = pd.DataFrame({"rs_ratio": ratio, "rs_momentum": mom}).dropna()
    return out


def quadrant(ratio: float, mom: float) -> str:
    if ratio >= 100:
        return "Leading" if mom >= 100 else "Weakening"
    return "Improving" if mom >= 100 else "Lagging"


def main() -> None:
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
        "window_weeks": WINDOW,
        "method": "Approximation of JdK RS-Ratio / RS-Momentum (not the proprietary RRG formula)",
        "sectors": [],
    }
    for t, df in result.items():
        tail = df.loc[:as_of].tail(TAIL)
        last = tail.iloc[-1]
        payload["sectors"].append({
            "ticker": t,
            "name": SECTORS[t],
            "rs_ratio": round(float(last.rs_ratio), 2),
            "rs_momentum": round(float(last.rs_momentum), 2),
            "quadrant": quadrant(last.rs_ratio, last.rs_momentum),
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
