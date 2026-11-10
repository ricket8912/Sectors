# Sectors

Data and tools behind the weekly **Monday Rotation Report**.

## Weekly RRG data

Every Sunday evening a GitHub Actions job (`.github/workflows/weekly-rrg.yml`)
downloads two years of daily prices for the 11 Select Sector SPDR ETFs and SPY,
converts them to weekly closes and computes an approximation of the
Relative Rotation Graph measures:

- **RS-Ratio** – each sector's relative strength vs SPY, 10-week average divided by 40-week average, centred on 100
- **RS-Momentum** – RS-Ratio divided by its own 10-week average

The real RRG formula (RRG Research / StockCharts) is proprietary. This is the
commonly published approximation, so positions should be close to StockCharts
but will not match exactly.

| File | Contents |
|---|---|
| `data/rrg_latest.json` | Latest week plus a 5-week tail per sector; read by the Monday report |
| `data/rrg_history.csv` | Every computed week, for checking how signals played out |
| `data/prices_weekly.csv` | Weekly closes used in the calculation |

To run it by hand: **Actions** tab → **Weekly sector RRG data** → **Run workflow**.

Settings (`SHORT`, `LONG`, `MOM`, `TAIL`, ticker list) are at the top of `rrg/compute_rrg.py`.

Not financial advice.
