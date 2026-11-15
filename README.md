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
| `data/rrg_latest.json` | Latest week, 5-week tail, price, 5-week change, heading, speed, distance and projected next quadrant crossing per sector; read by the Monday report |
| `data/rrg_history.csv` | Every computed week, for checking how signals played out |
| `data/prices_weekly.csv` | Weekly closes used in the calculation (about 5 years) |
| `data/top3_history.csv` | Each week's top-3 outlook picks, to track how they did |

To run it by hand: **Actions** tab → **Weekly sector RRG data** → **Run workflow**.

Settings (`SHORT`, `LONG`, `MOM`, `TAIL`, ticker list) are at the top of `rrg/compute_rrg.py`.

Not financial advice.

## Weekly outlook (top 3)

For each sector the job finds its current setup (quadrant + whether RS-Momentum rose or fell last week)
and looks back about four years for how often the sector's price was higher one week later in that same
setup. The sector's own history is blended with all sectors' history for that setup, so a handful of
weeks can't produce an extreme number. The three highest become the report's executive summary.
These are historical base rates, not forecasts.
