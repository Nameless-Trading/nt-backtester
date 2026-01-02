import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import statsmodels.formula.api as smf
    import datetime as dt
    return alt, dt, mo, pl, smf


@app.cell
def _(dt, mo):
    start = dt.date(2021, 7, 28)
    end = dt.date(2025, 12, 31)

    view_start = mo.ui.date(label="Start Date", start=start, stop=end, value=start)
    view_end = mo.ui.date(label="End Date", start=start, stop=end, value=end)
    return view_end, view_start


@app.cell
def _(pl, view_end, view_start):
    weights = pl.read_parquet("../nt_backtester/data/weights_star.parquet")
    metrics = pl.read_parquet(
        "../nt_backtester/data/metrics_star.parquet"
    ).with_columns(pl.col("active_risk").mul(100))

    returns = (
        pl.read_parquet("../nt_backtester/data/stock_returns.parquet")
        .sort("date")
        .with_columns(pl.col("return").shift(-1).over("ticker"))
    )

    etf_returns = (
        pl.read_parquet("../nt_backtester/data/etf_returns.parquet")
        .sort("date")
        .with_columns(pl.col("return").shift(-1).over("ticker"))
        .filter(pl.col("date").is_between(view_start.value, view_end.value))
        .sort("date", "ticker")
    )

    benchmark_returns = (
        pl.read_parquet("../nt_backtester/data/benchmark_returns.parquet")
        .sort("date")
        .with_columns(
            pl.lit("Benchmark").alias("portfolio"), pl.col("return").shift(-1)
        )
        .filter(pl.col("date").is_between(view_start.value, view_end.value))
        .sort("date")
    )
    return benchmark_returns, etf_returns, metrics, returns, weights


@app.cell
def _(pl, returns, view_end, view_start, weights):
    reversal_returns = (
        weights.join(other=returns, on=["date", "ticker"], how="left")
        .group_by("date")
        .agg(pl.col("return").mul(pl.col("weight")).sum())
        .filter(pl.col("date").is_between(view_start.value, view_end.value))
        .sort("date")
        .with_columns(pl.lit("Reversal").alias("portfolio"))
    )
    return (reversal_returns,)


@app.cell
def _(pl, reversal_returns):
    cumulative_returns = reversal_returns.sort("date").select(
        "date",
        "portfolio",
        pl.col("return").add(1).cum_prod().sub(1).mul(100).alias("cumulative_return"),
    )
    return (cumulative_returns,)


@app.cell
def _(benchmark_returns, pl):
    cumulative_benchmark_returns = benchmark_returns.select(
        "date",
        "portfolio",
        pl.col("return").add(1).cum_prod().sub(1).mul(100).alias("cumulative_return"),
    )
    return (cumulative_benchmark_returns,)


@app.cell
def _(mo, view_end, view_start):
    mo.vstack([view_start, view_end])
    return


@app.cell
def _(alt, cumulative_benchmark_returns, cumulative_returns, pl):
    (
        alt.Chart(pl.concat([cumulative_returns, cumulative_benchmark_returns]))
        .mark_line()
        .encode(
            x=alt.X("date", title=""),
            y=alt.Y("cumulative_return", title="Cumulative Return (%)"),
            color=alt.Color("portfolio", title="Portfolio"),
        )
    )
    return


@app.cell
def _(benchmark_returns, pl, reversal_returns):
    summary = (
        pl.concat([reversal_returns, benchmark_returns])
        .group_by("portfolio")
        .agg(
            pl.col("return").mean().mul(252 * 100).alias("mean"),
            pl.col("return").std().mul(pl.lit(252).sqrt() * 100).alias("stdev"),
        )
        .with_columns(pl.col("mean").truediv(pl.col("stdev")).alias("sharpe"))
        .with_columns(pl.exclude("portfolio").round(2))
        .sort("portfolio")
    )

    summary
    return


@app.cell
def _(etf_returns, pl, reversal_returns):
    regression_data = (
        reversal_returns.drop("portfolio")
        .join(
            other=etf_returns.pivot(index="date", on="ticker", values="return"),
            on="date",
            how="left",
        )
        .rename({"return": "portfolio_return"})
        .with_columns(pl.exclude("date").mul(100 * 252))
        .sort("date")
    )
    return (regression_data,)


@app.cell
def _(regression_data, smf):
    formula = "portfolio_return ~ MTUM + QUAL + SPY + USMV + VLUE"
    model = smf.ols(formula=formula, data=regression_data)
    results = model.fit()

    results.summary()
    return


@app.cell
def _(alt, metrics):
    (
        alt.Chart(metrics)
        .mark_line()
        .encode(x=alt.X("date", title=""), y=alt.Y("active_risk", title="Active Risk"))
    )
    return


@app.cell
def _(alt, metrics):
    (
        alt.Chart(metrics)
        .mark_line()
        .encode(x=alt.X("date", title=""), y=alt.Y("lambda", title="Lambda"))
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
