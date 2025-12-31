import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import statsmodels.formula.api as smf

    return alt, pl, smf


@app.cell
def _(pl):
    weights = pl.read_parquet("../nt_backtester/data/weights_256.parquet")

    returns = (
        pl.read_parquet("../nt_backtester/data/stock_returns.parquet")
        .sort("date")
        .with_columns(pl.col("return").shift(-1).over("ticker"))
    )

    etf_returns = (
        pl.read_parquet("../nt_backtester/data/etf_returns.parquet")
        .sort("date")
        .with_columns(pl.col("return").shift(-1).over("ticker"))
        .filter(pl.col("date").ge(weights["date"].min()))
        .sort("date", "ticker")
    )
    return etf_returns, returns, weights


@app.cell
def _(pl, returns, weights):
    reversal_returns = (
        weights.join(other=returns, on=["date", "ticker"], how="left")
        .group_by("date")
        .agg(pl.col("return").mul(pl.col("weight")).sum())
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
def _(etf_returns, pl):
    spy_returns = (
        etf_returns.filter(pl.col("ticker").eq("SPY"))
        .with_columns(
            pl.lit("SPY").alias("portfolio"),
        )
        .drop("ticker")
        .sort("date")
    )
    return (spy_returns,)


@app.cell
def _(pl, spy_returns):
    cumulative_spy_returns = spy_returns.select(
        "date",
        "portfolio",
        pl.col("return").add(1).cum_prod().sub(1).mul(100).alias("cumulative_return"),
    )
    return (cumulative_spy_returns,)


@app.cell
def _(alt, cumulative_returns, cumulative_spy_returns, pl):
    (
        alt.Chart(pl.concat([cumulative_returns, cumulative_spy_returns]))
        .mark_line()
        .encode(
            x=alt.X("date", title=""),
            y=alt.Y("cumulative_return", title="Cumulative Return (%)"),
            color=alt.Color("portfolio", title="Portfolio"),
        )
    )
    return


@app.cell
def _(pl, reversal_returns, spy_returns):
    summary = (
        pl.concat([reversal_returns, spy_returns])
        .group_by("portfolio")
        .agg(
            pl.col("return").mean().mul(252 * 100).alias("mean"),
            pl.col("return").std().mul(pl.lit(252).sqrt() * 100).alias("stdev"),
        )
        .with_columns(pl.col("mean").truediv(pl.col("stdev")).alias("sharpe"))
        .with_columns(pl.exclude("portfolio").round(2))
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
        .with_columns(pl.exclude("date").mul(100))
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
def _():
    return


if __name__ == "__main__":
    app.run()
