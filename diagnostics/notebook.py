import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return alt, pl


@app.cell
def _(pl):
    weights = pl.read_parquet("nt_backtester/data/weights.parquet")
    returns = (
        pl.read_parquet("nt_backtester/data/stock_returns.parquet")
        .sort("date")
        .with_columns(pl.col("return").shift(-1).over("ticker"))
    )
    return returns, weights


@app.cell
def _(pl, returns, weights):
    portfolios = (
        weights.join(other=returns, on=["date", "ticker"], how="left")
        .group_by("date")
        .agg(pl.col("return").mul(pl.col("weight")).sum())
        .sort("date")
    )

    portfolios
    return (portfolios,)


@app.cell
def _(pl, portfolios):
    cumulative_returns = portfolios.sort("date").select(
        "date", pl.col("return").log1p().cum_sum().alias("cumulative_return")
    )

    cumulative_returns
    return (cumulative_returns,)


@app.cell
def _(alt, cumulative_returns):
    (
        alt.Chart(cumulative_returns)
        .mark_line()
        .encode(
            x=alt.X("date", title=""),
            y=alt.Y("cumulative_return", title="Cumulative Log Return (%)"),
        )
    )
    return


@app.cell
def _(pl, portfolios):
    summary = (
        portfolios.select(
            pl.col("return").mean().mul(252 * 100).alias("mean"),
            pl.col("return").std().mul(pl.lit(252).sqrt() * 100).alias("stdev"),
        )
        .with_columns(pl.col("mean").truediv(pl.col("stdev")).alias("sharpe"))
        .with_columns(pl.all().round(2))
    )

    summary
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
