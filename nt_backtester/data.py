from clients import get_clickhouse_client
import datetime as dt
import polars as pl
import os


def get_stock_returns(start: dt.date, end: dt.date) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    stock_returns_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT
            u.date,
            u.ticker,
            s.return 
        FROM universe u
        INNER JOIN stock_returns s ON u.date = s.date AND u.ticker = s.ticker 
        WHERE u.date BETWEEN '{start}' AND '{end}'
        """
    )

    return (
        pl.from_arrow(stock_returns_arrow)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .sort("ticker", "date")
    )


def get_etf_returns(start: dt.date, end: dt.date) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    etf_returns_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT
            date,
            ticker,
            return 
        FROM etf_returns
        WHERE date BETWEEN '{start}' AND '{end}'
        """
    )

    return (
        pl.from_arrow(etf_returns_arrow)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .sort("ticker", "date")
    )


def get_alphas(start: dt.date, end: dt.date, signal_name: str) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    alphas_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT
            u.date,
            u.ticker,
            a.alpha 
        FROM universe u
        INNER JOIN alphas a ON u.date = a.date AND u.ticker = a.ticker 
        WHERE u.date BETWEEN '{start}' AND '{end}'
            AND signal = '{signal_name}'
        """
    )

    return (
        pl.from_arrow(alphas_arrow)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .sort("ticker", "date")
    )


def get_factor_loadings(start: dt.date, end: dt.date) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    factor_loadings_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT
            u.date,
            u.ticker,
            f.factor,
            f.loading 
        FROM universe u
        INNER JOIN factor_loadings f ON u.date = f.date AND u.ticker = f.ticker 
        WHERE u.date BETWEEN '{start}' AND '{end}'
        """
    )

    return (
        pl.from_arrow(factor_loadings_arrow)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .sort("ticker", "date", "factor")
    )


def get_idio_vol(start: dt.date, end: dt.date) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    idio_vol_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT
            u.date,
            u.ticker,
            i.idio_vol
        FROM universe u
        INNER JOIN idio_vol i ON u.date = i.date AND u.ticker = i.ticker 
        WHERE u.date BETWEEN '{start}' AND '{end}'
        """
    )

    return (
        pl.from_arrow(idio_vol_arrow)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .sort("ticker", "date")
    )


def get_factor_covariances(start: dt.date, end: dt.date) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    factor_covariances_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT *
        FROM factor_covariances
        WHERE date BETWEEN '{start}' AND '{end}'
        """
    )

    return (
        pl.from_arrow(factor_covariances_arrow)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .sort("factor_1", "factor_2", "date")
    )


def get_betas(start: dt.date, end: dt.date) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    betas_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT 
            u.ticker,
            u.date,
            b.predicted_beta AS beta
        FROM universe u
        INNER JOIN betas b ON u.date = b.date AND u.ticker = b.ticker
        WHERE u.date BETWEEN '{start}' AND '{end}'
        """
    )

    return (
        pl.from_arrow(betas_arrow)
        .select("ticker", pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"), "beta")
        .sort("ticker", "date")
    )


def get_benchmark_returns(start: dt.date, end: dt.date) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    benchmark_returns_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT
            date,
            return
        FROM benchmark_returns
        WHERE date BETWEEN '{start}' AND '{end}'
        """
    )

    return (
        pl.from_arrow(benchmark_returns_arrow)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .sort("date")
    )


def get_benchmark_weights(start: dt.date, end: dt.date) -> pl.DataFrame:
    clickhouse_client = get_clickhouse_client()

    benchmark_weights_arrow = clickhouse_client.query_arrow(
        f"""
        SELECT
            date,
            ticker,
            weight
        FROM benchmark_weights
        WHERE date BETWEEN '{start}' AND '{end}'
        """
    )

    return (
        pl.from_arrow(benchmark_weights_arrow)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .sort("date")
    )


def download_data():
    os.makedirs("nt_backtester/data", exist_ok=True)
    start = dt.date(2022, 7, 29)
    end = dt.date(2025, 12, 31)
    signal_name = "reversal"

    get_stock_returns(start, end).write_parquet(
        "nt_backtester/data/stock_returns.parquet"
    )
    get_etf_returns(start, end).write_parquet("nt_backtester/data/etf_returns.parquet")
    get_alphas(start, end, signal_name).write_parquet(
        "nt_backtester/data/alphas.parquet"
    )
    get_factor_loadings(start, end).write_parquet(
        "nt_backtester/data/factor_loadings.parquet"
    )
    get_idio_vol(start, end).write_parquet("nt_backtester/data/idio_vol.parquet")
    get_factor_covariances(start, end).write_parquet(
        "nt_backtester/data/factor_covariances.parquet"
    )
    get_betas(start, end).write_parquet("nt_backtester/data/betas.parquet")
    get_benchmark_returns(start, end).write_parquet(
        "nt_backtester/data/benchmark_returns.parquet"
    )
    get_benchmark_weights(start, end).write_parquet(
        "nt_backtester/data/benchmark_weights.parquet"
    )


if __name__ == "__main__":
    download_data()
