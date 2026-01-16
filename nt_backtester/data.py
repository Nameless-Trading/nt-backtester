from clients import get_bear_lake_client
import datetime as dt
import polars as pl
import os
import bear_lake as bl

def get_stock_returns(start: dt.date, end: dt.date) -> pl.DataFrame:
    bear_lake_client = get_bear_lake_client()

    return (
        bear_lake_client.query(
            bl.table("stock_returns")
            .filter(pl.col("date").is_between(start, end))
            .select("date", "ticker", "return")
            .sort("ticker", "date")
        )
    )


def get_etf_returns(start: dt.date, end: dt.date) -> pl.DataFrame:
    bear_lake_client = get_bear_lake_client()

    return (
        bear_lake_client.query(
            bl.table("etf_returns")
            .filter(pl.col("date").is_between(start, end))
            .select("date", "ticker", "return")
            .sort("ticker", "date")
        )
    )


def get_alphas(start: dt.date, end: dt.date, signal_name: str) -> pl.DataFrame:
    bear_lake_client = get_bear_lake_client()

    return (
        bear_lake_client.query(
            bl.table("universe")
            .join(other=bl.table("alphas"), on=["date", "ticker"], how="left")
            .filter(pl.col("date").is_between(start, end), pl.col("alpha").is_not_null(), pl.col('signal').eq(signal_name))
            .select("date", "ticker", "alpha")
            .sort("ticker", "date")
        )
    )


def get_factor_loadings(start: dt.date, end: dt.date) -> pl.DataFrame:
    bear_lake_client = get_bear_lake_client()

    return (
        bear_lake_client.query(
            bl.table("universe")
            .join(other=bl.table("factor_loadings"), on=["date", "ticker"], how="left")
            .filter(pl.col("date").is_between(start, end), pl.col("loading").is_not_null())
            .select("date", "ticker", "factor", "loading")
            .sort("ticker", "date")
        )
    )


def get_idio_vol(start: dt.date, end: dt.date) -> pl.DataFrame:
    bear_lake_client = get_bear_lake_client()

    return (
        bear_lake_client.query(
            bl.table("universe")
            .join(other=bl.table("idio_vol"), on=["date", "ticker"], how="left")
            .filter(pl.col("date").is_between(start, end), pl.col("idio_vol").is_not_null())
            .select("date", "ticker", "idio_vol")
            .sort("ticker", "date")
        )
    )


def get_factor_covariances(start: dt.date, end: dt.date) -> pl.DataFrame:
    bear_lake_client = get_bear_lake_client()

    return (
        bear_lake_client.query(
            bl.table("factor_covariances")
            .filter(pl.col("date").is_between(start, end))
            .select("date", "factor_1", "factor_2", "covariance")
            .sort("date")
        )
    )


def get_benchmark_returns(start: dt.date, end: dt.date) -> pl.DataFrame:
    bear_lake_client = get_bear_lake_client()

    return (
        bear_lake_client.query(
            bl.table("benchmark_returns").filter(pl.col("date").is_between(start, end))
        )
    )


def get_benchmark_weights(start: dt.date, end: dt.date) -> pl.DataFrame:
    bear_lake_client = get_bear_lake_client()

    return (
        bear_lake_client.query(
            bl.table("universe")
            .join(other=bl.table("benchmark_weights"), on=["date", "ticker"], how="left")
            .filter(pl.col("date").is_between(start, end))
            .select("date", "ticker", "weight")
            .sort("ticker", "date")
        )
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
    get_benchmark_returns(start, end).write_parquet(
        "nt_backtester/data/benchmark_returns.parquet"
    )
    get_benchmark_weights(start, end).write_parquet(
        "nt_backtester/data/benchmark_weights.parquet"
    )


if __name__ == "__main__":
    download_data()
