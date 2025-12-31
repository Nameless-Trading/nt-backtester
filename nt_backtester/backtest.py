import polars as pl
import datetime as dt
from covariance_matrix import get_covariance_matrix
from portfolio import get_optimal_weights
import os
import ray

# Suppress Ray GPU warning for CPU-only usage
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


def load_data(name: str) -> pl.DataFrame:
    return pl.scan_parquet(f"nt_backtester/data/{name}.parquet").collect()


@ray.remote
def backtest_step(
    alphas: pl.DataFrame, betas: pl.DataFrame, date_: dt.date, lambda_: float
):
    alphas_slice = alphas.filter(pl.col("date").eq(date_)).sort("date")
    betas_slice = betas.filter(pl.col("date").eq(date_)).sort("date")

    covariance_matrix = get_covariance_matrix(date_)

    optimal_weights = get_optimal_weights(
        alphas_slice, betas_slice, covariance_matrix, lambda_
    )

    return optimal_weights.select(pl.lit(date_).alias("date"), "ticker", "weight")


def backtest_parallel(
    alphas: pl.DataFrame, betas: pl.DataFrame, lambda_: float = 2.0
) -> pl.DataFrame:
    ray.init(
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
        ignore_reinit_error=True,
        num_cpus=os.cpu_count(),
    )

    dates = alphas["date"].unique().sort().to_list()

    # Put DataFrames in Ray's object store once to avoid repeated serialization
    alphas_ref = ray.put(alphas)
    betas_ref = ray.put(betas)

    weights_list_futures = [
        backtest_step.remote(alphas_ref, betas_ref, date_, lambda_) for date_ in dates
    ]

    weights_list = ray.get(weights_list_futures)

    return pl.concat(weights_list)


if __name__ == "__main__":
    alphas = load_data("alphas")
    betas = load_data("betas")
    lambda_ = 1024.0
    weights = backtest_parallel(alphas, betas, lambda_)
    weights.write_parquet(f"nt_backtester/data/weights_{int(lambda_)}.parquet")
