import polars as pl
import datetime as dt
from covariance_matrix import get_covariance_matrix
from portfolio import get_optimal_weights, get_active_risk
import os
import ray
import numpy as np

# Suppress Ray GPU warning for CPU-only usage
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


def load_data(name: str) -> pl.DataFrame:
    return pl.scan_parquet(f"nt_backtester/data/{name}.parquet").collect()


@ray.remote
def backtest_step_parallel(
    alphas: pl.DataFrame,
    benchmark_weights: pl.DataFrame,
    date_: dt.date,
    lambda_: str | float,
    target_active_risk: float | None = None,
):
    alphas_slice = alphas.filter(pl.col("date").eq(date_)).sort("ticker")
    tickers = alphas_slice['ticker'].to_list()
    benchmark_weights_slice = benchmark_weights.filter(pl.col("date").eq(date_)).sort(
        "date"
    )

    covariance_matrix = get_covariance_matrix(date_, tickers)

    if isinstance(lambda_, str) and lambda_ == "dynamic":
        optimal_weights, final_lambda, final_active_risk = get_optimal_weights_dynamic(
            alphas_slice,
            covariance_matrix,
            benchmark_weights_slice,
            target_active_risk,
        )

    elif isinstance(lambda_, (int, float)):
        optimal_weights = get_optimal_weights(
            alphas_slice, covariance_matrix, lambda_
        )
        final_lambda = lambda_
        # Calculate active risk for fixed lambda
        active_weights = get_active_weights(optimal_weights, benchmark_weights_slice)
        final_active_risk = get_active_risk(active_weights, covariance_matrix)

    else:
        raise ValueError(f"Lambda not supported:", lambda_)

    weights_df = optimal_weights.select(pl.lit(date_).alias("date"), "ticker", "weight")

    metrics_df = pl.DataFrame(
        {
            "date": [date_],
            "lambda": [final_lambda],
            "active_risk": [final_active_risk],
        }
    )

    return weights_df, metrics_df


def backtest_parallel(
    alphas: pl.DataFrame,
    benchmark_weights: pl.DataFrame,
    lambda_: str | float,
    target_active_risk: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ray.init(
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
        ignore_reinit_error=True,
        num_cpus=os.cpu_count(),
    )

    dates = alphas["date"].unique().sort().to_list()

    # Put DataFrames in Ray's object store once to avoid repeated serialization
    alphas_ref = ray.put(alphas)
    benchmark_weights_ref = ray.put(benchmark_weights)

    weights_list_futures = [
        backtest_step_parallel.remote(
            alphas_ref,
            benchmark_weights_ref,
            date_,
            lambda_,
            target_active_risk,
        )
        for date_ in dates
    ]

    results = ray.get(weights_list_futures)

    # Separate weights and metrics DataFrames
    weights_list = [r[0] for r in results]
    metrics_list = [r[1] for r in results]

    weights_df = pl.concat(weights_list)
    metrics_df = pl.concat(metrics_list).sort("date")

    return weights_df, metrics_df


def get_optimal_weights_dynamic(
    alphas: pl.DataFrame,
    covariance_matrix: pl.DataFrame,
    benchmark_weights: pl.DataFrame,
    target_active_risk: float = 0.05,
):
    active_risk = float("inf")
    lambda_ = None
    error = 0.005
    max_iterations = 5
    iterations = 1
    data = []

    while abs(active_risk - target_active_risk) > error:
        if lambda_ is None:
            lambda_ = 100
        else:
            lambda_ = predict_lambda(data, target_active_risk)

        optimal_weights = get_optimal_weights(alphas, covariance_matrix, lambda_)

        active_weights = get_active_weights(optimal_weights, benchmark_weights)
        active_risk = get_active_risk(active_weights, covariance_matrix)

        data.append((lambda_, active_risk))

        if iterations >= max_iterations:
            break
        else:
            iterations += 1

    return optimal_weights, lambda_, active_risk


def predict_lambda(data: list[tuple[float]], active_risk: float) -> float:
    def fit_model(data: np.ndarray) -> float:
        lambda_ = data[:, 0]
        sigma = data[:, 1]

        X = 1 / (2 * lambda_)

        M = np.dot(X, sigma) / np.dot(X, X)

        return M

    data = np.array(data)

    M = fit_model(data)

    return M / (2 * active_risk)


def backtest_step_sequential(
    alphas: pl.DataFrame,
    benchmark_weights: pl.DataFrame,
    date_: dt.date,
    lambda_: str | float,
    target_active_risk: float | None = None,
):
    alphas_slice = alphas.filter(pl.col("date").eq(date_)).sort("ticker")
    tickers = alphas_slice['ticker'].to_list()
    benchmark_weights_slice = benchmark_weights.filter(pl.col("date").eq(date_)).sort(
        "date"
    )

    covariance_matrix = get_covariance_matrix(date_, tickers)

    if isinstance(lambda_, str) and lambda_ == "dynamic":
        optimal_weights, final_lambda, final_active_risk = get_optimal_weights_dynamic(
            alphas_slice,
            covariance_matrix,
            benchmark_weights_slice,
            target_active_risk,
        )

    elif isinstance(lambda_, (int, float)):
        optimal_weights = get_optimal_weights(
            alphas_slice, covariance_matrix, lambda_
        )
        final_lambda = lambda_
        # Calculate active risk for fixed lambda
        active_weights = get_active_weights(optimal_weights, benchmark_weights_slice)
        final_active_risk = get_active_risk(active_weights, covariance_matrix)

    else:
        raise ValueError(f"Lambda not supported:", lambda_)

    weights_df = optimal_weights.select(pl.lit(date_).alias("date"), "ticker", "weight")

    metrics_df = pl.DataFrame(
        {
            "date": [date_],
            "lambda": [final_lambda],
            "active_risk": [final_active_risk],
        }
    )

    return weights_df, metrics_df


def get_active_weights(
    weights: pl.DataFrame, benchmark_weights: pl.DataFrame
) -> pl.DataFrame:
    return (
        weights.join(
            other=benchmark_weights.rename({"weight": "benchmark_weight"}).drop("date"),
            on=["ticker"],
            how="left",
        )
        .with_columns(
            pl.col("weight").sub(pl.col("benchmark_weight")).alias("active_weight")
        )
        .select("ticker", "active_weight")
    )


def backtest_sequential(
    alphas: pl.DataFrame,
    benchmark_weights: pl.DataFrame,
    lambda_: str | float,
    target_active_risk: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    dates = alphas["date"].unique().sort().to_list()

    weights_list = []
    metrics_list = []
    for date_ in dates:
        weights_df, metrics_df = backtest_step_sequential(
            alphas, benchmark_weights, date_, lambda_, target_active_risk
        )

        weights_list.append(weights_df)
        metrics_list.append(metrics_df)

    weights_df = pl.concat(weights_list)
    metrics_df = pl.concat(metrics_list).sort("date")

    return weights_df, metrics_df


if __name__ == "__main__":
    alphas = load_data("alphas")
    benchmark_weights = load_data("benchmark_weights")

    lambda_ = "dynamic"
    target_active_risk = 0.05

    weights, metrics = backtest_parallel(
        alphas, benchmark_weights, lambda_, target_active_risk
    )

    weights.write_parquet("nt_backtester/data/weights_star.parquet")
    metrics.write_parquet("nt_backtester/data/metrics_star.parquet")
