import polars as pl
import cvxpy as cp
import numpy as np


def solve_quadratic_problem(
    n_assets: int,
    alphas: np.ndarray,
    betas: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_: float,
):
    weights = cp.Variable(n_assets)

    objective = cp.Maximize(
        cp.matmul(weights, alphas)
        - 0.5 * lambda_ * cp.quad_form(weights, covariance_matrix)
    )

    constraints = [
        cp.sum(weights) == 1,  # Full investment
        weights >= 0,  # Long only
        # cp.matmul(weights, betas) == 1,  # Unit Beta
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value


def get_optimal_weights(
    alphas: pl.DataFrame,
    betas: pl.DataFrame,
    covariance_matrix: pl.DataFrame,
    lambda_: float,
) -> pl.DataFrame:
    tickers = alphas["ticker"].sort().to_list()

    optimal_weights = solve_quadratic_problem(
        n_assets=len(tickers),
        alphas=alphas["alpha"].to_numpy(),
        betas=betas["beta"].to_numpy(),
        covariance_matrix=covariance_matrix.drop("ticker").to_numpy(),
        lambda_=lambda_,
    )

    return pl.DataFrame({"ticker": tickers, "weight": optimal_weights})


def get_active_risk(
    active_weights: pl.DataFrame, covariance_matrix: pl.DataFrame
) -> float:
    active_weights = active_weights.sort("ticker")["active_weight"].to_numpy()
    covariance_matrix = covariance_matrix.drop("ticker").to_numpy()

    return np.sqrt(active_weights @ covariance_matrix @ active_weights.T) * np.sqrt(252)
