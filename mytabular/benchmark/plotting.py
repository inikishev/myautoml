from typing import Literal
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import load_iris


def plot_decision_boundary(
    estimator,
    X,
    y,
    response_method: Literal["auto", "predict_proba", "decision_function", "predict"] = "auto",
    alpha=0.5,
    scatter_size: int = 15,
    scatter_alpha: float = 0.5,
):
    estimator.fit(X, y)

    disp = DecisionBoundaryDisplay.from_estimator(
        estimator, X,
        response_method = response_method,
        alpha = alpha,
        plot_method = "pcolormesh",
    )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", linewidths=0.5, s=scatter_size)
    return disp


def plot_iris(
    estimator,
    response_method: Literal["auto", "predict_proba", "decision_function", "predict"] = "auto",
):
    X, y = load_iris(return_X_y=True)
    return plot_decision_boundary(estimator, X[:, :2], y, response_method=response_method)