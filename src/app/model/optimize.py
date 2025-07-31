"""
Optimize the Titanic model using GridSearchCV.
"""

import logging

import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.app.model.model import TitanicModel


class ModelOptimizer:
    """
    A class to optimize a model using GridSearchCV.

    Arguments:
    ----------
    model : TitanicModel
        The model to optimize.
    param_grid : dict
        The parameter grid to use for optimization.
    """

    def __init__(
        self,
        model: TitanicModel,
        param_grid: dict,
        mlflow_uri: str,
    ) -> None:
        """Initializes the ModelOptimizer with a model and parameter grid."""

        self.model = model
        self.param_grid = param_grid
        self.mlflow_uri = mlflow_uri

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Optimizes the model using GridSearchCV.

        Arguments:
        ----------
        X : pd.DataFrame
            The input features for optimization.
        y : pd.Series
            The target variable for optimization.
        """

        grid_search = GridSearchCV(self.model.model, self.param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X, y)
        self.model.model = grid_search.best_estimator_

        if self.mlflow_uri:
            try:
                # set mlflow server
                mlflow.set_tracking_uri(self.mlflow_uri)

                # set experiment name
                mlflow.set_experiment("titanic_model")

                with mlflow.start_run():
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("accuracy", grid_search.best_score_)
                    mlflow.sklearn.log_model(
                        sk_model=grid_search.best_estimator_,
                        name="titanic_model",
                        input_example=X.iloc[[0]],
                        signature=mlflow.models.infer_signature(X, y),
                        registered_model_name="Best Titanic Model",
                    )
            except Exception as e:
                logging.error("Failed to log model to MLflow: %s", str(e))

    def save_model(self, file_path: str) -> None:
        """Saves the optimized model to a file."""

        self.model.save_model(file_path)
