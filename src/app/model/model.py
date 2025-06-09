import logging
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TitanicModel:
    """A class to preprocess Titanic dataset features and train a logistic regression model.
    This class handles both categorical and numerical features, applying appropriate preprocessing steps
    such as imputation and scaling. It also includes a logistic regression model for classification tasks.
    """

    def __init__(self) -> None:
        """Initializes the TitanicModel with preprocessing pipelines and a logistic regression model."""

        self.nomical_cat_features = ["sex"]
        self.ordinal_cat_features = ["pclass"]
        self.num_features = ["age"]

        self.nomical_cat_features_preprocess = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        self.ordinal_cat_features_preprocess = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        self.num_features_preprocess = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("nominal_cat", self.nomical_cat_features_preprocess, self.nomical_cat_features),
                ("ordinal_cat", self.ordinal_cat_features_preprocess, self.ordinal_cat_features),
                ("num", self.num_features_preprocess, self.num_features),
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: Union[pd.Series | np.ndarray]) -> None:
        """Fits the model to the training data.

        Arguments:
        ----------
        X : pd.DataFrame
            The input features for training.
        y : pd.Series or np.ndarray
            The target variable for training.
        """

        logging.info("Fitting the Titanic model...")

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Union[np.ndarray, Any]:
        """Predicts the target variable for the given input features.

        Arguments:
        ----------
        X : pd.DataFrame
            The input features for prediction.

        Returns:
        -------
        np.ndarray
            The predicted target variable.
        """

        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the probabilities of the target variable for the given input features.

        Arguments:
        ----------
        X : pd.DataFrame
            The input features for prediction.

        Returns:
        -------
        np.ndarray
            The predicted probabilities of the target variable.
        """

        return np.asarray(self.model.predict_proba(X)[:, 1])

    def save_model(self, file_path: str) -> None:
        """Saves the trained model to a file.

        Arguments:
        ----------
        file_path : str
            The path where the model will be saved.
        """

        logging.info("Saving the model to %s...", file_path)

        joblib.dump(self.model, file_path)

    def load_model(self, file_path: str) -> None:
        """Loads a trained model from a file.

        Arguments:
        ----------
        file_path : str
            The path from where the model will be loaded.
        """
        logging.info("Loading the model from %s...", file_path)

        self.model = joblib.load(file_path)
