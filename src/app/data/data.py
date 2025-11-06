"""Download and return the Titanic dataset."""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def get_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches the Titanic dataset from seaborn."""

    dataset = sns.load_dataset("titanic")
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    return train, test
