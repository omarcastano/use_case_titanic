"""Download and return the Titanic dataset."""

import pandas as pd
import seaborn as sns


def get_dataset() -> pd.DataFrame:
    """Fetches the Titanic dataset from seaborn."""

    dataset = sns.load_dataset("titanic")

    return dataset
