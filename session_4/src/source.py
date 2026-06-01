import pandas as pd

from metadata import DATA_PATH


def load_dataset() -> pd.DataFrame:
    """Load the churn dataset from CSV."""
    return pd.read_csv(DATA_PATH)