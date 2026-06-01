import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from metadata import DROP_COLUMNS, BINARY_FEATURES, ONE_HOT_ENCODE_COLUMNS


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns that should not be used for training."""
    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def map_binary_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary categorical variables to numeric values."""
    df = df.copy()

    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].map({"Female": 1, "Male": 0})

    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode selected categorical columns."""
    df = df.copy()

    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False
    ).set_output(transform="pandas")

    encoded_df = encoder.fit_transform(df[ONE_HOT_ENCODE_COLUMNS])

    df = df.drop(columns=ONE_HOT_ENCODE_COLUMNS)

    df = pd.concat(
        [df.reset_index(drop=True), encoded_df.reset_index(drop=True)],
        axis=1
    )

    return df


def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full transformation pipeline."""
    df = df.copy()
    df = drop_columns(df)
    df = map_binary_variables(df)
    df = one_hot_encode(df)

    return df