import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from metadata import MODEL_PARAMS, TARGET_COLUMN


def train_model(df: pd.DataFrame) -> DecisionTreeClassifier:
    """Train a Decision Tree model and return it."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    return model