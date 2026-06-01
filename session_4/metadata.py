from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "datasets" / "Churn_Modelling_train_test.csv"
MODELS_DIR = BASE_DIR / "models"

DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
BINARY_FEATURES = ["Gender"]
ONE_HOT_ENCODE_COLUMNS = ["Geography"]
TARGET_COLUMN = "Exited"

MODEL_PARAMS = {
    "max_depth": 8,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "random_state": 42,
    "class_weight": "balanced",
}