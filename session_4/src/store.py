from datetime import datetime

import joblib

from metadata import MODELS_DIR


def save_model(model, name: str = "aaron-glanz") -> str:
    """Save the trained model to the models folder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = MODELS_DIR / f"class_model-{name}-{timestamp}.joblib"

    joblib.dump(model, model_path)

    return str(model_path)