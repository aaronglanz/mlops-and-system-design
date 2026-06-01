from src.source import load_dataset
from src.store import save_model
from src.train import train_model
from src.transform import transform_dataset


def main():
    df = load_dataset()
    transformed_df = transform_dataset(df)
    model = train_model(transformed_df)
    model_path = save_model(model)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

