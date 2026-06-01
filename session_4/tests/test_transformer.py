import pandas as pd

from src.transform import (
    drop_columns,
    map_binary_variables,
    transform_dataset,
)


def test_drop_columns_removes_identifier_columns():
    df = pd.DataFrame({
        "RowNumber": [1],
        "CustomerId": [123],
        "Surname": ["Smith"],
        "CreditScore": [650],
    })

    result = drop_columns(df)

    assert "RowNumber" not in result.columns
    assert "CustomerId" not in result.columns
    assert "Surname" not in result.columns
    assert "CreditScore" in result.columns


def test_map_binary_variables_maps_gender_correctly():
    df = pd.DataFrame({
        "Gender": ["Female", "Male"]
    })

    result = map_binary_variables(df)

    assert result["Gender"].tolist() == [1, 0]


def test_transform_dataset_outputs_expected_columns():
    df = pd.DataFrame({
        "RowNumber": [1, 2],
        "CustomerId": [123, 456],
        "Surname": ["Smith", "Jones"],
        "CreditScore": [650, 700],
        "Geography": ["France", "Germany"],
        "Gender": ["Female", "Male"],
        "Age": [35, 42],
        "Tenure": [3, 5],
        "Balance": [1000.0, 2000.0],
        "NumOfProducts": [1, 2],
        "HasCrCard": [1, 0],
        "IsActiveMember": [1, 0],
        "EstimatedSalary": [50000.0, 60000.0],
        "Exited": [0, 1],
    })

    result = transform_dataset(df)

    assert "RowNumber" not in result.columns
    assert "CustomerId" not in result.columns
    assert "Surname" not in result.columns
    assert "Geography" not in result.columns
    assert "Gender" in result.columns
    assert result["Gender"].tolist() == [1, 0]
    assert "Exited" in result.columns