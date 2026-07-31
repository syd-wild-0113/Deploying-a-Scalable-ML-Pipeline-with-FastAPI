import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

@pytest.fixture
def sample_data():
    """
    Fixture providing a small synthetic dataset for testing
    """
    data = pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28],
            "workclass": [
                "State-gov",
                "Self-emp-not-inc",
                "Private",
                "Private",
                "Private"
            ],
            "education": [
                "Bachelors",
                "Bachelors",
                "HS-grad",
                "11th",
                "Bachelors"
            ],
            "education-num": [13, 13, 9, 7, 13],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Never-married",
            ],
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                "Handlers-cleaners",
                "Prof-specialty",
            ],
            "relationship": [
                "Not-in-family",
                "Husband",
                "Not-in-family",
                "Husband",
                "Wife",
            ],
            "race": ["White", "White", "White", "Black", "Black"],
            "sex": ["Male", "Male", "Male", "Female", "Female"],
            "capital-gain": [2174, 0, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0, 0],
            "native-country": [
                "United-states",
                "United-states",
                "United-states",
                "United-states",
                "Cuba",
            ],
            "salary": ["<=50K", "<=50K", "<=50K", "<=50K", ">50K"],
        }
    )
    return data

# TODO: implement the first test. Change the function name and input as needed
def test_train_model(sample_data):
    """Test that train_model returns a trained RandomForestClassifier instance."""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)

    assert model is not None
    assert isinstance(model, RandomForestClassifier)

# TODO: implement the second test. Change the function name and input as needed
def test_inference(sample_data):
    """Test that inference returns predictions of the correct type and shape."""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(sample_data)
    assert set(preds).issubset({0, 1})


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """Test metric computation, returns floats between 0 and 1."""
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0

