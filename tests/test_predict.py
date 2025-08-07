import numpy as np
from joblib import load
import pytest

model = load("models/iris_model.joblib")

def test_valid_prediction():
    sample = np.array([[6.7, 3.1, 4.7, 1.5]])
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2]

def test_invalid_input_shape():
    with pytest.raises(ValueError):
        # Données avec 3 au lieu de 4 features
        model.predict(np.array([[6.7, 3.1, 4.7]]))
