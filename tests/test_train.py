import os
from joblib import load
import numpy as np

def test_model_exists():
    """Vérifie que le modèle a bien été sauvegardé"""
    assert os.path.exists("models/iris_model.joblib"), "Le modèle n'existe pas"

def test_model_prediction():
    """Vérifie que le modèle peut faire une prédiction"""
    model = load("models/iris_model.joblib")
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Ex. Setosa
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2], "Prédiction invalide"



import os
from joblib import load
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from src.train import preprocess_data

def test_model_exists():
    assert os.path.exists("models/iris_model.joblib"), "Le modèle n'existe pas"

def test_model_prediction():
    model = load("models/iris_model.joblib")
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2], "Prédiction invalide"

def test_preprocessing_output_shape():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    X, y = preprocess_data()
    assert X.shape[0] == y.shape[0], "Mismatch entre X et y"
