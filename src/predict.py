# src/predict.py

import sys
import numpy as np
from joblib import load
from sklearn.datasets import load_iris

# Charger le modèle
model = load("models/iris_model.joblib")

# Récupérer les features depuis les arguments du terminal
# Exemple d'appel : python src/predict.py 5.1 3.5 1.4 0.2
if len(sys.argv) != 5:
    print("Usage : python src/predict.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
    sys.exit(1)

features = np.array(sys.argv[1:], dtype=float).reshape(1, -1)

# Faire la prédiction
prediction = model.predict(features)

# Obtenir le nom de la classe
iris = load_iris()
predicted_class = iris.target_names[prediction[0]]

print(f"🌸 Prédiction : cette fleur est probablement une '{predicted_class}'")
