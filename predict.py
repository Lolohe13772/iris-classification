from joblib import load
import numpy as np

# Charger le modèle sauvegardé
model = load("iris_model.joblib")

# Exemple de nouvel échantillon (4 features)
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Prédiction
prediction = model.predict(new_sample)
print(f"Classe prédite : {prediction[0]}")
