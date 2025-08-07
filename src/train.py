import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

# Charger les données Iris
iris = load_iris()
X, y = iris.data, iris.target

def preprocess_data(df):
    # Exemple simple : séparer X et y, ici y = 'target' colonne
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

# Séparer en jeu d'entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Créer le dossier models/ s’il n’existe pas
if not os.path.exists("models"):
    os.makedirs("models")

# Sauvegarder le modèle
dump(model, "models/iris_model.joblib")

print("✅ Modèle entraîné et sauvegardé dans models/iris_model.joblib")
