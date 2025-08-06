from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Charger les données iris
iris = load_iris()
X, y = iris.data, iris.target

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner un modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prédire et évaluer
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

from joblib import dump

# Sauvegarder le modèle
dump(model, "iris_model.joblib")
print("Modèle sauvegardé dans iris_model.joblib")