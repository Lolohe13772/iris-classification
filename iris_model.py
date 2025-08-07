import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Charger les données
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Préparer les données
X = df[iris.feature_names]
y = df['target']

# Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prédire sur test
y_pred = model.predict(X_test)

# Évaluer
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
