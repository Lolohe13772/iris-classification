import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Charger les données
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = df[iris.feature_names]
y = iris.target

# Créer le modèle
model = DecisionTreeClassifier(random_state=42)

# Validation croisée avec 5 folds
scores = cross_val_score(model, X, y, cv=5)

print("Scores des folds :", scores)
print("Accuracy moyenne :", scores.mean())
