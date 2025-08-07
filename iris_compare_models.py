import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Charger les données
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = df[iris.feature_names]
y = iris.target

# Liste des modèles à tester
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Évaluer chaque modèle avec validation croisée
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} - Accuracy moyenne : {scores.mean():.3f} (écart-type : {scores.std():.3f})")
