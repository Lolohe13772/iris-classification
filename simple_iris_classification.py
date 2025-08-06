from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

def main():
    # 1. Charger les données Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Split train/test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Entraîner un modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Évaluer la performance
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # 5. Sauvegarder le modèle
    dump(model, "iris_random_forest.joblib")
    print("Modèle sauvegardé dans iris_random_forest.joblib")

if __name__ == "__main__":
    main()

