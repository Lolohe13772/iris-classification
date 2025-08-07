from sklearn.datasets import load_iris
import pandas as pd

# Charger les données Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Afficher les 5 premières lignes
print(df.head())
