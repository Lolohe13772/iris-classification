import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Charger les données
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = df[iris.feature_names]
y = iris.target

# 1. Visualiser l'arbre de décision
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Arbre de décision - Decision Tree")
plt.show()

# 2. Importance des variables avec Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

importances = pd.Series(rf_model.feature_importances_, index=iris.feature_names)
importances = importances.sort_values()

plt.figure(figsize=(10,6))
importances.plot(kind='barh')
plt.title("Importance des variables - Random Forest")
plt.show()
