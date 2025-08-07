import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

# Charger les données
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Visualisation de la distribution des longueurs de pétales
plt.figure(figsize=(8, 5))
sns.histplot(df['petal length (cm)'], bins=20, kde=True)
plt.title('Distribution des longueurs de pétales')
plt.show()

# Scatter plot : longueur vs largeur de pétales, coloré par classe
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='target', palette='Set1')
plt.title('Longueur vs Largeur des pétales selon la classe')
plt.show()
