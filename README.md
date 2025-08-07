Projet Iris Classification — Data Science
Description
Ce projet consiste à classifier les fleurs d’iris en trois espèces à partir de leurs caractéristiques physiques, en utilisant plusieurs modèles de machine learning.
Données
Les données proviennent du jeu de données classique Iris disponible dans la bibliothèque sklearn.datasets.
Le dataset comprend 150 observations, chacune avec 4 caractéristiques : longueur et largeur des sépales et pétales.
Étapes du projet
Exploration des données
Préparation et nettoyage (minimal ici car données propres)
Modélisation avec plusieurs algorithmes : Decision Tree, SVM, KNN, Random Forest
Évaluation via validation croisée
Analyse des résultats (accuracy, matrice de confusion)
Comparaison des modèles pour choisir le meilleur
Résultats
Le modèle KNN a obtenu la meilleure accuracy moyenne de 97,3 % avec une faible variance.
La matrice de confusion montre une excellente séparation des classes, notamment une parfaite classification de Setosa.

Installation et utilisation

Cloner le dépôt :
git clone git@github.com:Lolohe13772/iris-classification.git
cd iris-classification

Installer les dépendances :
pip install -r requirements.txt

Utilisation des scripts :

Exploration des données :
python3 iris_exploration.py

Entraînement et évaluation des modèles :
python3 iris_model.py

Validation croisée :
python3 iris_crossval.py


Comparaison des modèles :
python3 iris_compare_models.py

Optimisation des hyperparamètres :
python3 iris_gridsearch.py

Évaluation finale (matrice de confusion et rapport de classification) :
python3 iris_evaluation.py

Pipeline et sauvegarde du modèle

Un pipeline est utilisé pour enchaîner automatiquement les étapes de prétraitement (normalisation) et de classification (modèle KNN).  
Cela garantit que les mêmes transformations sont appliquées aux données à chaque fois, facilitant ainsi la maintenance et la reproductibilité.  

Le pipeline entraîné est sauvegardé dans un fichier (`iris_knn_pipeline.joblib`) pour pouvoir être rechargé facilement et utilisé en production sans réentraînement.


Résultats et conclusion
Plusieurs modèles ont été testés, notamment Decision Tree, SVM, KNN et Random Forest.
Le modèle KNN a obtenu la meilleure accuracy moyenne de 97,3 % après validation croisée.
La validation croisée a permis d’assurer la robustesse et la généralisation des modèles, évitant le surapprentissage.
La matrice de confusion finale montre une excellente classification des trois espèces d’iris sans erreur.
Ce projet illustre les étapes clés d’un workflow de machine learning complet : exploration, modélisation, validation, comparaison et optimisation.

Laurent Henon – Data Scientist