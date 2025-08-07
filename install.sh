#!/bin/bash

echo "Création du fichier requirements.txt..."
cat <<EOL > requirements.txt
scikit-learn==1.6.1
pandas==2.3.1
matplotlib==3.7.1
seaborn==0.12.2
EOL

echo "Installation des packages depuis requirements.txt..."
pip install -r requirements.txt

echo "Terminé !"