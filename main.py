from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# On charge le modèle entraîné (supposons que tu l'as sauvegardé en iris_model.pkl)
model = joblib.load("iris_model.pkl")

app = FastAPI()

# Définition du format d'entrée attendu
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Iris Classification"}

@app.post("/predict")
def predict(iris: IrisFeatures):
    features = [[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]]
    prediction = model.predict(features)
    species = ["setosa", "versicolor", "virginica"]
    return {"prediction": species[prediction[0]]}
