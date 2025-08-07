from iris_model import train_model
import joblib

model = train_model()
joblib.dump(model, "iris_model.pkl")
print("Modèle sauvegardé dans iris_model.pkl")
