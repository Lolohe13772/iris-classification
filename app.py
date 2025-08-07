from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route("/", methods=["GET", "POST"])
def predict_form():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"])
            ]
            prediction = int(model.predict([features])[0])
        except Exception as e:
            prediction = f"Erreur : {e}"

    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

