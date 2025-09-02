from flask import Flask, request, jsonify, render_template
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load("model/food_delivery_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    
    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    age = model.predict(img)[0]
    return jsonify({"predicted_age": int(age)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # For Render
    app.run(host="0.0.0.0", port=port)
