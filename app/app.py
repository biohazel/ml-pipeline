import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("models/best_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
scaler = data["scaler"]
encoder = data["encoder"]

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    features = input_data["features"]
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)
    species = encoder.inverse_transform(prediction)
    return jsonify({"species": species[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

