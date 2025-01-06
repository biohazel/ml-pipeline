import pickle

def predict_local(features):
    with open("models/best_model.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    encoder = data["encoder"]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    predicted_species = encoder.inverse_transform(prediction)
    return predicted_species[0]

if __name__ == "__main__":
    example = [5.1, 3.5, 1.4, 0.2]
    result = predict_local(example)
    print(f"Predicted species: {result}")

