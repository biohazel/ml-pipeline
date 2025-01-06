import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_data

def train_models(csv_path="data/raw/iris.csv"):
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(csv_path)
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    if rf_acc > lr_acc:
        best_model = rf
        best_acc = rf_acc
    else:
        best_model = lr
        best_acc = lr_acc
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "encoder": encoder
        }, f)
    return best_acc

if __name__ == "__main__":
    best_accuracy = train_models()
    print(f"Best model accuracy: {best_accuracy:.3f}")

