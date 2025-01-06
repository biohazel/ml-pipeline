import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(csv_path="data/raw/iris.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("species", axis=1)
    y = df["species"]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler, encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data()
    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

