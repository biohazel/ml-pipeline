import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df

if __name__ == "__main__":
    df_iris = load_data()
    df_iris.to_csv("data/raw/iris_raw.csv", index=False)

