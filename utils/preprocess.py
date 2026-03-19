import pandas as pd

def load_data():
    data = pd.read_csv("data/rockfall_dataset.csv")

    X = data.drop("rockfall", axis=1)
    y = data["rockfall"]

    return X, y