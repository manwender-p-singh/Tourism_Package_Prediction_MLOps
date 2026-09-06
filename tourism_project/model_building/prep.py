"""
Data Preparation
-----------------
Loads the raw dataset from the repository's data folder, cleans it, drops
identifier columns that carry no predictive signal, and splits it into
train/test sets. The splits are saved locally so the workflow can pass them
to the next job as an artifact.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Columns that carry no predictive signal (row index / unique ID)
DROP_COLS = ["Unnamed: 0", "CustomerID"]


def main():
    df = pd.read_csv(DATA_PATH)

    # Drop unnecessary columns (only if present, so the script is safe to
    # re-run on an already-cleaned file)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Fix a known data-entry inconsistency in Gender ("Fe Male" -> "Female")
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
    print(f"ytrain: {ytrain.shape}, ytest: {ytest.shape}")
    print(f"Train target distribution:\n{ytrain.value_counts(normalize=True)}")


if __name__ == "__main__":
    main()
