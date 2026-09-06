"""
Data Registration
------------------
Reads the tourism dataset from the repository's data folder, validates that
every expected column is present, and prints a short summary so that any
schema drift is caught early in the pipeline.
"""

import os
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: dataset is missing expected columns: {missing_cols}")
        sys.exit(1)

    print("Dataset registered successfully.")
    print(f"Path              : {DATA_PATH}")
    print(f"Rows, Columns     : {df.shape[0]}, {df.shape[1]}")
    print(f"Expected columns  : all {len(EXPECTED_COLUMNS)} present")
    print()
    print("Missing values per column:")
    print(df[EXPECTED_COLUMNS].isnull().sum())
    print()
    print("Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts())
    print()
    print("Dtypes:")
    print(df[EXPECTED_COLUMNS].dtypes)


if __name__ == "__main__":
    main()
