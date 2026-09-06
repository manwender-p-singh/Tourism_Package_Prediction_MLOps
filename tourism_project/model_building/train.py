"""
Model Training with Experiment Tracking
-----------------------------------------
Loads the train/test splits produced by the data-prep job, builds a
preprocessing + XGBoost pipeline, tunes it with GridSearchCV, logs every
run's parameters and metrics to MLflow, and saves the best model so the
workflow can commit it to the repository for deployment.
"""

import pandas as pd
import joblib
import mlflow

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score,
)
import xgboost as xgb

MODEL_OUT_PATH = "tourism_project/deployment/model.joblib"

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def load_data():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline():
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    # Class imbalance is handled via scale_pos_weight, tuned via grid search.
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    pipeline = make_pipeline(preprocessor, model)
    return pipeline


def main():
    Xtrain, Xtest, ytrain, ytest = load_data()

    pipeline = build_pipeline()

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__scale_pos_weight": [1, 4],
    }

    mlflow.set_experiment("tourism-package-prediction")

    with mlflow.start_run():
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_

        # Log all the tuned parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        # Evaluate on the held-out test set
        preds = best_model.predict(Xtest)
        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "precision": precision_score(ytest, preds),
            "recall": recall_score(ytest, preds),
            "f1_score": f1_score(ytest, preds),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        print("Best parameters:", grid_search.best_params_)
        print()
        print("Test set performance:")
        print(classification_report(ytest, preds))

        # Save the classification report as an MLflow artifact
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(ytest, preds))
        mlflow.log_artifact(report_path)

        # Save the best model so the workflow can commit it to the repo
        joblib.dump(best_model, MODEL_OUT_PATH)
        print(f"\nBest model saved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
