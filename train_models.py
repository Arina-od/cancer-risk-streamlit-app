
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from cancer_risk_config import (
    CANCER_SLUGS,
    CANCER_TYPE_MAP,
    CATEGORICAL_FEATURES,
    DATA_URL,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_TEMPLATE,
    NUMERIC_FEATURES,
    RAW_TO_UNIFIED_COLUMNS,
    TARGET_COLUMN,
)

RANDOM_STATE = 42
TEST_SIZE = 0.1
CV_FOLDS = 5

REG_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "KNN": KNeighborsRegressor(),
}

PARAM_GRID = {
    "Linear Regression": {},
    "Ridge Regression": {
        "model__alpha": np.linspace(0.001, 10, 20),
    },
    "Decision Tree": {
        "model__max_depth": np.arange(2, 15),
        "model__min_samples_split": np.arange(2, 10),
    },
    "KNN": {
        "model__n_neighbors": np.arange(1, 15),
        "model__weights": ["uniform", "distance"],
    },
    "Random Forest": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 3, 5],
        "model__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [2, 3],
    },
}


def load_source_data(data_path_or_url: str | Path = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(data_path_or_url)
    df = df.rename(columns=RAW_TO_UNIFIED_COLUMNS)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate_and_fit_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict]:
    results = []

    for name, model in REG_MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", clone(model)),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRID[name],
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        cv_scores = cross_validate(
            grid.best_estimator_,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            return_train_score=False,
            n_jobs=-1,
        )

        results.append(
            {
                "model_name": name,
                "best_params": grid.best_params_,
                "mean_rmse": float(-cv_scores["test_score"].mean()),
                "std_rmse": float(cv_scores["test_score"].std()),
                "best_estimator": grid.best_estimator_,
            }
        )

    results = sorted(results, key=lambda item: item["mean_rmse"])
    best = results[0]
    summary = {
        "leaderboard": [
            {
                "model_name": item["model_name"],
                "best_params": item["best_params"],
                "mean_rmse": item["mean_rmse"],
                "std_rmse": item["std_rmse"],
            }
            for item in results
        ],
        "selected_model": best["model_name"],
        "selected_params": best["best_params"],
        "selected_cv_mean_rmse": best["mean_rmse"],
        "selected_cv_std_rmse": best["std_rmse"],
    }
    return best["best_estimator"], summary


def train_single_cancer_model(df: pd.DataFrame, cancer_type: str) -> tuple[dict, Pipeline]:
    subset = df.loc[df["cancer_type"] == cancer_type].copy()
    X = subset[FEATURE_COLUMNS].copy()
    y = subset[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model, summary = evaluate_and_fit_best_model(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    summary.update(
        {
            "cancer_type_en": cancer_type,
            "cancer_type_ru": CANCER_TYPE_MAP[cancer_type],
            "metrics": metrics,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target_column": TARGET_COLUMN,
        }
    )
    return summary, model


def main() -> None:
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    df = load_source_data(DATA_URL)

    all_summaries: list[dict] = []

    for cancer_type in CANCER_TYPE_MAP:
        summary, model = train_single_cancer_model(df, cancer_type)
        slug = CANCER_SLUGS[cancer_type]
        artifact_path = project_root / MODEL_ARTIFACT_TEMPLATE.format(slug=slug)

        bundle = {
            "model": model,
            "metadata": summary,
        }
        joblib.dump(bundle, artifact_path)
        all_summaries.append(summary)

        print(
            f"[OK] {summary['cancer_type_ru']}: "
            f"{summary['selected_model']} | RMSE(test)={summary['metrics']['rmse']:.4f}"
        )

    summary_df = pd.DataFrame(
        [
            {
                "Тип рака": item["cancer_type_ru"],
                "Лучшая модель": item["selected_model"],
                "CV RMSE": item["selected_cv_mean_rmse"],
                "Test MAE": item["metrics"]["mae"],
                "Test RMSE": item["metrics"]["rmse"],
                "Test R2": item["metrics"]["r2"],
            }
            for item in all_summaries
        ]
    )
    summary_df.to_csv(reports_dir / "model_summary.csv", index=False)
    (reports_dir / "model_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nИтоговая таблица сохранена в reports/model_summary.csv")
    print("Артефакты моделей сохранены в папке models/")


if __name__ == "__main__":
    main()
