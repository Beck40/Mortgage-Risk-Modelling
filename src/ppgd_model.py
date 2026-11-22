"""PPGD modelling pipeline and fitted preprocessing utilities."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numerical values to bounds learned by quantile."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X_array, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return np.clip(X_array, self.lower_bounds_, self.upper_bounds_)


def build_preprocessor(numerical_columns, categorical_columns):
    """Build the exact Development-fitted preprocessing pipeline."""
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", QuantileWinsorizer(0.01, 0.99)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ])


def compute_rebalanced_weights(y, target_rate=0.20):
    """Compute sample weights that produce the target training prevalence."""
    y_array = np.asarray(y)
    observed_rate = y_array.mean()
    return compute_sample_weight(
        class_weight={
            0: (1 - target_rate) / (1 - observed_rate),
            1: target_rate / observed_rate,
        },
        y=y_array,
    )


def restore_population_prior(probabilities, sampled_rate, population_rate):
    """Undo a minority-class prior shift while preserving the probability scale."""
    probabilities = np.asarray(probabilities, dtype=float)
    odds = probabilities / np.clip(1 - probabilities, 1e-12, None)
    population_odds = odds * (population_rate / (1 - population_rate)) / (
        sampled_rate / (1 - sampled_rate)
    )
    return population_odds / (1 + population_odds)


def train_champion_xgboost(X_dev, y_dev, target_rate=0.20,
                           artefact_dir="data/model_artefacts"):
    """Tune and save the champion XGBoost model with exact cross-validation."""
    root = Path(__file__).resolve().parents[1]
    artefact_dir = root / artefact_dir if not Path(artefact_dir).is_absolute() else Path(artefact_dir)
    artefact_dir.mkdir(parents=True, exist_ok=True)
    cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [250, 400, 500],
        "learning_rate": [0.02, 0.04, 0.06],
        "max_depth": [4, 5, 6],
        "min_child_weight": [10, 20, 40],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.7, 0.85],
        "reg_alpha": [0.0, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    weights = compute_rebalanced_weights(y_dev, target_rate)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "neg_log_loss": "neg_log_loss"},
        refit="roc_auc",
        cv=cv_scheme,
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_dev, y_dev, sample_weight=weights)
    champion = search.best_estimator_
    joblib.dump(champion, artefact_dir / "champion_xgboost_model.pkl")
    return {"model": champion, "search": search}


def score_cohort(model, preprocessor, df, target_rate, population_rate, features):
    """Transform and population-prior-correct a cohort."""
    prepared_features = preprocessor.transform(df[features])
    probabilities = model.predict_proba(prepared_features)[:, 1]
    scored_df = df.copy()
    scored_df["PPGD"] = restore_population_prior(probabilities, target_rate, population_rate)
    return scored_df


def main():
    """Run the model-build pipeline and save notebook-compatible artefacts."""
    root = Path(__file__).resolve().parents[1]
    train_full_df = pd.read_csv(
        root / "data/synthetic_portfolio/train_portfolio_2019_2024.csv",
        keep_default_na=False,
        na_values=[""],
    )
    oot_df = pd.read_csv(
        root / "data/synthetic_portfolio/oot_portfolio_2025_2026.csv",
        keep_default_na=False,
        na_values=[""],
    )
    with open(root / "data/model_artefacts/selected_features_metadata.json", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    features = metadata["final_14_features"]
    numerical_columns = metadata["numerical_features"]
    categorical_columns = metadata["categorical_features"]
    target = "possession_flag"
    dev_df, test_df = train_test_split(
        train_full_df, test_size=0.20, stratify=train_full_df[target], random_state=42
    )
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    preprocessor = build_preprocessor(numerical_columns, categorical_columns)
    prepared_dev = preprocessor.fit_transform(dev_df[features])
    prepared_test = preprocessor.transform(test_df[features])
    prepared_oot = preprocessor.transform(oot_df[features])
    y_dev = dev_df[target]
    observed_dev_rate = y_dev.mean()
    tuning = train_champion_xgboost(
        prepared_dev, y_dev, target_rate=0.20,
        artefact_dir=root / "data/model_artefacts",
    )
    champion = tuning["model"]

    keep_columns = ["account_id", "default_quarter", "default_year", "current_balance", "current_ltv", target]
    scored_cohorts = []
    for cohort_df, prepared_features in (
        (dev_df, prepared_dev),
        (test_df, prepared_test),
        (oot_df, prepared_oot),
    ):
        scored_df = cohort_df[keep_columns].copy()
        probabilities = champion.predict_proba(prepared_features)[:, 1]
        scored_df["PPGD"] = restore_population_prior(probabilities, 0.20, observed_dev_rate)
        scored_cohorts.append(scored_df)

    artefact_dir = root / "data/model_artefacts"
    joblib.dump(preprocessor, artefact_dir / "global_preprocessor.pkl")
    joblib.dump(champion, artefact_dir / "champion_xgboost_model.pkl")
    encoded_categorical_columns = list(
        preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(categorical_columns)
    )
    prepared_feature_names = numerical_columns + encoded_categorical_columns
    pd.DataFrame(prepared_dev, columns=prepared_feature_names).to_csv(artefact_dir / "X_dev_prep.csv", index=False)
    pd.DataFrame(prepared_test, columns=prepared_feature_names).to_csv(artefact_dir / "X_test_prep.csv", index=False)
    pd.DataFrame(prepared_oot, columns=prepared_feature_names).to_csv(artefact_dir / "X_oot_prep.csv", index=False)
    for cohort_name, scored_df in zip(("dev", "test", "oot"), scored_cohorts):
        scored_df.to_csv(artefact_dir / f"{cohort_name}_scored_predictions.csv", index=False)
    print("Saved champion XGBoost and scored cohort artefacts.")


if __name__ == "__main__":
    main()
