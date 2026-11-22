"""Sequential feature selection for the PPGD challenger model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_woe_and_gauc(series, y, is_cat=False, n_bins=10):
    """Return Weight of Evidence, gAUC and Information Value."""
    temp = pd.DataFrame({"x": series, "y": y})
    if not is_cat and temp["x"].nunique() > n_bins:
        temp["bin"] = pd.qcut(temp["x"], q=n_bins, duplicates="drop").astype(str)
    else:
        temp["bin"] = temp["x"].astype(str)

    stats = temp.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    stats["non_event"] = stats["count"] - stats["sum"]
    event_distribution = np.maximum(stats["sum"] / stats["sum"].sum(), 1e-6)
    non_event_distribution = np.maximum(stats["non_event"] / stats["non_event"].sum(), 1e-6)
    stats["woe"] = np.log(event_distribution / non_event_distribution)
    stats["iv"] = (event_distribution - non_event_distribution) * stats["woe"]

    woe_feature = temp["bin"].map(stats["woe"]).astype(float)
    auc = roc_auc_score(y, woe_feature)
    gauc = max(auc, 1 - auc)
    return woe_feature, float(gauc), float(stats["iv"].sum())


def apply_feature_filters(df, target="possession_flag", id_cols=None):
    """Apply validity, completeness and concentration filters, then winsorise numerical features."""
    id_cols = id_cols or ["account_id", "default_date", "default_quarter", "default_year"]
    candidate_features = [col for col in df.columns if col not in id_cols + [target]]
    categorical_features = df[candidate_features].select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = df[candidate_features].select_dtypes(exclude=["object", "category"]).columns.tolist()

    valid_features = [col for col in candidate_features if df[col].nunique(dropna=True) > 1]
    complete_features = [col for col in valid_features if df[col].isnull().mean() <= 0.05]

    unconcentrated_features = []
    for col in complete_features:
        if col in categorical_features:
            dominant_share = df[col].value_counts(normalize=True, dropna=True).iloc[0]
            if dominant_share <= 0.95:
                unconcentrated_features.append(col)
        else:
            unconcentrated_features.append(col)

    filtered_df = df[unconcentrated_features + [target]].copy()
    winsor_bounds = {}
    for col in unconcentrated_features:
        if col in numerical_features:
            lower, upper = filtered_df[col].quantile([0.01, 0.99])
            winsor_bounds[col] = (float(lower), float(upper))
            filtered_df[col] = filtered_df[col].clip(lower=lower, upper=upper)

    ranking_records = []
    woe_df = pd.DataFrame(index=filtered_df.index)
    for col in unconcentrated_features:
        woe_feature, gauc, iv = compute_woe_and_gauc(
            filtered_df[col], filtered_df[target], is_cat=col in categorical_features
        )
        woe_df[f"WoE_{col}"] = woe_feature
        ranking_records.append({
            "Feature": col,
            "Type": "Categorical" if col in categorical_features else "Numerical",
            "gAUC": round(gauc, 4),
            "IV": round(iv, 4),
        })

    ranking_df = pd.DataFrame(ranking_records).sort_values("gAUC", ascending=False).reset_index(drop=True)
    ordered_woe_columns = [f"WoE_{feature}" for feature in ranking_df["Feature"]]
    spearman_matrix = woe_df[ordered_woe_columns].corr(method="spearman").abs()

    accepted_features = []
    for feature in ranking_df["Feature"]:
        woe_column = f"WoE_{feature}"
        correlated = any(
            float(spearman_matrix.loc[woe_column, f"WoE_{kept_feature}"]) > 0.70
            for kept_feature in accepted_features
        )
        if not correlated:
            accepted_features.append(feature)

    return filtered_df, accepted_features, winsor_bounds, numerical_features, ranking_df


def select_features(df, target="possession_flag", final_count=14, id_cols=None):
    """Run sequential feature selection and return selected features and metadata."""
    _, accepted_features, winsor_bounds, numerical_features, _ = apply_feature_filters(
        df, target=target, id_cols=id_cols
    )
    selected_features = accepted_features[:final_count]
    categorical_features = [
        col for col in df.columns
        if col in selected_features and col not in numerical_features
    ]
    metadata = {
        "final_14_features": selected_features,
        "numerical_features": [col for col in selected_features if col in numerical_features],
        "categorical_features": categorical_features,
        "winsor_bounds": {col: winsor_bounds[col] for col in selected_features if col in winsor_bounds},
    }
    return selected_features, metadata


def main(input_path="data/synthetic_portfolio/train_portfolio_2019_2024.csv",
         artefact_dir="data/model_artefacts"):
    """Run feature selection and write selected-feature metadata."""
    root = Path(__file__).resolve().parents[1]
    input_path = root / input_path if not Path(input_path).is_absolute() else Path(input_path)
    artefact_dir = root / artefact_dir if not Path(artefact_dir).is_absolute() else Path(artefact_dir)
    df = pd.read_csv(input_path, keep_default_na=False, na_values=[""])

    for col in ("legacy_affordability_score", "secondary_employer_tenure_m"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _, accepted_features, winsor_bounds, numerical_features, ranking_df = apply_feature_filters(df)
    final_features = accepted_features[:14]
    categorical_features = [
        col for col in df.columns
        if col in final_features
        and col not in numerical_features
    ]
    metadata = {
        "final_14_features": final_features,
        "numerical_features": [col for col in final_features if col in numerical_features],
        "categorical_features": categorical_features,
        "winsor_bounds": {col: winsor_bounds[col] for col in final_features if col in winsor_bounds},
    }

    artefact_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = artefact_dir / "selected_features_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    print(ranking_df.head(20).to_string(index=False))
    print(f"Saved {len(final_features)} features to {metadata_path}")
    return metadata


if __name__ == "__main__":
    main()
