"""Regulatory validation metrics for PPGD and risk grades."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from statsmodels.stats.proportion import proportions_ztest


def compute_grade_table(df, target_col="possession_flag", grade_col="Risk_Grade"):
    """Summarise grade volume, calibration, HHI and heterogeneity."""
    total_accounts = len(df)
    grade_count = df[grade_col].nunique()
    if "Grade_PPGD" not in df:
        grade_values = df.assign(Grade_PPGD=df.groupby(grade_col)[target_col].transform("mean"))
    else:
        grade_values = df

    grade_summary = grade_values.groupby(grade_col).agg(
        Volume=(target_col, "count"),
        Defaults=(target_col, "sum"),
        Observed_PPGD=(target_col, "mean"),
        Assigned_PPGD=("Grade_PPGD", "mean"),
    ).reset_index()
    grade_summary["Share"] = grade_summary["Volume"] / total_accounts
    grade_summary["A_E_Ratio"] = grade_summary["Observed_PPGD"] / grade_summary["Assigned_PPGD"]

    raw_hhi = float((grade_summary["Share"] ** 2).sum())
    grade_summary["Raw_HHI"] = raw_hhi
    grade_summary["Normalised_HHI"] = (raw_hhi - 1 / grade_count) / (1 - 1 / grade_count)

    p_values = [np.nan]
    for grade_index in range(1, len(grade_summary)):
        current_defaults = grade_summary.iloc[grade_index]["Defaults"]
        previous_defaults = grade_summary.iloc[grade_index - 1]["Defaults"]
        current_accounts = grade_summary.iloc[grade_index]["Volume"]
        previous_accounts = grade_summary.iloc[grade_index - 1]["Volume"]
        _, p_value = proportions_ztest(
            np.array([current_defaults, previous_defaults]),
            np.array([current_accounts, previous_accounts]),
            alternative="larger",
        )
        p_values.append(p_value) #type:ignore
    grade_summary["Heterogeneity_p_value"] = p_values
    return grade_summary


def standard_gini(y_true, scores):
    """Return 2 * AUC - 1."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(2 * roc_auc_score(y_true, scores) - 1)


def adjusted_gini(y_true, scores):
    """Return CAP-normalised adjusted Gini for tied or bucketed scores."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    standard = 2 * roc_auc_score(y_true, scores) - 1 if len(np.unique(y_true)) > 1 else 0.0

    if len(np.unique(scores)) > 20:
        binned_scores = pd.qcut(scores, q=20, labels=False, duplicates="drop")
    else:
        binned_scores = scores

    cap_df = pd.DataFrame({"target": y_true, "score": binned_scores})
    bucket_summary = cap_df.groupby("score", observed=True).agg(
        account_count=("target", "count"),
        default_count=("target", "sum"),
    ).sort_index(ascending=False)
    total_accounts = len(y_true)
    total_defaults = y_true.sum()
    if total_defaults <= 0:
        return float(standard)

    population_coordinates = np.insert(
        bucket_summary["account_count"].cumsum().to_numpy() / total_accounts, 0, 0.0
    )
    default_coordinates = np.insert(
        bucket_summary["default_count"].cumsum().to_numpy() / total_defaults, 0, 0.0
    )
    model_area = np.trapezoid(default_coordinates, population_coordinates) - 0.5

    default_first_sequence = []
    for n_accounts, n_defaults in zip(bucket_summary["account_count"], bucket_summary["default_count"]):
        default_first_sequence.extend([1] * int(n_defaults))
        default_first_sequence.extend([0] * int(n_accounts - n_defaults))

    perfect_population = np.linspace(0.0, 1.0, total_accounts + 1)
    perfect_defaults = np.insert(
        np.cumsum(np.asarray(default_first_sequence, dtype=float)) / total_defaults,
        0,
        0.0,
    )
    maximum_area = np.trapezoid(perfect_defaults, perfect_population) - 0.5
    return float(model_area / maximum_area if maximum_area > 0 else standard)


def evaluate_rank_ordering_4qma(df, target_col="possession_flag", grade_col="Risk_Grade"):
    """Calculate quarterly 4-quarter moving-average rates and inversions."""
    rates = df.groupby(["default_quarter", grade_col])[target_col].mean().unstack()
    moving_average = rates.rolling(window=4, min_periods=1).mean()
    inversions = int(sum(
        (np.diff(row.dropna().to_numpy()) <= 0).sum()
        for _, row in moving_average.iterrows()
        if row.dropna().size > 1
    ))
    return {"quarterly_4qma": moving_average, "inversion_count": inversions}


def compute_accuracy_metrics(y_true, scores):
    """Return MAE, RMSE and Brier score."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    return {
        "MAE": float(mean_absolute_error(y_true, scores)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, scores))),
        "Brier": float(brier_score_loss(y_true, scores)),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    artefact_dir = root / "data/model_artefacts"
    dev_df = pd.read_csv(artefact_dir / "dev_scored_predictions.csv")
    oot_df = pd.read_csv(artefact_dir / "oot_scored_predictions.csv")
    for cohort_name, cohort_df in (("Development", dev_df), ("OOT", oot_df)):
        gini = standard_gini(cohort_df["possession_flag"], cohort_df["PPGD"])
        accuracy = compute_accuracy_metrics(cohort_df["possession_flag"], cohort_df["PPGD"])
        print(f"{cohort_name}: Gini={gini:.4f}, MAE={accuracy['MAE']:.4f}")
