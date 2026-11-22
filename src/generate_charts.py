"""Generate validation charts from the scored XGBoost artefacts."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, roc_auc_score

from src.risk_grading import assign_validation_quartiles


ROOT = Path(__file__).resolve().parents[1]
ARTEFACT_DIR = ROOT / "data" / "model_artefacts"
CHART_DIR = ROOT / "assets" / "charts"
NAVY = "#1F4E78"
GREEN = "#D9F2D9"
AMBER = "#FFF2CC"
RED = "#FFCCCC"
RISK_GRADE_NAMES = {
    1: "Low Risk",
    2: "Developing Risk",
    3: "Elevated Risk",
    4: "Critical Risk",
}


def load_cohorts() -> dict[str, pd.DataFrame]:
    cohorts = {
        "Development (2019–2024)": "dev_scored_predictions.csv",
        "OOT Holdout (2025–2026)": "oot_scored_predictions.csv",
    }
    loaded = {label: pd.read_csv(ARTEFACT_DIR / filename) for label, filename in cohorts.items()}
    return assign_validation_quartiles(loaded)


def save_chart(fig: plt.Figure, filename: str) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)


def observation_year_ticks(quarters: list[str]) -> tuple[list[int], list[str]]:
    years = [str(quarter)[:4] for quarter in quarters]
    positions = [index for index, year in enumerate(years) if index == 0 or year != years[index - 1]]
    return positions, [years[position] for position in positions]


def add_hhi_bands(ax: plt.Axes) -> None:
    ax.axhspan(0.00, 0.10, color=GREEN, alpha=0.7)
    ax.axhspan(0.10, 0.25, color=AMBER, alpha=0.7)
    ax.axhspan(0.25, 0.75, color=RED, alpha=0.7)
    ax.axhline(0.10, color="#B8860B", linestyle="--", linewidth=1.1)
    ax.axhline(0.25, color="#A94442", linestyle="--", linewidth=1.1)


def add_gini_bands(ax: plt.Axes) -> None:
    ax.axhspan(0.00, 0.50, color=RED, alpha=0.7)
    ax.axhspan(0.50, 0.65, color=AMBER, alpha=0.7)
    ax.axhspan(0.65, 1.05, color=GREEN, alpha=0.7)
    ax.axhline(0.50, color="#A94442", linestyle="--", linewidth=1.1)
    ax.axhline(0.65, color="#B8860B", linestyle="--", linewidth=1.1)


def add_mae_bands(ax: plt.Axes) -> None:
    ax.axhspan(0.00, 0.10, color=GREEN, alpha=0.7)
    ax.axhspan(0.10, 0.25, color=AMBER, alpha=0.7)
    ax.axhspan(0.25, 0.35, color=RED, alpha=0.7)
    ax.axhline(0.10, color="#B8860B", linestyle="--", linewidth=1.1)
    ax.axhline(0.25, color="#A94442", linestyle="--", linewidth=1.1)


def plot_rank_ordering(cohorts: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    maximum_rate = 0.0
    moving_averages = {}
    for label, cohort_df in cohorts.items():
        rates = cohort_df.groupby(["default_quarter", "Validation_Grade"])["possession_flag"].mean().unstack()
        moving_averages[label] = rates.rolling(window=4, min_periods=1).mean() * 100
        maximum_rate = max(maximum_rate, float(moving_averages[label].max().max()))
    y_upper = max(10.0, float(np.ceil(maximum_rate * 1.10 / 5.0) * 5.0))
    for ax, (label, cohort_df) in zip(axes, cohorts.items()):
        moving_average = moving_averages[label]
        quarters = moving_average.index.tolist()
        grade_colours = ["#1B7837", "#FF6701", "#FDAE61", "#D73027"]
        for grade in sorted(moving_average.columns):
            ax.plot(
                np.arange(len(moving_average)),
                moving_average[grade],
                marker="o",
                markersize=4,
                linewidth=2,
                color=grade_colours[int(grade) - 1],
                label=RISK_GRADE_NAMES[int(grade)],
            )
        inversions = int(sum(
            (np.diff(row.dropna().to_numpy()) <= 0).sum()
            for _, row in moving_average.iterrows()
            if row.dropna().size > 1
        ))
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Observation year")
        positions, tick_labels = observation_year_ticks(quarters)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylim(0, y_upper)
        style_axis(ax)
    axes[0].set_ylabel("Possession rate (%)")
    rank_handles, rank_labels = axes[0].get_legend_handles_labels()
    rank_labels = [RISK_GRADE_NAMES[int(grade)] for grade in sorted(moving_average.columns)]
    fig.legend(rank_handles, rank_labels, frameon=False, fontsize=8,
               loc="center left", bbox_to_anchor=(0.84, 0.5))
    fig.suptitle("4-Quarter Moving Average Rank Ordering",
                 x=0.41, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.82, 0.94))
    save_chart(fig, "01_rank_ordering_rag.png")


def plot_hhi(cohorts: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (label, frame) in zip(axes, cohorts.items()):
        raw_values = frame.groupby("default_quarter")["Validation_Grade"].apply(
            lambda series: (series.value_counts(normalize=True) ** 2).sum()
        )
        values = (raw_values - 0.25) / 0.75
        quarters = values.index.tolist()
        ax.plot(np.arange(len(values)), values.to_numpy(), marker="s",
                color=NAVY, linewidth=2, label="Quarterly HHI*")
        add_hhi_bands(ax)
        ax.set_title(label, fontweight="bold")
        positions, labels = observation_year_ticks(quarters)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0.00, 0.75)
        ax.set_xlabel("Observation year")
        style_axis(ax)
    axes[0].set_ylabel("Herfindahl-Hirschman Index")
    axes[1].legend(
        handles=[
            Patch(facecolor=GREEN, label="Green: HHI < 0.10"),
            Patch(facecolor=AMBER, label="Amber: 0.10 < HHI < 0.25"),
            Patch(facecolor=RED, label="Red: HHI > 0.25"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
    )
    fig.suptitle("HHI (Herfindahl-Hirschman Index)",
                 x=0.41, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.82, 0.94))
    save_chart(fig, "02_hhi_concentration_rag.png")


def quarterly_gini(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quarter, group in frame.groupby("default_quarter", sort=True):
        if group["possession_flag"].nunique() < 2:
            continue
        rows.append({
            "quarter": quarter,
            "continuous": 2 * roc_auc_score(group["possession_flag"], group["PPGD"]) - 1,
        })
    return pd.DataFrame(rows).set_index("quarter")


def plot_discrimination(cohorts: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (label, frame) in zip(axes, cohorts.items()):
        values = quarterly_gini(frame)
        x = np.arange(len(values))
        add_gini_bands(ax)
        ax.plot(x, values["continuous"], marker="o", color=NAVY,
                linewidth=2, label="Continuous standard Gini")
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0.00, 1.05)
        ax.set_xlabel("Observation year")
        positions, tick_labels = observation_year_ticks(values.index.tolist())
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        style_axis(ax)
    axes[0].set_ylabel("Gini coefficient")
    axes[1].legend(
        handles=[
            Patch(facecolor=GREEN, label="Green: Gini > 0.65"),
            Patch(facecolor=AMBER, label="Amber: 0.50 < Gini < 0.65"),
            Patch(facecolor=RED, label="Red: Gini < 0.50"),
        ],
        labels=[
            "Green: Gini > 0.65",
            "Amber: 0.50 < Gini < 0.65",
            "Red: Gini < 0.50",
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    fig.suptitle("GINI",
                 x=0.41, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.82, 0.94))
    save_chart(fig, "03_discrimination_stability_rag.png")


def plot_mae(cohorts: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (label, frame) in zip(axes, cohorts.items()):
        values = frame.groupby("default_quarter", sort=True).apply(
            lambda group: mean_absolute_error(group["possession_flag"], group["PPGD"]),
            include_groups=False,
        ) #type:ignore
        quarters = values.index.tolist()
        add_mae_bands(ax)
        ax.plot(np.arange(len(values)), values.to_numpy(), marker="o",
                color=NAVY, linewidth=2, label="Quarterly MAE")
        ax.set_title(f"{label}\nOverall MAE: {mean_absolute_error(frame['possession_flag'], frame['PPGD']):.3f}",
                     fontweight="bold")
        positions, labels = observation_year_ticks(quarters)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0.00, 0.35)
        ax.set_xlabel("Observation year")
        style_axis(ax)
    axes[0].set_ylabel("Mean absolute error")
    axes[1].legend(
        handles=[
            Patch(facecolor=GREEN, label="Green: MAE < 0.10"),
            Patch(facecolor=AMBER, label="Amber: 0.10 < MAE < 0.25"),
            Patch(facecolor=RED, label="Red: MAE > 0.25"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
    )
    fig.suptitle("MAE",
                 x=0.41, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.82, 0.94))
    save_chart(fig, "04_mae_accuracy_rag.png")


def clean_feature_name(name: str) -> str:
    return name.replace("_", " ").title().replace("Ltv", "LTV").replace("Boe", "BoE")


def plot_shap() -> None:
    model = joblib.load(ARTEFACT_DIR / "champion_xgboost_model.pkl")
    features = pd.read_csv(ARTEFACT_DIR / "X_oot_prep.csv")
    oot = pd.read_csv(ARTEFACT_DIR / "oot_scored_predictions.csv")
    feature_names = [clean_feature_name(name) for name in features.columns]
    explainer = shap.TreeExplainer(model)
    sample = features.sample(n=min(1500, len(features)), random_state=42)
    sample_for_plot = sample.copy()
    sample_for_plot.columns = feature_names
    shap_values = explainer.shap_values(sample_for_plot)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot.barh(ax=ax, color=NAVY)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title("XGBoost Global SHAP Feature Importance", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_chart(fig, "05a_SHAP_global.png")

    high_risk_index = int(oot["PPGD"].idxmax())
    local_row = features.iloc[[high_risk_index]].copy()
    local_row.columns = feature_names
    local_values = explainer.shap_values(local_row)
    if isinstance(local_values, list):
        local_values = local_values[1]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    explanation = shap.Explanation(
        values=local_values[0],
        base_values=base_value,
        data=local_row.iloc[0].to_numpy(),
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(
        f"XGBoost Local SHAP Explanation — {oot.loc[high_risk_index, 'account_id']} "
        f"({oot.loc[high_risk_index, 'PPGD']:.2%} PPGD)",
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(CHART_DIR / "05b_SHAP_local.png", dpi=300, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    cohorts = load_cohorts()
    plot_rank_ordering(cohorts)
    plot_hhi(cohorts)
    plot_discrimination(cohorts)
    plot_mae(cohorts)
    plot_shap()
    print(f"Saved data-driven charts to {CHART_DIR}")


if __name__ == "__main__":
    main()
