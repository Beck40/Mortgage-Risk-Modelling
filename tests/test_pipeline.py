import numpy as np
import pandas as pd

from src.data_generator import generate_portfolio
from src.data_preparation import apply_feature_filters
from src.ppgd_model import QuantileWinsorizer, restore_population_prior
from src.risk_grading import assign_risk_grades, assign_validation_quartiles, fit_kmeans_grader
from src.regulatory_validation import adjusted_gini, compute_grade_table, standard_gini


def test_generate_portfolio_creates_splits(tmp_path):
    portfolio = generate_portfolio(n_accounts=600, seed=42, output_dir=tmp_path)
    assert set(portfolio) == {
        "synthetic_loan_book",
        "train_portfolio_2019_2024",
        "test_portfolio_2019_2024",
        "oot_portfolio_2025_2026",
    }

    master_df = portfolio["synthetic_loan_book"]
    assert len(master_df) == 600
    assert {"account_id", "possession_flag", "default_quarter", "credit_score"} <= set(master_df)
    assert (tmp_path / "synthetic_loan_book.csv").exists()
    assert master_df["default_year"].max() >= 2025


def test_feature_filters_drop_traps_and_correlated_features():
    rng = np.random.default_rng(42)
    n_accounts = 300
    y = np.array([0, 1] * (n_accounts // 2))
    rng.shuffle(y)
    strong_feature = rng.normal(size=n_accounts)
    df = pd.DataFrame({
        "account_id": range(n_accounts),
        "possession_flag": y,
        "constant": 1,
        "missing": [np.nan] * 30 + list(rng.normal(size=n_accounts - 30)),
        "concentrated": ["A"] * 290 + ["B"] * 10,
        "strong": strong_feature,
        "correlated": strong_feature.copy(),
        "independent": rng.normal(size=n_accounts),
    })

    filtered_df, accepted_features, winsor_bounds, _, _ = apply_feature_filters(df)
    assert "constant" not in filtered_df
    assert "missing" not in filtered_df
    assert "concentrated" not in filtered_df
    assert winsor_bounds
    assert "strong" not in accepted_features or "correlated" not in accepted_features


def test_winsorizer_and_population_prior():
    training_values = np.array([[0.0], [1.0], [2.0], [100.0]])
    winsorizer = QuantileWinsorizer(0.25, 0.75).fit(training_values)
    transformed_values = winsorizer.transform(np.array([[-100.0], [50.0], [200.0]]))
    assert transformed_values.min() >= winsorizer.lower_bounds_[0]
    assert transformed_values.max() <= winsorizer.upper_bounds_[0]

    sampled_probabilities = np.array([0.1, 0.2, 0.5])
    restored_probabilities = restore_population_prior(
        sampled_probabilities, sampled_rate=0.20, population_rate=0.07
    )
    assert restored_probabilities.mean() >= 0.0
    assert restored_probabilities.mean() <= 1.0


def test_risk_grades_and_validation_quartiles():
    dev_df = pd.DataFrame({
        "PPGD": np.repeat([0.05, 0.15, 0.35, 0.80], 25),
        "possession_flag": np.repeat([0, 0, 1, 1], 25),
    })
    grader, cluster_map = fit_kmeans_grader(dev_df["PPGD"])
    graded_df = assign_risk_grades(dev_df, grader, cluster_map)
    grade_means = graded_df.groupby("Risk_Grade")["Grade_PPGD"].mean()
    assert grade_means.is_monotonic_increasing
    assert set(graded_df["Risk_Grade_Name"]) == {
        "Low Risk", "Moderate Risk", "High Risk", "Very High Risk"
    }

    cohorts = {
        "Development (2019–2024)": pd.DataFrame({"PPGD": np.linspace(0.01, 0.9, 40)}),
        "OOT (2025–2026)": pd.DataFrame({"PPGD": np.linspace(0.02, 0.8, 30)}),
    }
    quartiled_cohorts = assign_validation_quartiles(cohorts)
    assert quartiled_cohorts["OOT (2025–2026)"]["Validation_Grade"].between(1, 4).all()


def test_regulatory_metrics_and_hhi_bounds():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert -1.0 <= standard_gini(y_true, scores) <= 1.0
    assert -1.0 <= adjusted_gini(y_true, scores) <= 1.0

    rng = np.random.default_rng(42)
    grade_df = pd.DataFrame({
        "Risk_Grade": np.repeat([1, 2, 3, 4], 250),
        "Grade_PPGD": np.repeat([0.04, 0.08, 0.12, 0.16], 250),
    })
    event_rates = np.repeat([0.04, 0.08, 0.12, 0.16], 250)
    grade_df["possession_flag"] = rng.binomial(1, event_rates)
    grade_summary = compute_grade_table(grade_df)
    assert grade_summary["Normalised_HHI"].between(0.0, 1.0).all()
