"""Four-grade K-Means risk grading and validation quartile assignment."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans_grader(dev_ppgd, n_grades=4, random_state=42):
    """Fit K-Means and return it with monotonically ordered grade labels."""
    ppgd = np.asarray(dev_ppgd, dtype=float).reshape(-1, 1)
    grader = KMeans(n_clusters=n_grades, random_state=random_state, n_init=20)
    grader.fit(ppgd)
    sorted_clusters = np.argsort(grader.cluster_centers_.flatten())
    cluster_map = {
        int(cluster): int(grade + 1)
        for grade, cluster in enumerate(sorted_clusters)
    }
    return grader, cluster_map


def assign_risk_grades(df, grader, cluster_map, grade_pooled_map=None):
    """Assign grade names and empirical Development PPGD rates."""
    scored_df = df.copy()
    ppgd = np.asarray(scored_df["PPGD"], dtype=float).reshape(-1, 1)
    clusters = grader.predict(ppgd)
    scored_df["Risk_Grade"] = pd.Series(clusters).map(cluster_map).to_numpy()
    scored_df["Risk_Grade_Name"] = scored_df["Risk_Grade"].map({
        1: "Low Risk",
        2: "Moderate Risk",
        3: "High Risk",
        4: "Very High Risk",
    })
    if grade_pooled_map is None:
        grade_pooled_map = scored_df.groupby("Risk_Grade")["possession_flag"].mean().to_dict()
    scored_df["Grade_PPGD"] = scored_df["Risk_Grade"].map(grade_pooled_map)
    return scored_df


def assign_validation_quartiles(cohorts):
    """Apply Development-fitted PPGD quartiles to validation cohorts."""
    development_key = next(key for key in cohorts if "Development" in key)
    cut_points = cohorts[development_key]["PPGD"].quantile([0.25, 0.50, 0.75]).to_numpy()
    grade_names = {
        1: "Low Risk",
        2: "Developing Risk",
        3: "Elevated Risk",
        4: "Critical Risk",
    }
    quartiled_cohorts = {}
    for cohort_name, cohort_df in cohorts.items():
        quartiled_df = cohort_df.copy()
        quartiled_df["Validation_Grade"] = np.digitize(quartiled_df["PPGD"], cut_points) + 1
        quartiled_df["Validation_Grade_Name"] = quartiled_df["Validation_Grade"].map(grade_names)
        quartiled_cohorts[cohort_name] = quartiled_df
    return quartiled_cohorts


def main():
    """Fit the K-Means grader and grade all scored cohorts."""
    root = Path(__file__).resolve().parents[1]
    artefact_dir = root / "data/model_artefacts"
    dev_df = pd.read_csv(artefact_dir / "dev_scored_predictions.csv")
    test_df = pd.read_csv(artefact_dir / "test_scored_predictions.csv")
    oot_df = pd.read_csv(artefact_dir / "oot_scored_predictions.csv")

    grader, cluster_map = fit_kmeans_grader(dev_df["PPGD"])
    dev_ppgd = np.asarray(dev_df["PPGD"], dtype=float).reshape(-1, 1)
    dev_clusters = grader.predict(dev_ppdg)
    dev_grades = pd.Series(dev_clusters).map(cluster_map)
    grade_pooled_map = dev_df.assign(Risk_Grade=dev_grades).groupby("Risk_Grade")["possession_flag"].mean().to_dict()

    dev_df = assign_risk_grades(dev_df, grader, cluster_map, grade_pooled_map)
    test_df = assign_risk_grades(test_df, grader, cluster_map, grade_pooled_map)
    oot_df = assign_risk_grades(oot_df, grader, cluster_map, grade_pooled_map)

    joblib.dump(grader, artefact_dir / "kmeans_risk_grader.pkl")
    dev_df.to_csv(artefact_dir / "dev_scored_predictions.csv", index=False)
    test_df.to_csv(artefact_dir / "test_scored_predictions.csv", index=False)
    oot_df.to_csv(artefact_dir / "oot_scored_predictions.csv", index=False)
    print("Saved K-Means risk grader and regraded scored cohorts.")


if __name__ == "__main__":
    main()
