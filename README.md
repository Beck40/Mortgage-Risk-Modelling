## Challenger Modelling for Mortgage Possession Risk (PPGD)

This repository summarises my Level 7 Artificial Intelligence (AI) apprenticeship (standard [ST0763](https://skillsengland.education.gov.uk/apprenticeship-standards/st0763-v1-0)), delivered by Cambridge Spark while I worked in the quantitative modelling team. It describes the assessments, the project they were based on, and the professional evidence behind them.

> **Confidentiality note**
> This repository does not contain proprietary employer data, internal code, documents, or live model results. To demonstrate the end-to-end engineering and validation framework publicly, it includes a bespoke Monte Carlo data generator calibrated to public UK macroeconomic series and aligned with published regulatory standards, including Basel III/IV and PRA SS1/23.

---

## Assessment summary

| Assessment | Format | Outcome |
|---|---|---|
| **Project report and presentation** | Approximately 5,000-word technical report, 30-minute presentation, 45-minute technical Q&A | **Distinction** |
| **Professional discussion** | 90-minute structured discussion supported by a portfolio of evidence | **Distinction** |
| **Technical knowledge test** | 100-minute closed-book exam, four long-response scenario questions | **Distinction** |

---

## 1. Commercial project background

### Regulatory context

Banks that model retail mortgage losses under the Internal Ratings-Based (IRB) and IFRS 9 frameworks estimate Loss Given Default (LGD). A primary component of mortgage LGD is the **Probability of Possession Given Default (PPGD)**: once a loan enters default, how likely is it that the property will ultimately be repossessed? PPGD feeds directly into LGD, which combines with Probability of Default (PD) and Exposure at Default (EAD) to determine regulatory capital requirements and impairment provisions. Under UK **PRA SS1/23** Model Risk Management principles, firms are expected to develop and benchmark challenger models alongside incumbent production models.

### Scope and objectives

A decision tree that creates account risk grades using a limited feature set can lead to oversimplified risk assessments. I developed a machine-learning challenger framework covering accounts in default and pre-default cohorts. The objective was to capture non-linear risk interactions across a broader feature space, improve calibration accuracy, and establish a method for migrating model development from SAS to Python.

### Trade-offs and real-world governance considerations

- **Explainability:** A gradient-boosted ensemble requires mathematical justification and post-hoc explainability using SHAP to satisfy internal validation and regulatory scrutiny.
- **Productionisation and compute:** Transitioning large-scale data pipelines from legacy SAS servers to local or cloud Python environments requires careful memory management and modular preprocessing.
- **External data governance:** Macroeconomic and house-price-index inputs sit outside internal system controls and require stability and boundary checks before ingestion.

### Professional discussion: broader impact

- **Knowledge transfer:** Established Python adoption infrastructure, including technical channels, shared repositories, and training materials, to support a 60-person quantitative analytics function.
- **Quality assurance:** Peer-reviewed codebase updates for internal analysts and external consultants, and validated SAS-to-Python logic translations.
- **Stakeholder management:** Translated statistical outputs into actionable governance packs for Finance, Internal Audit, and Legal teams.
- **Model monitoring:** Designed a quarterly Red-Amber-Green (RAG) performance monitoring framework for senior risk committees.

---

## 2. Open-source Python replication architecture

The public replication provides a self-contained pipeline covering synthetic portfolio generation, regulatory feature selection, model tuning, K-Means Master Rating Scale grading, and multi-period validation.

```text
Mortgage-Risk-Modelling/
├── assets/
│   └── charts/                                    # Exported RAG validation and SHAP visuals
├── data/
│   ├── model_artefacts/                           # Serialised models, metadata and scored CSVs
│   ├── raw_macro/                                 # Public UK macroeconomic series
│   ├── synthetic_portfolio/                       # Generated train, test and OOT cohorts
│   └── README.md                                  # Public macro-data sourcing instructions
├── notebooks/
│   ├── 01_macro_data_exploration.ipynb            # Macro regime analysis and portfolio generation
│   ├── 02_feature_selection_and_ranking.ipynb     # Sequential regulatory feature filter
│   ├── 03_model_build.ipynb                       # Leak-free model build and K-Means grading
│   └── 04_model_evaluation.ipynb                  # RAG validation and SHAP explainability
├── Project-Planning/
│   └── ML Project Plan.pdf                        # Architecture and project scoping
├── src/
│   ├── 01_synthetic_data_generator.py             # Monte Carlo portfolio generator
│   ├── 02_data_preparation.py                     # Data preparation utilities
│   ├── 03_xgboost_ppgd_model.py                   # PPGD modelling utilities
│   ├── 04_kmeans_risk_grading.py                  # Risk-grade assignment utilities
│   └── 05_regulatory_validation.py                # Validation utilities
└── README.md
```

### End-to-end pipeline walkthrough

1. **Macroeconomic and portfolio generation (`01_macro_data_exploration.ipynb`)**

   The pipeline combines public UK macroeconomic series with a Monte Carlo simulation in [`src/01_synthetic_data_generator.py`](src/01_synthetic_data_generator.py) to generate a master book of 100,000 accounts. The modelling population contains 84,737 defaulted residential mortgage exposures across the candidate variables, with an approximately 7% possession rate.

   To prevent data leakage and test temporal resilience, the population is split into:

   - **Development set (80% in-time, 2019–2024):** 48,838 accounts; observed PPGD 6.80%.
   - **In-time test set (20% holdout, 2019–2024):** 12,210 accounts; observed PPGD 6.81%.
   - **Out-of-time (OOT) holdout (2025–2026):** 23,689 accounts; observed PPGD 7.36%.

2. **Sequential regulatory feature selection (`02_feature_selection_and_ranking.ipynb`)**

   A transparent six-stage funnel is used instead of a black-box recursive wrapper:

   - **Stage 1:** Validity checks.
   - **Stage 2:** Completeness, with a 5% missingness threshold.
   - **Stage 3:** Categorical concentration, with a 95% single-level cap.
   - **Stage 4:** Winsorisation at the 1st and 99th percentiles.
   - **Stage 5:** Univariate Weight of Evidence (WoE) and group-weighted AUC ranking.
   - **Stage 6:** Spearman rank intercorrelation filtering at (|rho| > 0.70).

   This reduces the candidate variables to 14 orthogonal credit-risk drivers.

3. **Model build and Master Rating Scale segmentation (`03_model_build.ipynb`)**

   - Fits a `ColumnTransformer` using median imputation, learned quantile winsorisation, standard scaling, and one-hot encoding strictly on the Development set.
   - Tunes Logistic Regression, Random Forest, LightGBM, and Champion XGBoost using stratified cross-validation.
   - Applies training-only class reweighting to improve minority-event learning while preserving the observed population prevalence during scoring.
   - Applies analytical prior correction before evaluation and grading.
   - Fits one-dimensional K-Means clustering (k=4) to calibrated Development PPGDs.
   - Maps the continuous scores into four named risk grades:
     - Grade 1: **Low Risk**
     - Grade 2: **Moderate Risk**
     - Grade 3: **High Risk**
     - Grade 4: **Very High Risk**

4. **Regulatory validation and explainability (`04_model_evaluation.ipynb`)**

   The Champion XGBoost model and K-Means rating system are evaluated across Development, In-Time Test, and OOT cohorts. The notebook includes rank ordering, concentration, heterogeneity, discrimination, accuracy, classification diagnostics, matched-sample Welch tests, and global/local SHAP explainability.

---

## 3. Key methodological highlights

### A. Minority-class reweighting and analytical Bayes prior restoration

Possession is a minority event in the defaulted mortgage population, with a 6.80% Development base rate. Training tree ensembles on raw imbalanced priors can under-allocate splits to high-risk tail interactions.

During training, instance weights target an effective 20.00% minority event rate. Because reweighting changes the model's predicted log-odds, raw outputs are not used directly for capital or provisioning calculations. Before risk grading or evaluation, predicted probabilities are analytically recalibrated to the observed Development prior:

$$
\text{Odds}_{\text{pop}} =
\frac{\hat{p}_{\text{weighted}}}{1-\hat{p}_{\text{weighted}}}
\times
\frac{\pi_{\text{pop}}/(1-\pi_{\text{pop}})}
{\pi_{\text{train}}/(1-\pi_{\text{train}})}
$$

$$
\text{PPGD}_{\text{calibrated}} =
\frac{\text{Odds}_{\text{pop}}}{1+\text{Odds}_{\text{pop}}}
$$

This preserves minority-feature learning while maintaining portfolio-level calibration.

### B. CAP-adjusted Gini for tied grade distributions

The standard Gini coefficient is:

$$
\text{Standard Gini} = 2 \times (\text{AUC} - 0.5)
$$

Mapping continuous probabilities into four grades creates tied scores. The evaluation therefore reports a CAP-adjusted Gini using the maximum CAP area achievable while preserving the observed grade bucket sizes. Within each grade, the benchmark places defaults before cures; it does not reorder grades by their realised default rates.

$$
\text{Adjusted Gini} =
\frac{\text{Model CAP Area}}
{\text{Maximum Achievable CAP Area}}
$$

This separates the discriminatory power achieved by the discrete scale from the loss caused solely by ties within grades.

---

## 4. Regulatory validation and RAG governance visuals

The exported charts are available in [`assets/charts/`](assets/charts/), and the underlying calculations are in [`notebooks/04_model_evaluation.ipynb`](notebooks/04_model_evaluation.ipynb).

### 4-quarter moving-average rank ordering

Grades should remain distinct and strictly monotonic over time:

**Grade 1 < Grade 2 < Grade 3 < Grade 4**

A four-quarter moving average reduces seasonal noise. The notebook checks quarterly crossovers and compressed adjacent-grade differences, with the result displayed using the rank-ordering RAG chart.

### Portfolio concentration and risk heterogeneity

The Herfindahl-Hirschman Index (HHI) is:

$$
\text{HHI} = \sum_i s_i^2
$$

where \(s_i\) is the share of accounts in grade \(i\). With four grades, an equal split has a theoretical minimum HHI of 0.25. The current evaluation RAG bands are:

- **Green:** HHI ≤ 0.38
- **Amber:** 0.38 < HHI ≤ 0.52
- **Red:** HHI > 0.52

K-Means identifies natural score boundaries rather than forcing equal grade volumes. Consequently, right-skewed mortgage portfolios can concentrate curing accounts in Grade 1. The Welch tests use equal-sized matched samples of up to 200 accounts per grade pair and report grade names, mean PPGD differences, effect sizes, and a green/red significance flag.

### Discrimination stability and Accuracy

The notebook tracks quarterly continuous Gini, four-grade standard Gini, and four-grade CAP-adjusted Gini across all cohorts. The discrimination RAG bands are:

- **Green:** Gini ≥ 0.60
- **Amber:** 0.50 ≤ Gini < 0.60
- **Red:** Gini < 0.50

Quarterly MAE is also charted for Development, In-Time Test, and OOT:

- **Green:** MAE ≤ 10%
- **Amber:** 10% < MAE ≤ 20%
- **Red:** MAE > 20%

The overall table additionally reports KS statistic, RMSE, and Brier score. The time-series charts are intended to show whether performance remains stable rather than relying only on a single point estimate.

### Model explainability

To support PRA SS1/23 transparency expectations, `shap.TreeExplainer` is used to audit:

- global portfolio drivers using the OOT population;
- local account-level risk attributions for high-risk Grade 4 exposures.

---

## 5. How to reproduce

1. **Clone the repository**

   ```bash
   git clone https://github.com/Beck40/Mortgage-Risk-Modelling.git
   cd Mortgage-Risk-Modelling
   ```

2. **Install the Python dependencies**

   ```bash
   pip install -r requirement.txt
   ```

3. **Acquire the public macroeconomic datasets**

   Follow [`data/README.md`](data/README.md) to source the required public datasets into `data/raw_macro/`.

4. **Generate the synthetic portfolio**

   ```bash
   python src/01_synthetic_data_generator.py
   ```

5. **Run the notebooks in order**

   Execute notebooks 01 through 04. The generated model artefacts and scored cohorts are written under `data/model_artefacts/`; generated portfolio CSVs are written under `data/synthetic_portfolio/`.

---

## References

- [PRA SS1/23: Model risk management principles for banks](https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/supervisory-statement/2023/ss123.pdf)
- [Basel III: international regulatory framework for banks](https://www.bis.org/bcbs/basel3.htm)
- [Cambridge Spark apprenticeship standard ST0763](https://skillsengland.education.gov.uk/apprenticeship-standards/st0763-v1-0)
