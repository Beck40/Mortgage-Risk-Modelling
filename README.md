# Artificial Intelligence Apprenticeship Portfolio

## Challenger Modelling for Mortgage Possession Risk (PPGD)

This repository summarises my Level 7 Artificial Intelligence (AI) apprenticeship (standard [ST0763](https://skillsengland.education.gov.uk/apprenticeship-standards/st0763-v1-0)), delivered by Cambridge Spark while I worked in the quantitative modelling team. It describes the assessments, the project they were based on, and the professional evidence behind them.

> **Confidentiality note**
> This repository does not contain employer data, code, documents or model results. The project report covered sensitive data and methodology, so this is a summary only. It draws on published guidance (Basel III/IV, PRA SS1/23) and standard machine learning practice.

---

## Assessment summary

| Assessment | Format | Outcome |
|---|---|---|
| Project report and presentation | ~5,000-word report, 30-minute presentation, 45-minute technical Q&A | Distinction |
| Professional discussion | 90-minute structured discussion supported by a portfolio of evidence | Distinction |
| Technical knowledge test | 100-minute closed-book exam, four long-response scenario questions | Distinction |

---

## The project

### Background

Banks that model mortgage losses under the internal ratings-based approach estimate Loss Given Default (LGD). One component is the Probability of Possession given Default (PPGD): of the loans that default, how likely is it that the property is repossessed? PPGD feeds into LGD, which combines with Probability of Default and Exposure at Default to estimate capital requirements for regulatory reporting. Regulators expect firms to consider alternative (challenger) models alongside the model in use.

### Scope

The existing approach was a decision tree that grouped accounts using a single feature. I developed a challenger model, with two separate PPGD models covering accounts not in default and accounts in default. A secondary aim was to build a framework that would support moving model development from SAS to Python.

### Approach

1. **Data preparation.** I combined mortgage transaction, property possession and collections/recoveries data, working with data engineers to source it. I engineered features before selection so they could be compared fairly with the raw ones. Data quality checks covered outliers and uninformative values for numerical features, and low-volume classes for categorical features. Missing values were kept as their own category rather than imputed, because the datasets were large enough not to need it.
2. **Feature selection.** I used a filter method (validity, completeness, ranking by group-weighted AUC and intercorrelation), chosen because it is fast, transparent and less prone to overfitting than automated selection. The challenger used 14 features, compared with the single feature in the existing tree.
3. **Model development.** XGBoost estimated each account's probability of possession. K-Means then grouped those probabilities into risk grades (for example low, medium and high). Hyperparameters were tuned with randomised search and cross-validation, with class imbalance handled explicitly because possession is rare.
4. **Python framework.** The code included a connection to SAS so datasets could move between the existing environment and Python.

### How the grades were evaluated

In priority order:

1. **Rank ordering:** grades should stay distinct and keep their order over time, assessed on a 4-quarter moving average to reduce quarterly noise.
2. **Concentration:** accounts should be spread sensibly across grades (Herfindahl-Hirschman Index).
3. **Heterogeneity:** grades should carry meaningfully different levels of risk.
4. **Discrimination:** Gini / ROC AUC on test sets, overall and over time.
5. **Accuracy:** MAE and RMSE of the predicted probabilities.

Further checks were a classification report, ROC curve and Brier score for the reliability of the probabilities. Statistical rigour for a PRA-regulated setting meant including uncertainty, bias and error estimates.

### Outcome and next steps

I presented the challenger, its validation results and how it compared with the existing decision tree to senior stakeholders and the people who would run the model. This included introducing Brier score testing and the handling of imbalanced data. The next step was to test the grades in a live environment alongside the existing model as part of model monitoring.

### Trade-offs and limitations

- **Explainability.** A more flexible model needs a clear rationale for regulators, and for why it is preferred to a simple tree.
- **Productionisation.** Moving the code into cloud infrastructure was harder than for a single tree.
- **Compute.** Training took longer, and limited local memory caused problems on large datasets when not running on cloud infrastructure.
- **External data.** Some inputs, such as house price data and outlook, were outside my control and needed checking before use.

---

## Professional discussion: evidence themes

- **Knowledge transfer:** set up Python adoption support (a Teams channel, shared repositories and presentation materials) to help build capability across a 60-person quantitative analytics function.
- **Quality assurance:** peer-reviewed code for internal colleagues and external contractors, as well as validating the transalation of SAS code to Python.
- **Stakeholder management:** explained model outputs to finance, audit and legal teams.
- **Monitoring:** established a quarterly monitoring framework for multiple mortgage risk models, with RAG-status reporting to senior leadership.
- **Continuous improvement:** keen intrest in model optimisation and model explainability.

---

## Open-Source Pipeline

To demonstrate the modelling framework without exposing confidential employer data, this repository includes a Python pipeline that generates a realistic and statistically calibrated synthetic mortgage portfolio. 

The data generator (`src/01_synthetic_data_generator.py`) uses a Monte Carlo simulation combined with real macroeconomic datasets (ONS, MoJ, FCA, HM Land Registry) to produce 600,000 defaulted mortgage accounts. The target variable (Probability of Possession given Default) is mathematically weighted to reflect real-world stress scenarios, achieving an authentic ~7% baseline possession rate across a 78-feature space.

### How to Reproduce

1. **Clone the repository:**
   `git clone https://github.com/Beck40/Mortgage-Risk-Modelling.git`
2. **Install dependencies:**
   `pip install -r requirements.txt`
3. **Acquire the public macro datasets:**
   Follow the instructions in `data/README.md` to download the required public datasets into `data/raw_macro/`.
4. **Generate the synthetic portfolio:**
   `python src/01_synthetic_data_generator.py`
5. **Output:**
   The script will generate four safely ignored `.csv` files inside `data/synthetic_portfolio/`, including the master book, an 80/20 Train/Test split (2019-2024), and an Out-of-Time validation set (2025-2026).

## References

- [PRA SS1/23: Model risk management principles for banks](https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/supervisory-statement/2023/ss123.pdf)
- [Basel III: international regulatory framework for banks](https://www.bis.org/bcbs/basel3.htm)
