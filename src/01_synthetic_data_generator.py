import os
import numpy as np
import pandas as pd

print("Generating realistic non-stationary defaulted mortgage portfolio (100,000 accounts)...")

N_ACCOUNTS = 100000
np.random.seed(42)

# 1. Non-Stationary Quarterly Cohort Dates (2019 Q1 to 2026 Q2 - 30 Quarters)
quarters = pd.date_range(start="2019-01-01", end="2026-06-30", freq="QS")
# Mirror UK macro reality: 2020-2021 COVID moratorium dip, 2022-2024 BoE rate hike surge, 2025-2026 stabilisation
q_weights = np.array([
    0.95, 0.98, 1.00, 0.96,  # 2019
    0.85, 0.52, 0.48, 0.55,  # 2020 (Pandemic forbearance / FCA moratorium)
    0.60, 0.65, 0.72, 0.78,  # 2021
    0.82, 0.88, 0.96, 1.08,  # 2022 (Inflation & rate rises begin)
    1.18, 1.28, 1.40, 1.52,  # 2023 (Cost of living & fixed-rate roll-offs)
    1.60, 1.65, 1.58, 1.50,  # 2024 (Peak arrears stress)
    1.42, 1.36, 1.30, 1.24,  # 2025
    1.18, 1.14               # 2026 H1
])
q_weights = q_weights / q_weights.sum()

chosen_q_idx = np.random.choice(len(quarters), size=N_ACCOUNTS, p=q_weights)
base_q_dates = quarters[chosen_q_idx]
random_day_offsets = pd.to_timedelta(np.random.randint(0, 89, size=N_ACCOUNTS), unit="D")
default_dates = base_q_dates + random_day_offsets

df = pd.DataFrame({
    "account_id": [f"DEF_{str(i).zfill(7)}" for i in range(1, N_ACCOUNTS + 1)],
    "default_date": default_dates.strftime("%Y-%m-%d"),
    "default_quarter": pd.PeriodIndex(default_dates, freq="Q").astype(str),
    "default_year": default_dates.year
})

# Macro time-trend factor per quarter (drives non-stationary risk drift + seasonal wave)
q_macro_stress = np.array([
    -0.15, -0.12, -0.10, -0.08,  # 2019
    -0.25, -0.55, -0.60, -0.48,  # 2020 (Moratorium suppresses possessions)
    -0.40, -0.32, -0.25, -0.18,  # 2021
    -0.10, -0.02,  0.08,  0.18,  # 2022
     0.28,  0.38,  0.46,  0.52,  # 2023
     0.58,  0.62,  0.55,  0.48,  # 2024
     0.42,  0.36,  0.30,  0.25,  # 2025
     0.22,  0.18                 # 2026
])[chosen_q_idx]

# 2. Categorical Variables (26 Categorical Features - Realistic Banking Mix)
df["employment_type"] = np.random.choice(["PAYE_Permanent", "Self_Employed", "Fixed_Term_Contract", "Unemployed", "Retired"], N_ACCOUNTS, p=[0.58, 0.19, 0.08, 0.10, 0.05])
df["marital_status"] = np.random.choice(["Married", "Single", "Divorced", "Widowed"], N_ACCOUNTS, p=[0.46, 0.36, 0.14, 0.04])
df["repayment_type"] = np.random.choice(["Capital_and_Interest", "Interest_Only", "Part_and_Part"], N_ACCOUNTS, p=[0.76, 0.18, 0.06])
df["product_rate_type"] = np.random.choice(["Fixed_Rate", "Standard_Variable_Rate", "Tracker_Rate", "Discounted_Variable"], N_ACCOUNTS, p=[0.62, 0.24, 0.09, 0.05])
df["property_type"] = np.random.choice(["Semi_Detached", "Terraced", "Detached", "Purpose_Built_Flat", "Converted_Flat"], N_ACCOUNTS, p=[0.32, 0.30, 0.18, 0.14, 0.06])
df["property_tenure"] = np.random.choice(["Freehold", "Leasehold"], N_ACCOUNTS, p=[0.79, 0.21])
df["uk_region"] = np.random.choice(["London", "South_East", "North_West", "Yorkshire", "West_Midlands", "Scotland", "Wales", "South_West"], N_ACCOUNTS, p=[0.14, 0.18, 0.16, 0.13, 0.12, 0.10, 0.08, 0.09])
df["occupancy_status"] = np.random.choice(["Owner_Occupied", "Buy_To_Let", "Abandoned_Vacant", "Second_Home"], N_ACCOUNTS, p=[0.78, 0.15, 0.05, 0.02])
df["forbearance_type"] = np.random.choice(["No_Forbearance", "Payment_Concession", "Term_Extension", "Interest_Only_Switch", "Capitalisation"], N_ACCOUNTS, p=[0.50, 0.20, 0.14, 0.10, 0.06])
df["litigation_status"] = np.random.choice(["Pre_Litigation", "Letter_Before_Action", "Solicitor_Instructed", "Court_Hearing_Set", "Warrant_Issued"], N_ACCOUNTS, p=[0.48, 0.22, 0.15, 0.10, 0.05])
df["origination_channel"] = np.random.choice(["Broker_Introduced", "Direct_Online", "Branch_Network", "Telephony"], N_ACCOUNTS, p=[0.64, 0.18, 0.12, 0.06])
df["epc_energy_rating"] = np.random.choice(["A_B", "C", "D", "E", "F_G"], N_ACCOUNTS, p=[0.12, 0.38, 0.32, 0.13, 0.05])
df["collections_contact_status"] = np.random.choice(["Engaged_Cooperative", "Sporadic_Contact", "No_Contact_Unresponsive", "Refused_To_Pay"], N_ACCOUNTS, p=[0.42, 0.30, 0.20, 0.08])
df["arrears_reason_code"] = np.random.choice(["Income_Reduction", "Unemployment", "Relationship_Breakdown", "Illness", "Rate_Shock"], N_ACCOUNTS, p=[0.30, 0.24, 0.16, 0.12, 0.18])
df["valuation_type"] = np.random.choice(["Full_Physical", "AVM_Automated", "Drive_By_Exterior"], N_ACCOUNTS, p=[0.55, 0.35, 0.10])
df["income_verification_type"] = np.random.choice(["Fully_Verified_HMRC", "Accountant_Certificate", "Legacy_Fast_Track"], N_ACCOUNTS, p=[0.78, 0.16, 0.06])
df["solicitor_panel_tier"] = np.random.choice(["Tier_1_National", "Tier_2_Regional", "In_House_Legal"], N_ACCOUNTS, p=[0.50, 0.35, 0.15])
df["vulnerability_flag_type"] = np.random.choice(["Not_Vulnerable", "Financial_Shock", "Health_Impairment", "Bereavement"], N_ACCOUNTS, p=[0.74, 0.14, 0.08, 0.04])
df["direct_debit_status"] = np.random.choice(["Active_Paying", "Cancelled_By_Customer", "Repeatedly_Returned"], N_ACCOUNTS, p=[0.38, 0.36, 0.26])
df["customer_segment"] = np.random.choice(["Mass_Market", "Affluent", "Sub_Prime_Specialist", "First_Time_Buyer"], N_ACCOUNTS, p=[0.52, 0.14, 0.12, 0.22])
df["second_charge_consent"] = np.random.choice(["No_Second_Charge", "Authorized_Second_Charge", "Unauthorized_Charge"], N_ACCOUNTS, p=[0.84, 0.11, 0.05])
df["insurance_lapse_status"] = np.random.choice(["Active_Policy", "Lapsed_Uninsured", "Lender_Force_Placed"], N_ACCOUNTS, p=[0.72, 0.18, 0.10])

# Deliberate Concentration Trap Categoricals (>95% in single class so Notebook 2 Concentration Filter drops them!)
df["currency_denomination"] = "GBP"  # Zero-variance constant trap (Validity filter drop)
df["shared_ownership_scheme"] = np.random.choice(["Standard_100_Pct", "Shared_Equity"], N_ACCOUNTS, p=[0.975, 0.025])
df["Guarantor_Backed_Flag"] = np.random.choice(["No_Guarantor", "Active_Guarantor"], N_ACCOUNTS, p=[0.968, 0.032])
df["staff_mortgage_indicator"] = np.random.choice(["Non_Staff", "Bank_Employee"], N_ACCOUNTS, p=[0.988, 0.012])

# 3. Numerical Variables (49 Numerical Features - Core Behavioural, Loan, Macro & Operational)
df["borrower_age"] = np.random.randint(22, 69, N_ACCOUNTS)
df["applicant_income"] = np.random.lognormal(mean=10.45, sigma=0.48, size=N_ACCOUNTS).round(2)
df["co_applicant_income"] = np.where(np.random.rand(N_ACCOUNTS) > 0.52, 0.0, np.random.lognormal(mean=10.1, sigma=0.42, size=N_ACCOUNTS).round(2))
df["months_at_address"] = np.random.randint(6, 240, N_ACCOUNTS)
df["number_of_dependents"] = np.random.choice([0, 1, 2, 3, 4], N_ACCOUNTS, p=[0.38, 0.26, 0.22, 0.10, 0.04])

# Property & Exposure Dynamics (Higher exposures correlated with structural leverage)
df["property_value_origination"] = np.random.lognormal(mean=12.42, sigma=0.46, size=N_ACCOUNTS).round(2)
df["hpi_index_change_pct"] = np.random.normal(4.0 - q_macro_stress * 6.5, 8.0, N_ACCOUNTS).round(2)
df["current_property_value"] = (df["property_value_origination"] * (1 + df["hpi_index_change_pct"] / 100.0)).clip(40000, 5000000).round(2)
df["original_ltv"] = np.clip(np.random.normal(0.78, 0.13, N_ACCOUNTS), 0.25, 0.95).round(3)
df["original_balance"] = (df["property_value_origination"] * df["original_ltv"]).round(2)
df["current_balance"] = (df["original_balance"] * np.random.uniform(0.72, 1.06, N_ACCOUNTS)).round(2)
df["current_ltv"] = (df["current_balance"] / df["current_property_value"]).round(3)

# Interest Rate & Term
df["interest_rate"] = np.clip(np.random.normal(4.2 + q_macro_stress * 2.2, 1.1, N_ACCOUNTS), 1.4, 9.8).round(2)
df["original_term_months"] = np.random.choice([180, 240, 300, 360, 420], N_ACCOUNTS, p=[0.10, 0.22, 0.40, 0.23, 0.05])
df["remaining_term_months"] = (df["original_term_months"] * np.random.uniform(0.35, 0.95, N_ACCOUNTS)).astype(int)
df["loan_seasoning_months"] = df["original_term_months"] - df["remaining_term_months"]

# Arrears & Credit Behaviour (Strong Signal + Correlated Pairs for WoE Spearman Filter)
df["credit_score"] = np.clip(np.random.normal(505 - q_macro_stress * 25, 88, N_ACCOUNTS), 180, 820).astype(int)
df["external_bureau_score"] = np.clip(df["credit_score"] + np.random.normal(0, 22, N_ACCOUNTS), 150, 850).astype(int) # Correlated with credit_score (>0.70)
df["months_in_arrears"] = np.random.choice([3, 4, 5, 6, 8, 10, 12, 15, 18, 24], N_ACCOUNTS, p=[0.20, 0.17, 0.14, 0.13, 0.10, 0.08, 0.07, 0.05, 0.04, 0.02])
df["months_since_default"] = np.clip(df["months_in_arrears"] - 2 + np.random.randint(0, 6, N_ACCOUNTS), 1, 36) # Correlated with months_in_arrears (>0.70)
df["monthly_instalment_gbp"] = (df["current_balance"] * (df["interest_rate"] / 100 / 12) * 1.35).round(2)
df["arrears_balance"] = (df["monthly_instalment_gbp"] * df["months_in_arrears"] * np.random.uniform(0.85, 1.15, N_ACCOUNTS)).round(2)
df["arrears_to_balance_ratio"] = (df["arrears_balance"] / df["current_balance"]).round(4)
df["missed_payments_12m"] = np.clip(df["months_in_arrears"] + np.random.randint(-1, 3, N_ACCOUNTS), 1, 12)
df["partial_payments_12m"] = np.random.poisson(lam=2.4, size=N_ACCOUNTS)
df["broken_arrangements_12m"] = np.random.choice([0, 1, 2, 3, 4, 5], N_ACCOUNTS, p=[0.36, 0.28, 0.16, 0.11, 0.06, 0.03])
df["cure_attempts_count"] = np.random.poisson(lam=1.5, size=N_ACCOUNTS)
df["days_past_due"] = df["months_in_arrears"] * 30 + np.random.randint(0, 29, N_ACCOUNTS) # Heavily correlated with months_in_arrears

# Indebtedness & Bureau Metrics
df["ccj_count"] = np.where(df["credit_score"] < 480, np.random.choice([0, 1, 2, 3], N_ACCOUNTS, p=[0.50, 0.30, 0.14, 0.06]), np.random.choice([0, 1], N_ACCOUNTS, p=[0.90, 0.10]))
df["ccj_total_value_gbp"] = (df["ccj_count"] * np.random.exponential(scale=2800, size=N_ACCOUNTS)).round(2)
df["unsecured_debt_balance"] = np.random.exponential(scale=7800, size=N_ACCOUNTS).round(2)
df["debt_to_income_ratio"] = ((df["current_balance"] + df["unsecured_debt_balance"]) / (df["applicant_income"] + df["co_applicant_income"] + 1.0)).round(2)
df["revolving_credit_utilization"] = np.clip(np.random.normal(0.74, 0.21, N_ACCOUNTS), 0.0, 1.0).round(2)
df["overdraft_days_12m"] = np.random.randint(0, 365, N_ACCOUNTS)
df["bounced_dd_count_12m"] = np.random.poisson(lam=2.3, size=N_ACCOUNTS)
df["disposable_income_estimate"] = np.clip((df["applicant_income"] / 12.0) - df["monthly_instalment_gbp"] - 950, -2500, 10000).round(2)

# Macro Regional Indicators
df["local_unemployment_rate"] = np.clip(np.random.normal(4.4 + q_macro_stress * 1.4, 1.1, N_ACCOUNTS), 2.1, 11.8).round(2)
df["regional_claimant_rate"] = (df["local_unemployment_rate"] * np.random.uniform(0.82, 1.12, N_ACCOUNTS)).round(2) # Correlated with local_unemployment_rate
df["boe_base_rate"] = np.clip(1.25 + q_macro_stress * 4.5 + np.random.normal(0, 0.3, N_ACCOUNTS), 0.10, 5.25).round(2)
df["cpi_inflation_rate"] = np.clip(2.8 + q_macro_stress * 5.2 + np.random.normal(0, 0.6, N_ACCOUNTS), 0.5, 11.1).round(2)

# Operational / Warehouse Numerical Features
df["inbound_collections_calls_6m"] = np.random.poisson(lam=3.2, size=N_ACCOUNTS)
df["outbound_collections_calls_6m"] = np.random.poisson(lam=6.0, size=N_ACCOUNTS)
df["online_banking_logins_30d"] = np.random.poisson(lam=7.5, size=N_ACCOUNTS)
df["liquid_savings_balance"] = np.random.exponential(scale=950, size=N_ACCOUNTS).round(2)
df["property_age_years"] = np.random.randint(1, 125, N_ACCOUNTS)
df["number_of_bedrooms"] = np.random.choice([1, 2, 3, 4, 5], N_ACCOUNTS, p=[0.11, 0.31, 0.39, 0.15, 0.04])
df["legacy_system_constant_id"] = 1  # Numerical zero-variance trap for Validity Filter

# Deliberate Completeness Traps (>5% missing values so Completeness Filter drops them!)
df["legacy_affordability_score"] = np.where(np.random.rand(N_ACCOUNTS) < 0.14, np.nan, np.random.normal(50, 15, N_ACCOUNTS).round(1))
df["secondary_employer_tenure_m"] = np.where(np.random.rand(N_ACCOUNTS) < 0.22, np.nan, np.random.randint(1, 120, N_ACCOUNTS))

# 4. Calibrate PPGD Target (Volume Rate ~6.9%, Exposure-Weighted Rate ~8.4% due to higher LTV/Balance risk)
litigation_weight = np.where(df["litigation_status"].isin(["Court_Hearing_Set", "Warrant_Issued"]), 0.95, 0.0)
occupancy_weight = np.where(df["occupancy_status"].isin(["Abandoned_Vacant", "Buy_To_Let"]), 0.65, 0.0)
contact_weight = np.where(df["collections_contact_status"].isin(["No_Contact_Unresponsive", "Refused_To_Pay"]), 0.55, 0.0)
balance_log_stress = (np.log(df["current_balance"]) - 12.1) * 0.42  # Larger exposures face higher possession severity

z_score = (
    (580 - df["credit_score"]) * 0.0065 +
    (df["months_in_arrears"] - 6) * 0.155 +
    (df["current_ltv"] - 0.78) * 3.95 +
    (df["broken_arrangements_12m"] * 0.34) +
    litigation_weight +
    occupancy_weight +
    contact_weight +
    balance_log_stress +
    (df["local_unemployment_rate"] - 4.5) * 0.18 +
    q_macro_stress * 0.75
)

z_norm = (z_score - np.mean(z_score)) / np.std(z_score) #type:ignore
internal_ppgd_prob = 1 / (1 + np.exp(-(z_norm * 1.40 - 3.38)))
df["possession_flag"] = np.random.binomial(1, internal_ppgd_prob)

# 5. Export Chronological Splits (2019-2024 Train/Test, 2025-2026 OOT)
output_dir = "data/synthetic_portfolio"
os.makedirs(output_dir, exist_ok=True)

in_time_mask = df["default_year"] <= 2024
in_time_df = df[in_time_mask].sample(frac=1.0, random_state=42).reset_index(drop=True)
oot_df = df[~in_time_mask].reset_index(drop=True)

split_idx = int(len(in_time_df) * 0.80)
train_df = in_time_df.iloc[:split_idx]
test_df = in_time_df.iloc[split_idx:]

df.to_csv(os.path.join(output_dir, "synthetic_loan_book.csv"), index=False)
train_df.to_csv(os.path.join(output_dir, "train_portfolio_2019_2024.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_portfolio_2019_2024.csv"), index=False)
oot_df.to_csv(os.path.join(output_dir, "oot_portfolio_2025_2026.csv"), index=False)

vol_rate = train_df["possession_flag"].mean() * 100
ead_rate = (train_df.loc[train_df["possession_flag"] == 1, "current_balance"].sum() / train_df["current_balance"].sum()) * 100

print(f" Generated {len(df):,} rows | {df.shape[1]} total columns (26 Categorical, 49 Numerical, 3 ID/Date, 1 Target)")
print(f" Train Set: {len(train_df):,} rows | Volume PPGD: {vol_rate:.2f}% | Exposure-Weighted PPGD: {ead_rate:.2f}%")
print(f" Test Set:  {len(test_df):,} rows | OOT Set (2025-2026): {len(oot_df):,} rows")