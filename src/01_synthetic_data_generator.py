import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("Generating defaulted mortgage portfolio for PPGD modelling...")


# Configuration: 600,000 defaulted accounts (representing the distressed cohort of a major UK lender)
N_ACCOUNTS = 600000
np.random.seed(42)

# 1. Base Identifiers & Default Cohort Dates (2019 to 2026)
start_date = datetime(2019, 1, 1)
total_days = (datetime(2026, 6, 30) - start_date).days
random_days = np.random.randint(0, total_days, N_ACCOUNTS)
default_dates = pd.to_datetime([start_date + timedelta(days=int(d)) for d in random_days])

df = pd.DataFrame({
    "account_id": [f"DEF_{str(i).zfill(7)}" for i in range(1, N_ACCOUNTS + 1)],
    "default_date": default_dates.strftime("%Y-%m-%d"),
    "default_year": default_dates.year
})

# 2. Application & Borrower Demographics (10 features)
df["borrower_age"] = np.random.randint(22, 68, N_ACCOUNTS)
df["applicant_income"] = np.random.lognormal(mean=10.4, sigma=0.45, size=N_ACCOUNTS).round(2)
df["co_applicant_income"] = np.where(np.random.rand(N_ACCOUNTS) > 0.55, 0, np.random.lognormal(mean=10.1, sigma=0.4, size=N_ACCOUNTS).round(2))
df["employment_type"] = np.random.choice(["PAYE", "Self-Employed", "Contractor", "Unemployed"], N_ACCOUNTS, p=[0.60, 0.20, 0.08, 0.12])
df["months_at_address"] = np.random.randint(12, 180, N_ACCOUNTS)
df["number_of_dependents"] = np.random.choice([0, 1, 2, 3, 4], N_ACCOUNTS, p=[0.35, 0.25, 0.22, 0.12, 0.06])
df["marital_status"] = np.random.choice(["Single", "Married", "Divorced"], N_ACCOUNTS, p=[0.40, 0.45, 0.15])
df["first_time_buyer_flag"] = np.random.binomial(1, 0.35, N_ACCOUNTS)
df["joint_application_flag"] = np.where(df["co_applicant_income"] > 0, 1, 0)
df["borrower_region_tier"] = np.random.choice([1, 2, 3, 4, 5], N_ACCOUNTS, p=[0.15, 0.25, 0.30, 0.20, 0.10])

# 3. Loan & Collateral Characteristics (14 features)
df["property_value_origination"] = np.random.lognormal(mean=12.45, sigma=0.42, size=N_ACCOUNTS).round(2)
df["hpi_adjustment_factor"] = np.random.normal(1.05, 0.12, N_ACCOUNTS).clip(0.75, 1.40).round(3)
df["current_property_value"] = (df["property_value_origination"] * df["hpi_adjustment_factor"]).round(2)
df["original_ltv"] = np.clip(np.random.normal(0.78, 0.12, N_ACCOUNTS), 0.30, 0.95).round(3)
df["original_balance"] = (df["property_value_origination"] * df["original_ltv"]).round(2)
df["current_balance"] = (df["original_balance"] * np.random.uniform(0.75, 1.05, N_ACCOUNTS)).round(2)
df["current_ltv"] = (df["current_balance"] / df["current_property_value"]).round(3)
df["interest_rate"] = np.where(df["default_year"] >= 2023, np.clip(np.random.normal(5.8, 1.1, N_ACCOUNTS), 3.0, 9.5), np.clip(np.random.normal(3.6, 0.9, N_ACCOUNTS), 1.5, 6.5)).round(2)
df["term_months"] = np.random.choice([180, 240, 300, 360, 420], N_ACCOUNTS, p=[0.10, 0.20, 0.40, 0.25, 0.05])
df["remaining_term_months"] = (df["term_months"] * np.random.uniform(0.4, 0.95, N_ACCOUNTS)).astype(int)
df["repayment_type"] = np.random.choice(["Repayment", "Interest Only", "Part and Part"], N_ACCOUNTS, p=[0.78, 0.17, 0.05])
df["product_rate_type"] = np.random.choice(["Fixed", "SVR", "Tracker"], N_ACCOUNTS, p=[0.65, 0.25, 0.10])
df["property_type"] = np.random.choice(["Detached", "Semi-Detached", "Terraced", "Flat"], N_ACCOUNTS, p=[0.18, 0.34, 0.32, 0.16])
df["property_tenure"] = np.random.choice(["Freehold", "Leasehold"], N_ACCOUNTS, p=[0.80, 0.20])

# 4. Distressed Behavioural, Arrears & Forbearance Features (18 features - Core Signal)
df["credit_score"] = np.clip(np.random.normal(510, 85, N_ACCOUNTS), 200, 800).astype(int)
df["months_in_arrears"] = np.random.choice([3, 4, 5, 6, 9, 12, 15, 18, 24], N_ACCOUNTS, p=[0.22, 0.18, 0.15, 0.14, 0.11, 0.08, 0.05, 0.04, 0.03])
df["months_since_default"] = np.clip(df["months_in_arrears"] - 2 + np.random.randint(0, 10, N_ACCOUNTS), 1, 36)
df["arrears_balance"] = (df["current_balance"] * (df["interest_rate"] / 100 / 12) * df["months_in_arrears"]).round(2)
df["missed_payments_12m"] = np.clip(df["months_in_arrears"] + np.random.randint(-1, 3, N_ACCOUNTS), 1, 12)
df["partial_payments_12m"] = np.random.poisson(lam=2.5, size=N_ACCOUNTS)
df["forbearance_flag"] = np.random.binomial(1, 0.45, N_ACCOUNTS)
df["broken_payment_arrangements_12m"] = np.random.choice([0, 1, 2, 3, 4], N_ACCOUNTS, p=[0.40, 0.30, 0.15, 0.10, 0.05])
df["payment_holiday_history_flag"] = np.random.binomial(1, 0.30, N_ACCOUNTS)
df["ccj_registered_flag"] = np.where(df["credit_score"] < 480, np.random.binomial(1, 0.45, N_ACCOUNTS), np.random.binomial(1, 0.10, N_ACCOUNTS))
df["ccj_value"] = np.where(df["ccj_registered_flag"] == 1, np.random.exponential(scale=3500, size=N_ACCOUNTS).round(2), 0.0)
df["iva_bankruptcy_flag"] = np.random.binomial(1, 0.06, N_ACCOUNTS)
df["unsecured_debt_balance"] = np.random.exponential(scale=8500, size=N_ACCOUNTS).round(2)
df["debt_to_income_ratio"] = ((df["current_balance"] + df["unsecured_debt_balance"]) / (df["applicant_income"] + df["co_applicant_income"] + 1)).round(2)
df["credit_card_utilization_pct"] = np.clip(np.random.normal(0.72, 0.20, N_ACCOUNTS), 0.0, 1.0).round(2)
df["days_in_overdraft_12m"] = np.random.randint(0, 365, N_ACCOUNTS)
df["bounced_direct_debits_12m"] = np.random.poisson(lam=2.2, size=N_ACCOUNTS)
df["litigation_stage_reached"] = np.where(df["months_since_default"] >= 9, np.random.binomial(1, 0.55, N_ACCOUNTS), 0)

# 5. Macroeconomic & Regional Features (6 features)
df["local_unemployment_rate"] = np.clip(np.random.normal(4.2 + df["borrower_region_tier"] * 0.6, 1.1, N_ACCOUNTS), 2.2, 11.5).round(2)
df["regional_claimant_count_rate"] = (df["local_unemployment_rate"] * np.random.uniform(0.75, 1.15, N_ACCOUNTS)).round(2)
df["local_authority_possession_rate"] = np.clip(np.random.normal(0.05 + df["borrower_region_tier"] * 0.01, 0.015, N_ACCOUNTS), 0.01, 0.15).round(4)
df["boe_base_rate_at_default"] = np.where(df["default_year"] >= 2023, np.random.choice([4.25, 4.50, 5.00, 5.25], N_ACCOUNTS), np.random.choice([0.10, 0.25, 0.75, 1.75], N_ACCOUNTS))
df["regional_house_price_growth_12m"] = np.where(df["default_year"] >= 2023, np.random.normal(-1.2, 2.5, N_ACCOUNTS), np.random.normal(4.5, 3.0, N_ACCOUNTS)).round(2)
df["cpi_inflation_rate"] = np.where(df["default_year"] >= 2023, np.random.normal(6.5, 2.0, N_ACCOUNTS), np.random.normal(2.2, 1.0, N_ACCOUNTS)).round(2)

# 6. Realistic Bank Data Warehouse / Operational Features (27 low-signal features for Feature Selection)
warehouse_features = {
    "has_current_account": np.random.binomial(1, 0.65, N_ACCOUNTS),
    "has_savings_account": np.random.binomial(1, 0.40, N_ACCOUNTS),
    "has_credit_card": np.random.binomial(1, 0.55, N_ACCOUNTS),
    "has_personal_loan": np.random.binomial(1, 0.25, N_ACCOUNTS),
    "has_home_insurance": np.random.binomial(1, 0.50, N_ACCOUNTS),
    "has_life_insurance": np.random.binomial(1, 0.30, N_ACCOUNTS),
    "has_auto_loan": np.random.binomial(1, 0.20, N_ACCOUNTS),
    "has_investment_isa": np.random.binomial(1, 0.12, N_ACCOUNTS),
    "online_banking_logins_30d": np.random.poisson(lam=6, size=N_ACCOUNTS),
    "mobile_app_logins_30d": np.random.poisson(lam=14, size=N_ACCOUNTS),
    "branch_visits_12m": np.random.poisson(lam=1.5, size=N_ACCOUNTS),
    "inbound_collections_calls_6m": np.random.poisson(lam=3.0, size=N_ACCOUNTS),
    "outbound_collections_calls_6m": np.random.poisson(lam=5.5, size=N_ACCOUNTS),
    "paperless_billing_flag": np.random.binomial(1, 0.78, N_ACCOUNTS),
    "marketing_email_opt_in": np.random.binomial(1, 0.45, N_ACCOUNTS),
    "marketing_sms_opt_in": np.random.binomial(1, 0.35, N_ACCOUNTS),
    "active_direct_debits_count": np.random.poisson(lam=5, size=N_ACCOUNTS),
    "active_standing_orders_count": np.random.poisson(lam=1.2, size=N_ACCOUNTS),
    "authorized_overdraft_limit": np.random.choice([0, 250, 500, 1000, 2000], N_ACCOUNTS, p=[0.35, 0.20, 0.25, 0.15, 0.05]),
    "total_liquid_savings": np.random.exponential(scale=1200, size=N_ACCOUNTS).round(2),
    "spend_groceries_6m": np.random.exponential(scale=1800, size=N_ACCOUNTS).round(2),
    "spend_utilities_6m": np.random.exponential(scale=1100, size=N_ACCOUNTS).round(2),
    "spend_retail_6m": np.random.exponential(scale=650, size=N_ACCOUNTS).round(2),
    "spend_travel_6m": np.random.exponential(scale=300, size=N_ACCOUNTS).round(2),
    "property_age_years": np.random.randint(1, 130, N_ACCOUNTS),
    "epc_energy_rating_score": np.random.randint(35, 92, N_ACCOUNTS),
    "number_of_bedrooms": np.random.choice([1, 2, 3, 4, 5], N_ACCOUNTS, p=[0.12, 0.32, 0.38, 0.14, 0.04])
}
for col_name, values in warehouse_features.items():
    df[col_name] = values

# 7. Calibrate Probability of Possession Given Default (PPGD Target ~7%)
# Driven by LTV, arrears depth, time since default (12m+ lag), broken arrangements, and macro stress
z_score = (
    (600 - df["credit_score"]) * 0.007 +
    (df["months_in_arrears"] - 6) * 0.16 +
    (df["months_since_default"] >= 12).astype(int) * 0.95 +
    (df["current_ltv"] - 0.80) * 3.8 +
    (df["broken_payment_arrangements_12m"] * 0.35) +
    (df["ccj_registered_flag"] * 0.65) +
    (df["local_unemployment_rate"] - 5.0) * 0.18 +
    (df["default_year"] >= 2023).astype(int) * 0.30
)

# Normalise log-odds and shift intercept to achieve ~7.0% possession rate across the defaulted portfolio
z_norm = (z_score - np.mean(z_score)) / np.std(z_score) # type: ignore
internal_ppgd_prob = 1 / (1 + np.exp(-(z_norm * 1.35 - 3.35)))

# Generate binary target without leaking internal_ppgd_prob into the CSV
df["possession_flag"] = np.random.binomial(1, internal_ppgd_prob)

# 8. Chronological Train / In-Sample Test / Out-of-Time (OOT) Splits
output_dir = "data/synthetic_portfolio"
os.makedirs(output_dir, exist_ok=True)

# Split 2019-2022 into 80% Train and 20% In-Sample Test; 2025-2026 into Out-of-Time (OOT)
in_time_mask = df["default_year"] <= 2024
in_time_df = df[in_time_mask].sample(frac=1.0, random_state=42).reset_index(drop=True)
oot_df = df[~in_time_mask].reset_index(drop=True)

split_idx = int(len(in_time_df) * 0.80)
train_df = in_time_df.iloc[:split_idx]
test_df = in_time_df.iloc[split_idx:]

# Save all files
df.to_csv(os.path.join(output_dir, "synthetic_loan_book.csv"), index=False)
train_df.to_csv(os.path.join(output_dir, "train_portfolio_2019_2024.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_portfolio_2019_2024.csv"), index=False)
oot_df.to_csv(os.path.join(output_dir, "oot_portfolio_2025_2026.csv"), index=False)

print(f"✓ Total Columns Generated: {df.shape[1]} (77 features + 1 target)")
print(f"✓ Full Defaulted Portfolio: {len(df):,} rows | PPGD Rate: {df['possession_flag'].mean()*100:.2f}%")
print(f"✓ Train Set (2019-2024, 80%): {len(train_df):,} rows | PPGD Rate: {train_df['possession_flag'].mean()*100:.2f}%")
print(f"✓ In-Sample Test (2019-2024, 20%): {len(test_df):,} rows | PPGD Rate: {test_df['possession_flag'].mean()*100:.2f}%")
print(f"✓ Out-of-Time Set (2025-2026): {len(oot_df):,} rows | PPGD Rate: {oot_df['possession_flag'].mean()*100:.2f}%")