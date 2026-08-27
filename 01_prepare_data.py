"""
01_prepare_data.py
Loads the raw Kaggle 'Credit Card Customers' (BankChurners) dataset,
engineers an estimated revenue target, and builds the 'Revenue Opportunity
Score' — customers with unused capacity + engagement who are under-monetized.
"""
import os
import pandas as pd
import numpy as np

RAW_PATH = "BankChurners.csv"
OUT_PATH = "artifacts/customers_processed.csv"

# Make sure the artifacts/ folder exists before anything tries to write to it
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.read_csv(RAW_PATH)

# Drop the two leaked Naive Bayes columns (a well-known quirk of this dataset)
# and the raw client id — not needed for modeling.
nb_cols = [c for c in df.columns if c.startswith("Naive_Bayes_Classifier")]
df = df.drop(columns=nb_cols + ["CLIENTNUM"])

# --- Revenue engineering -----------------------------------------------
# Interchange revenue: banks earn ~1.5-2% of every dollar swiped
INTERCHANGE_RATE = 0.018
# Interest revenue: APR applied to the revolving (carried) balance
APR = 0.199
# Annual fee by card tier (typical US issuer pricing, used as an illustrative proxy)
FEE_MAP = {"Blue": 0, "Silver": 95, "Gold": 195, "Platinum": 595}

df["Interchange_Revenue"] = df["Total_Trans_Amt"] * INTERCHANGE_RATE
df["Interest_Revenue"] = df["Total_Revolving_Bal"] * APR
df["Fee_Revenue"] = df["Card_Category"].map(FEE_MAP)
df["Estimated_Annual_Revenue"] = (
    df["Interchange_Revenue"] + df["Interest_Revenue"] + df["Fee_Revenue"]
)

# --- The twist: Revenue Opportunity Score --------------------------------
# Idea: some customers have plenty of *unused* capacity (Avg_Open_To_Buy is
# high relative to their Credit_Limit) AND are still engaged with the bank
# (decent Total_Relationship_Count, not inactive) — but their revenue is low.
# These are the customers most worth targeting: they can spend more and are
# still reachable, unlike a maxed-out or disengaged customer.
#
# Capacity_Unused_Pct: how much of their line is untouched (0-1)
df["Capacity_Unused_Pct"] = df["Avg_Open_To_Buy"] / df["Credit_Limit"].replace(0, np.nan)
df["Capacity_Unused_Pct"] = df["Capacity_Unused_Pct"].fillna(0).clip(0, 1)

# Engagement_Score: normalized blend of relationship depth and recency of contact
rel_norm = (df["Total_Relationship_Count"] - df["Total_Relationship_Count"].min()) / (
    df["Total_Relationship_Count"].max() - df["Total_Relationship_Count"].min()
)
inactive_norm = 1 - (
    (df["Months_Inactive_12_mon"] - df["Months_Inactive_12_mon"].min())
    / (df["Months_Inactive_12_mon"].max() - df["Months_Inactive_12_mon"].min())
)
df["Engagement_Score"] = (rel_norm + inactive_norm) / 2

# Revenue_Percentile within same Card_Category (peer group), so we compare
# apples to apples — a Blue customer isn't benchmarked against a Platinum one.
df["Revenue_Percentile_In_Tier"] = df.groupby("Card_Category")["Estimated_Annual_Revenue"].rank(pct=True)

# Opportunity Score: high unused capacity + high engagement + LOW current revenue
# = the "money on the table" segment. Scaled 0-100 for readability.
df["Revenue_Opportunity_Score"] = (
    0.4 * df["Capacity_Unused_Pct"]
    + 0.35 * df["Engagement_Score"]
    + 0.25 * (1 - df["Revenue_Percentile_In_Tier"])
) * 100

# Only meaningful for customers still with the bank
df["Is_Active_Customer"] = (df["Attrition_Flag"] == "Existing Customer").astype(int)

df.to_csv(OUT_PATH, index=False)

print(f"Rows: {len(df)}")
print(f"Existing customers: {df['Is_Active_Customer'].sum()}")
print(f"Total estimated portfolio revenue: ${df['Estimated_Annual_Revenue'].sum():,.0f}")
print(f"Mean revenue: ${df['Estimated_Annual_Revenue'].mean():,.0f}")
print("\nTop-opportunity customers (existing only), sample:")
top_opp = df[df.Is_Active_Customer == 1].sort_values("Revenue_Opportunity_Score", ascending=False)
print(top_opp[["Customer_Age", "Card_Category", "Credit_Limit", "Capacity_Unused_Pct",
               "Engagement_Score", "Estimated_Annual_Revenue", "Revenue_Opportunity_Score"]].head(10).to_string(index=False))

# Quantify the "hidden revenue" opportunity: if the top-quartile-opportunity
# customers moved to their tier's median revenue, how much extra revenue?
active = df[df.Is_Active_Customer == 1].copy()
threshold = active["Revenue_Opportunity_Score"].quantile(0.75)
opp_segment = active[active["Revenue_Opportunity_Score"] >= threshold]
tier_median = active.groupby("Card_Category")["Estimated_Annual_Revenue"].median()
opp_segment = opp_segment.merge(tier_median.rename("Tier_Median_Revenue"), on="Card_Category")
opp_segment["Uplift_Potential"] = (opp_segment["Tier_Median_Revenue"] - opp_segment["Estimated_Annual_Revenue"]).clip(lower=0)

print(f"\nOpportunity segment size (top quartile): {len(opp_segment)} customers")
print(f"Total revenue uplift potential if brought to tier median: ${opp_segment['Uplift_Potential'].sum():,.0f}")
print(f"That's {opp_segment['Uplift_Potential'].sum() / df['Estimated_Annual_Revenue'].sum() * 100:.1f}% of current total portfolio revenue")
