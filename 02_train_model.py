"""
02_train_model.py
Trains a revenue-driver model (XGBoost) and uses SHAP to identify the
top factors driving Estimated_Annual_Revenue. Excludes the raw dollar
fields that mechanically compose the revenue formula (Total_Trans_Amt,
Total_Revolving_Bal) so the model explains revenue via genuine behavioral
and relationship drivers, not algebra.
"""
import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

df = pd.read_csv("artifacts/customers_processed.csv")
active = df[df.Is_Active_Customer == 1].copy()

TARGET = "Estimated_Annual_Revenue"

FEATURES = [
    "Customer_Age", "Gender", "Dependent_count", "Education_Level",
    "Marital_Status", "Income_Category", "Card_Category", "Months_on_book",
    "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
    "Credit_Limit", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1",
    "Total_Amt_Chng_Q4_Q1", "Avg_Utilization_Ratio",
]
CAT_FEATURES = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]

X = active[FEATURES].copy()
y = active[TARGET].copy()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[CAT_FEATURES] = encoder.fit_transform(X[CAT_FEATURES])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
print(f"R2: {r2:.3f}  |  MAE: ${mae:,.0f}")

# --- SHAP for the 4-factor story -----------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\nFeature importance (SHAP mean |value|):")
print(importance.to_string(index=False))

top4 = importance.head(4)["feature"].tolist()
print(f"\nTop 4 revenue drivers: {top4}")

# Save everything the Streamlit app needs
joblib.dump(model, "artifacts/revenue_model.joblib")
joblib.dump(encoder, "artifacts/encoder.joblib")
joblib.dump(FEATURES, "artifacts/features.joblib")
joblib.dump(CAT_FEATURES, "artifacts/cat_features.joblib")
np.save("artifacts/shap_values.npy", shap_values)
X.to_csv("artifacts/X_encoded.csv", index=False)
importance.to_csv("artifacts/feature_importance.csv", index=False)

with open("artifacts/metrics.txt", "w") as f:
    f.write(f"R2={r2:.4f}\nMAE={mae:.2f}\nTop4={','.join(top4)}\n")
