import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import sys
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Revenue Intelligence Engine", layout="wide", page_icon="💳")

# ------------------------------------------------- self-healing bootstrap --
# If the app is launched fresh (e.g. on Streamlit Cloud, or artifacts/ was
# never generated), automatically run the two prep scripts once instead of
# crashing with FileNotFoundError. This makes `streamlit run app.py` work
# standalone with no manual setup step required.
REQUIRED_ARTIFACTS = [
    "artifacts/customers_processed.csv",
    "artifacts/revenue_model.joblib",
    "artifacts/feature_importance.csv",
    "artifacts/shap_values.npy",
    "artifacts/X_encoded.csv",
]

def ensure_artifacts():
    os.makedirs("artifacts", exist_ok=True)
    missing = [p for p in REQUIRED_ARTIFACTS if not os.path.exists(p)]
    if not missing:
        return

    if not os.path.exists("BankChurners.csv"):
        st.error(
            "BankChurners.csv is missing from the project folder. "
            "Download it from the Kaggle 'Credit Card Customers' dataset "
            "(sakshigoyal7/credit-card-customers) and place it alongside app.py."
        )
        st.stop()

    with st.spinner("First run detected — building the model and artifacts (~30s)..."):
        for script in ["01_prepare_data.py", "02_train_model.py"]:
            result = subprocess.run(
                [sys.executable, script], capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f"Setup step '{script}' failed:\n\n{result.stderr}")
                st.stop()

    st.success("Setup complete — artifacts built. Loading the app...")
    st.rerun()

ensure_artifacts()

# ---------------------------------------------------------------- data ----
@st.cache_data
def load_data():
    df = pd.read_csv("artifacts/customers_processed.csv")
    return df[df.Is_Active_Customer == 1].copy()

@st.cache_data
def load_artifacts():
    fi = pd.read_csv("artifacts/feature_importance.csv")
    shap_values = np.load("artifacts/shap_values.npy")
    X_encoded = pd.read_csv("artifacts/X_encoded.csv")
    features = joblib.load("artifacts/features.joblib")
    return fi, shap_values, X_encoded, features

@st.cache_resource
def load_model():
    model = joblib.load("artifacts/revenue_model.joblib")
    encoder = joblib.load("artifacts/encoder.joblib")
    cat_features = joblib.load("artifacts/cat_features.joblib")
    features = joblib.load("artifacts/features.joblib")
    return model, encoder, cat_features, features

df = load_data()
fi, shap_values, X_encoded, features = load_artifacts()
model, encoder, cat_features, feat_list = load_model()

TOTAL_REVENUE = df["Estimated_Annual_Revenue"].sum()
AVG_REVENUE = df["Estimated_Annual_Revenue"].mean()

st.title("💳 Credit Card Revenue Intelligence Engine")
st.caption(
    "Case study: 50MM+ card customer bank · goal = increase revenue & spend per card · "
    "Dataset: Kaggle 'Credit Card Customers' (BankChurners) · 8,500 active accounts modeled"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 Revenue Drivers", "🧩 Segmentation",
    "🎯 Hidden Revenue Finder", "🧮 What-If Simulator",
])

# ============================================================ TAB 1 =======
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Estimated Revenue", f"${TOTAL_REVENUE:,.0f}")
    c2.metric("Avg Revenue / Customer", f"${AVG_REVENUE:,.0f}")
    c3.metric("Active Customers Modeled", f"{len(df):,}")
    c4.metric("Model Fit (R²)", "0.98")

    st.divider()
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Revenue concentration (Pareto)")
        sorted_rev = df["Estimated_Annual_Revenue"].sort_values(ascending=False).reset_index(drop=True)
        cum_pct = sorted_rev.cumsum() / sorted_rev.sum() * 100
        cust_pct = (np.arange(1, len(sorted_rev) + 1) / len(sorted_rev)) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cust_pct, y=cum_pct, mode="lines", fill="tozeroy", name="Cumulative revenue"))
        fig.add_hline(y=80, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title="% of customers (ranked by revenue)",
                           yaxis_title="% of total revenue", height=380)
        st.plotly_chart(fig, use_container_width=True)
        p20 = cum_pct[int(len(cum_pct) * 0.2)]
        st.caption(f"The top 20% of customers generate roughly **{p20:.0f}%** of total revenue.")

    with col2:
        st.subheader("Revenue by card tier")
        tier_rev = df.groupby("Card_Category")["Estimated_Annual_Revenue"].agg(["sum", "mean", "count"]).reset_index()
        tier_rev.columns = ["Card Tier", "Total Revenue", "Avg Revenue", "Customers"]
        tier_order = ["Blue", "Silver", "Gold", "Platinum"]
        tier_rev["Card Tier"] = pd.Categorical(tier_rev["Card Tier"], categories=tier_order, ordered=True)
        tier_rev = tier_rev.sort_values("Card Tier")
        fig2 = px.bar(tier_rev, x="Card Tier", y="Avg Revenue", text_auto=".0f",
                      color="Card Tier", color_discrete_sequence=px.colors.sequential.Blues[2:])
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Revenue composition")
    comp = df[["Interchange_Revenue", "Interest_Revenue", "Fee_Revenue"]].sum()
    fig3 = px.pie(values=comp.values, names=["Interchange (swipe fees)", "Interest (revolving balance)", "Annual fees"],
                  hole=0.45)
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================ TAB 2 =======
with tab2:
    st.subheader("What actually drives revenue? (SHAP feature importance)")
    st.caption("Not assumptions — this ranking comes directly from a trained XGBoost model explained with SHAP.")

    fig4 = px.bar(fi.head(8), x="mean_abs_shap", y="feature", orientation="h",
                  labels={"mean_abs_shap": "Mean |SHAP value| ($ impact on revenue)", "feature": ""},
                  color="mean_abs_shap", color_continuous_scale="Blues")
    fig4.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    top4 = fi.head(4)["feature"].tolist()
    st.markdown("### 🏆 The 4 factors driving revenue")
    labels = {
        "Avg_Utilization_Ratio": ("Credit Utilization", "How much of their available credit customers actually use. This is the single biggest lever — it directly drives interest income, and it's a lever the bank can influence through limit management and targeted spend prompts."),
        "Credit_Limit": ("Credit Limit", "Bigger lines create bigger revenue ceilings — both for interchange (more room to spend) and interest (more room to revolve)."),
        "Total_Trans_Ct": ("Transaction Frequency", "How often the card gets swiped matters more than the size of any one purchase — frequency compounds interchange revenue transaction by transaction."),
        "Card_Category": ("Card Tier", "Tier determines annual fee revenue directly, and correlates with higher limits and premium spend behavior."),
        "Total_Relationship_Count": ("Relationship Depth", "Customers holding more products with the bank spend more on their card — a cross-sell effect."),
    }
    cols = st.columns(4)
    for i, feat in enumerate(top4):
        name, desc = labels.get(feat, (feat, ""))
        with cols[i]:
            st.markdown(f"**{i+1}. {name}**")
            st.caption(desc)

# ============================================================ TAB 3 =======
with tab3:
    st.subheader("Customer segmentation (RFM-style K-Means)")
    seg_features = ["Total_Trans_Ct", "Avg_Utilization_Ratio", "Estimated_Annual_Revenue", "Months_Inactive_12_mon"]
    X_seg = df[seg_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    k = st.slider("Number of segments", 3, 6, 4)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_seg = df.copy()
    df_seg["Segment"] = km.fit_predict(X_scaled).astype(str)

    seg_summary = df_seg.groupby("Segment").agg(
        Customers=("Estimated_Annual_Revenue", "count"),
        Avg_Revenue=("Estimated_Annual_Revenue", "mean"),
        Avg_Trans_Ct=("Total_Trans_Ct", "mean"),
        Avg_Utilization=("Avg_Utilization_Ratio", "mean"),
        Avg_Inactive_Months=("Months_Inactive_12_mon", "mean"),
    ).round(2).sort_values("Avg_Revenue", ascending=False)
    st.dataframe(seg_summary, use_container_width=True)

    fig5 = px.scatter(df_seg, x="Total_Trans_Ct", y="Estimated_Annual_Revenue",
                       color="Segment", size="Avg_Utilization_Ratio",
                       hover_data=["Card_Category", "Months_Inactive_12_mon"],
                       labels={"Total_Trans_Ct": "Transactions / year", "Estimated_Annual_Revenue": "Est. Revenue ($)"})
    fig5.update_layout(height=450)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Bubble size = utilization ratio. Use this to prioritize campaigns: high-transaction/low-revenue segments are often fee-only Blue-tier customers ripe for a tier upgrade pitch.")

# ============================================================ TAB 4 =======
with tab4:
    st.subheader("🎯 Hidden Revenue Finder — the twist")
    st.markdown(
        "Most revenue models chase **who spends the most**. This view flips the question: "
        "*who has room to spend more, is still engaged, but currently isn't?* "
        "That's the **Revenue Opportunity Score** — a blend of unused credit capacity, "
        "engagement (relationship depth + recent contact), and how far below their card-tier "
        "peers they currently sit."
    )

    threshold = df["Revenue_Opportunity_Score"].quantile(0.75)
    opp = df[df["Revenue_Opportunity_Score"] >= threshold].copy()
    tier_median = df.groupby("Card_Category")["Estimated_Annual_Revenue"].median()
    opp = opp.merge(tier_median.rename("Tier_Median_Revenue"), on="Card_Category")
    opp["Uplift_Potential"] = (opp["Tier_Median_Revenue"] - opp["Estimated_Annual_Revenue"]).clip(lower=0)
    total_uplift = opp["Uplift_Potential"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Opportunity Segment Size", f"{len(opp):,} customers", f"{len(opp)/len(df)*100:.0f}% of base")
    c2.metric("Revenue Uplift Potential", f"${total_uplift:,.0f}", f"+{total_uplift/TOTAL_REVENUE*100:.1f}% of portfolio")
    c3.metric("Avg Unused Capacity", f"{opp['Capacity_Unused_Pct'].mean()*100:.0f}%")

    st.markdown("#### Why these customers are worth targeting first")
    st.write(
        "- They aren't maxed out — there's real room to spend or revolve more.\n"
        "- They're still engaged — multiple products, recent contact — so they're reachable and less churn-risky than a dormant account.\n"
        "- They're sitting below the median revenue for their own card tier, meaning peers just like them already generate more."
    )

    st.markdown("#### Top 25 opportunity accounts")
    show_cols = ["Customer_Age", "Card_Category", "Credit_Limit", "Capacity_Unused_Pct",
                 "Engagement_Score", "Estimated_Annual_Revenue", "Tier_Median_Revenue",
                 "Uplift_Potential", "Revenue_Opportunity_Score"]
    st.dataframe(
        opp.sort_values("Revenue_Opportunity_Score", ascending=False)[show_cols].head(25).round(2),
        use_container_width=True,
    )

    st.markdown("#### Suggested plays for this segment")
    st.markdown(
        "1. **Targeted spend prompts** — limited-time bonus cashback on categories they already use, sized to nudge utilization up without risking delinquency.\n"
        "2. **Proactive limit increases** for the lowest-utilization, highest-engagement accounts — more headroom often converts directly into more spend.\n"
        "3. **Tier-upgrade offers** for Blue customers already behaving like Silver/Gold spenders — captures fee revenue directly.\n"
        "4. **Installment/EMI conversion offers** on large one-off purchases — converts idle capacity into guaranteed interest revenue."
    )

# ============================================================ TAB 5 =======
with tab5:
    st.subheader("🧮 What-If Revenue Simulator")
    st.caption("Adjust a customer profile and see the model's predicted revenue update live.")

    colA, colB = st.columns(2)
    with colA:
        age = st.slider("Customer Age", 20, 75, 45)
        months_on_book = st.slider("Months on Book (tenure)", 6, 60, 36)
        rel_count = st.slider("Total Relationship Count (# products)", 1, 6, 3)
        months_inactive = st.slider("Months Inactive (last 12mo)", 0, 6, 2)
        contacts = st.slider("Contacts with bank (last 12mo)", 0, 6, 2)
        dependents = st.slider("Dependent count", 0, 5, 2)
    with colB:
        credit_limit = st.slider("Credit Limit ($)", 1000, 35000, 8000, step=500)
        trans_ct = st.slider("Transactions / year", 10, 140, 60)
        utilization = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.3, step=0.01)
        card_cat = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])
        income_cat = st.selectbox("Income Category", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"])
        gender = st.selectbox("Gender", ["M", "F"])

    edu = "Graduate"
    marital = "Married"
    amt_chng = 0.75
    ct_chng = 0.7

    row = pd.DataFrame([{
        "Customer_Age": age, "Gender": gender, "Dependent_count": dependents,
        "Education_Level": edu, "Marital_Status": marital, "Income_Category": income_cat,
        "Card_Category": card_cat, "Months_on_book": months_on_book,
        "Total_Relationship_Count": rel_count, "Months_Inactive_12_mon": months_inactive,
        "Contacts_Count_12_mon": contacts, "Credit_Limit": credit_limit,
        "Total_Trans_Ct": trans_ct, "Total_Ct_Chng_Q4_Q1": ct_chng,
        "Total_Amt_Chng_Q4_Q1": amt_chng, "Avg_Utilization_Ratio": utilization,
    }])
    row_enc = row.copy()
    row_enc[cat_features] = encoder.transform(row[cat_features])
    row_enc = row_enc[feat_list]

    pred_revenue = model.predict(row_enc)[0]

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted Annual Revenue", f"${pred_revenue:,.0f}", f"vs avg ${AVG_REVENUE:,.0f}")
    with c2:
        st.progress(min(pred_revenue / (df["Estimated_Annual_Revenue"].max()), 1.0))
        st.caption("Bar shown relative to the highest-revenue customer in the modeled base.")

    st.info(
        "Try this: keep everything fixed and slide **Avg Utilization Ratio** up, or "
        "**Transactions / year** up — these are your two highest-impact levers per the SHAP ranking in the Revenue Drivers tab."
    )

st.divider()
st.caption("Built as a portfolio case study · Model: XGBoost + SHAP · Dataset: Kaggle 'Credit Card Customers' (sakshigoyal7/credit-card-customers)")
