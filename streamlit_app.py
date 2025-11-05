import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import timedelta

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SmartWaste – Germany",
    page_icon="🥦",
    layout="wide"
)

# Hide sidebar
hide_sidebar = """
<style>
[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E7D32;'>🥦 SmartWaste Dashboard</h1>
    <h4 style='text-align:center; color:#388E3C;'>AI-Powered Food Waste Forecasting for German Supermarkets 🇩🇪</h4>
    """,
    unsafe_allow_html=True
)

st.write("")

# -------------------- LOAD DATA & MODEL --------------------
df = pd.read_csv("demo_german_sales.csv")
model, features = joblib.load("xgb_model.joblib")

# -------------------- FILTERS --------------------
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.markdown("<h5 style='text-align:center;'>Filter Data</h5>", unsafe_allow_html=True)
    city = st.selectbox("Select City", ["All"] + sorted(df["city"].unique()))
    supermarket = st.selectbox("Select Supermarket", ["All"] + sorted(df["supermarket"].unique()))
    product = st.selectbox("Select Product", ["All"] + sorted(df["product"].unique()))

filtered_df = df.copy()
if city != "All":
    filtered_df = filtered_df[filtered_df["city"] == city]
if supermarket != "All":
    filtered_df = filtered_df[filtered_df["supermarket"] == supermarket]
if product != "All":
    filtered_df = filtered_df[filtered_df["product"] == product]

st.write("---")

# -------------------- DATE RANGE INPUT --------------------
st.subheader("📅 Forecast Date Range")

max_date = pd.to_datetime(df["date"]).max()
default_start = max_date + pd.Timedelta(days=1)
default_end = default_start + pd.Timedelta(days=6)

start_date = st.date_input("Start Date", value=default_start)
end_date = st.date_input("End Date", value=default_end)

# -------------------- FORECAST BUTTON --------------------
if st.button("Predict Range"):
    if start_date > end_date:
        st.error("❌ Start date must be before end date")
    else:
        current_date = start_date
        results = []

        last_row = filtered_df.tail(1).copy()
        lag1 = last_row["sales"].iloc[0]
        lag7 = filtered_df.iloc[-7]["sales"] if len(filtered_df) >= 7 else lag1

        while current_date <= end_date:
            row = last_row.copy()
            row["date"] = current_date
            row["dayofweek"] = current_date.weekday()
            row["month"] = current_date.month
            row["is_weekend"] = 1 if current_date.weekday() >= 5 else 0
            row["sales_lag1"] = lag1
            row["sales_lag7"] = lag7

            pred = model.predict(row[features])[0]

            waste = pred * 0.15
            co2 = waste * 2.5

            results.append([current_date, pred, waste, co2])

            # update lags
            lag7 = lag1
            lag1 = pred

            current_date += timedelta(days=1)

        result_df = pd.DataFrame(results, columns=["date","forecast","waste","co2"])

        # KPI totals
        colA, colB, colC = st.columns(3)
        colA.metric("📅 Days Forecasted", f"{len(result_df)} days")
        colB.metric("🔮 Total Forecasted Sales", f"{int(result_df['forecast'].sum())} units")
        colC.metric("🗑️ Est. Waste / CO₂", 
                    f"{int(result_df['waste'].sum())} kg / {int(result_df['co2'].sum())} kg")

        st.write("---")

        # Table
        st.write("### 📊 Forecast Results")
        st.dataframe(result_df)

        # Chart
        fig = px.bar(
            result_df, x="date", y="forecast",
            title=f"Forecasted Sales from {start_date} to {end_date}",
            labels={"forecast": "Predicted Sales"}
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⏳ Choose a start & end date, then press **Predict Range**")

# -------------------- HISTORICAL SALES PLOT --------------------
st.subheader("📈 Historical Sales Trend")
fig1 = px.line(
    filtered_df, x="date", y="sales", color="product",
    title="Sales Trend Over Time"
)
st.plotly_chart(fig1, use_container_width=True)

# -------------------- FOOTER --------------------
st.write("---")
st.markdown(
    "<p style='text-align:center; font-size:13px;'>SmartWaste © Student Project | M516 Submission</p>",
    unsafe_allow_html=True
)
