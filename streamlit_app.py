import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

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
st.subheader("📅 Forecast for Date Range")

# Default range = last date + next 7 days
min_date = pd.to_datetime(df["date"]).min()
max_date = pd.to_datetime(df["date"]).max()
default_start = max_date - pd.Timedelta(days=7)
default_end = max_date + pd.Timedelta(days=7)

start_date, end_date = st.date_input(
    "Select start and end dates:",
    value=[default_start, default_end]
)

# -------------------- PREDICT BUTTON --------------------
if st.button("Predict"):
    st.success(f"✅ Forecast generated for {start_date} to {end_date}")

    # Prepare range of dates
    date_range = pd.date_range(start=start_date, end=end_date)
    forecast_rows = []

    latest = filtered_df.tail(1).copy()

    for d in date_range:
        input_row = latest.copy()
        input_row["date"] = d
        input_row["dayofweek"] = d.weekday()
        input_row["month"] = d.month
        input_row["is_weekend"] = 1 if d.weekday() >= 5 else 0
        input_row["sales_lag1"] = latest["sales"].iloc[0]
        input_row["sales_lag7"] = filtered_df.iloc[-7]["sales"] if len(filtered_df) >= 7 else latest["sales"].iloc[0]

        X = input_row[features]
        pred_val = model.predict(X)[0]
        forecast_rows.append([d, pred_val])

    forecast_df = pd.DataFrame(forecast_rows, columns=["date", "predicted_sales"])

    # -------------------- MERGE WITH ACTUAL DATA --------------------
    actual_df = filtered_df.copy()
    actual_df["date"] = pd.to_datetime(actual_df["date"])
    merged = pd.merge(actual_df, forecast_df, on="date", how="outer")

    # KPIs
    total_pred = forecast_df["predicted_sales"].sum()
    total_actual = actual_df[
        (actual_df["date"] >= pd.to_datetime(start_date)) & (actual_df["date"] <= pd.to_datetime(end_date))
    ]["sales"].sum()
    waste = total_pred * 0.15
    co2 = waste * 2.5

    colA, colB, colC, colD = st.columns(4)
    colA.metric("📊 Actual Sales (Range)", f"{int(total_actual)} units")
    colB.metric("🔮 Forecasted Sales", f"{int(total_pred)} units")
    colC.metric("🗑️ Estimated Waste", f"{int(waste)} kg")
    colD.metric("🌍 CO₂ Impact", f"{int(co2)} kg")

    st.write("---")

    # -------------------- COMBINED CHART --------------------
    fig = px.line(
        merged,
        x="date",
        y=["sales", "predicted_sales"],
        labels={"value": "Sales"},
        title=f"Actual vs Predicted Sales ({start_date} to {end_date})"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⏳ Select a date range & press **Predict** to view results.")

# -------------------- HISTORICAL SALES TREND --------------------
st.subheader("📈 Historical Sales Trend")

fig1 = px.line(
    filtered_df, x="date", y="sales", color="product",
    title="Historical Sales Trend"
)
st.plotly_chart(fig1, use_container_width=True)

# -------------------- FOOTER --------------------
st.write("---")
st.markdown(
    "<p style='text-align:center; font-size:13px;'>SmartWaste © Student Project | Built for M516</p>",
    unsafe_allow_html=True
)
