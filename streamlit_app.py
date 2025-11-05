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

# -------------------- DATE INPUT + BUTTON --------------------
st.subheader("📅 Forecast for a Selected Date")

default_date = pd.to_datetime(df["date"]).max() + pd.Timedelta(days=1)

selected_date = st.date_input(
    "Choose date to predict:",
    value=default_date
)

# -------------------- PREDICTION LOGIC --------------------
if st.button("Predict"):
    st.success(f"✅ Forecast generated for: **{selected_date}**")

    # Get latest row of filtered data
    latest = filtered_df.tail(1).copy()

    # Build input row for model
    input_row = latest.copy()
    input_row["date"] = selected_date
    input_row["dayofweek"] = pd.to_datetime(selected_date).weekday()
    input_row["month"] = pd.to_datetime(selected_date).month
    input_row["is_weekend"] = 1 if input_row["dayofweek"].iloc[0] >= 5 else 0

    input_row["sales_lag1"] = latest["sales"].iloc[0]
    input_row["sales_lag7"] = filtered_df.iloc[-7]["sales"] if len(filtered_df) >= 7 else latest["sales"].iloc[0]

    # Predict
    X = input_row[features]
    predicted_value = model.predict(X)[0]

    waste = predicted_value * 0.15
    co2 = waste * 2.5

    # -------------------- KPI CARDS --------------------
    colA, colB, colC = st.columns(3)
    colA.metric("📅 Date", f"{selected_date}")
    colB.metric("🔮 Forecasted Sales", f"{int(predicted_value)} units")
    colC.metric("🗑️ Estimated Waste", f"{int(waste)} kg | 🌍 CO₂: {int(co2)} kg")

    st.write("---")

    # -------------------- BAR CHART (ONLY SELECTED DATE) --------------------
    result_df = pd.DataFrame({
        "date": [selected_date],
        "predicted_sales": [predicted_value]
    })

    fig = px.bar(
        result_df, x="date", y="predicted_sales",
        title=f"Predicted Sales for {selected_date}",
        labels={"predicted_sales": "Forecasted Sales"}
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⏳ Choose a date & press **Predict** to display forecast.")

# -------------------- HISTORICAL LINE CHART --------------------
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
