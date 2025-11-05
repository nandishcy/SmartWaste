import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="SmartWaste – Germany",
    page_icon="🥦",
    layout="wide"
)

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
    <h4 style='text-align:center; color:#388E3C;'>AI-Driven Food Waste Forecasting for German Supermarkets 🇩🇪</h4>
    """,
    unsafe_allow_html=True
)
st.write("")

# -------------------- LOAD DATA & MODEL --------------------
df = pd.read_csv("demo_german_sales.csv")
model, features = joblib.load("xgb_model.joblib")

# -------------------- FILTERS --------------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown("<h5 style='text-align:center;'>Filter Data</h5>", unsafe_allow_html=True)
    city = st.selectbox("Select City", ["All"] + sorted(df["city"].unique()))
    product = st.selectbox("Select Product", ["All"] + sorted(df["product"].unique()))

filtered_df = df.copy()
if city != "All":
    filtered_df = filtered_df[filtered_df["city"] == city]
if product != "All":
    filtered_df = filtered_df[filtered_df["product"] == product]

st.write("---")

# -------------------- DATE INPUT + PREDICT BUTTON --------------------
st.subheader("📅 Forecast for a Specific Date")

# Default = next day after last date in data
default_date = pd.to_datetime(df["date"]).max() + pd.Timedelta(days=1)

selected_date = st.date_input("Choose a date to forecast:", value=default_date)

if st.button("Predict"):
    st.success(f"✅ Forecast generated for: **{selected_date}**")

    # Last 7 days as model input
    history = filtered_df.tail(7).copy()
    X = history[features]
    preds = model.predict(X)

    # Calculations
    predicted_total = preds.sum()
    waste = predicted_total * 0.15   # 15% assumption
    co2 = waste * 2.5                # 2.5kg CO2 per kg food waste

    # -------------------- KPI CARDS --------------------
    colA, colB, colC, colD = st.columns(4)
    colA.metric("📊 Last Week Sales", f"{int(history['sales'].sum())} units")
    colB.metric("🔮 Forecasted Sales", f"{int(predicted_total)} units")
    colC.metric("🗑️ Estimated Waste", f"{int(waste)} kg")
    colD.metric("🌍 CO₂ Impact", f"{int(co2)} kg")

    st.write("---")

    # -------------------- FORECAST BAR CHART --------------------
    temp = history.copy()
    temp["predicted_sales"] = preds

    fig2 = px.bar(
        temp, x="date", y=["sales","predicted_sales"],
        barmode="group",
        title=f"Actual vs Predicted Sales (up to {selected_date})"
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("⏳ Select a date & press **Predict** to view results.")

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
    "<p style='text-align:center; font-size:13px;'>SmartWaste © Student Project</p>",
    unsafe_allow_html=True
)
