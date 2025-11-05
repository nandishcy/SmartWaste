import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="SmartWaste – Food Waste Forecast (Germany)", page_icon="🥦", layout="wide")

st.title("🥦 SmartWaste – AI-Powered Food Waste Forecasting (Germany)")

# Load data and model
df = pd.read_csv("demo_german_sales.csv")
model, features = joblib.load("xgb_model.joblib")

st.sidebar.header("Filters")
city = st.sidebar.selectbox("Select City", ["All"] + sorted(df["city"].unique()))
product = st.sidebar.selectbox("Select Product", ["All"] + sorted(df["product"].unique()))

plot_df = df.copy()
if city != "All":
    plot_df = plot_df[plot_df["city"] == city]
if product != "All":
    plot_df = plot_df[plot_df["product"] == product]

st.subheader("📈 Historical Sales")
fig = px.line(plot_df, x="date", y="sales", color="product", title="Sales over Time")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔮 Predict Next Week’s Demand")
sample = plot_df.tail(7).copy()
X = sample[features]
pred = model.predict(X)
sample["predicted_sales"] = pred

fig2 = px.bar(sample, x="date", y=["sales","predicted_sales"], barmode="group",
              title="Actual vs Predicted Sales (Last Week)")
st.plotly_chart(fig2, use_container_width=True)

predicted_total = sample["predicted_sales"].sum()
waste = predicted_total * 0.15
co2 = waste * 2.5
st.success(f"Predicted Total Sales Next Week: {predicted_total:.0f}")
st.warning(f"Estimated Avoidable Waste: {waste:.0f} kg → CO₂ Savings ≈ {co2:.0f} kg")
