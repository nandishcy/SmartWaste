import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SmartWaste – Food Waste Forecast (Germany)",
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
    <h1 style='text-align:center; color:#2E7D32;'>🥦 SmartWaste</h1>
    <h4 style='text-align:center; color:#388E3C;'>AI-Powered Food Waste Forecasting for German Supermarkets 🇩🇪</h4>
    """,
    unsafe_allow_html=True
)

st.write("")

# -------------------- LOAD DATA & MODEL --------------------
df = pd.read_csv("demo_german_sales.csv")
model, features = joblib.load("xgb_model.joblib")

# -------------------- FILTER SECTION (CENTER) --------------------
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

# -------------------- KPIs --------------------
last_week = filtered_df.tail(7)
X = last_week[features]
pred = model.predict(X)

predicted_total = pred.sum()
waste = predicted_total * 0.15     # assume 15% waste
co2 = waste * 2.5                   # 2.5 kg CO₂ per 1kg food waste

colA, colB, colC, colD = st.columns(4)
colA.metric("📊 Historical Sales (last week)", f"{int(last_week['sales'].sum())} units")
colB.metric("🔮 Predicted Sales (next week)", f"{int(predicted_total)} units")
colC.metric("🗑️ Estimated Waste", f"{int(waste)} kg")
colD.metric("🌍 CO₂ Impact Saved", f"{int(co2)} kg")

st.write("---")

# -------------------- GRAPH 1: HISTORICAL SALES --------------------
st.subheader("📈 Historical Sales Trend")

fig1 = px.line(filtered_df, x="date", y="sales", color="product",
               title="Sales Trend Over Time")
st.plotly_chart(fig1, use_container_width=True)

# -------------------- GRAPH 2: ACTUAL vs PREDICTED --------------------
st.subheader("🔮 Actual vs Predicted Sales (Last 7 Days Baseline)")

temp_df = last_week.copy()
temp_df["predicted_sales"] = pred

fig2 = px.bar(temp_df, x="date", y=["sales","predicted_sales"],
              barmode="group",
              title="Actual vs Predicted Sales")
st.plotly_chart(fig2, use_container_width=True)

# -------------------- FOOTER --------------------
st.write("---")
st.markdown(
    """
    <p style='text-align:center; font-size:13px; color:gray;'>
    SmartWaste • Predict & Prevent Food Waste • Built for M516 Project
    </p>
    """,
    unsafe_allow_html=True
)
