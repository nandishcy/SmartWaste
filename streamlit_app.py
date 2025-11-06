import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import timedelta
import pydeck as pdk

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SmartWaste – Germany", page_icon="🥦", layout="wide")
st.markdown("<style>[data-testid='stSidebar'] {display:none;}</style>", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<h1 style='text-align:center;color:#2E7D32;'>🥦 SmartWaste Dashboard</h1>
<h4 style='text-align:center;color:#388E3C;'>AI-Powered Food Waste Forecasting for German Supermarkets 🇩🇪</h4>
""", unsafe_allow_html=True)
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
st.subheader("📅 Forecast / Compare Date Range")
min_date = pd.to_datetime(df["date"]).min()
max_date = pd.to_datetime(df["date"]).max()

default_start = max_date - pd.Timedelta(days=14)
default_end = max_date

start_date = st.date_input("Start Date", value=default_start, min_value=min_date)
end_date   = st.date_input("End Date",   value=default_end,   min_value=min_date)

# -------------------- PREDICT BUTTON --------------------
if st.button("Predict / Compare"):
    if start_date > end_date:
        st.error("❌ Start date must be before end date")
    else:
        results = []
        current_date = start_date
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

            lag7 = lag1
            lag1 = pred
            current_date += timedelta(days=1)

        pred_df = pd.DataFrame(results, columns=["date","predicted_sales","waste","co2"])

        # --- merge with actual sales (if exist in data range)
        df["date"] = pd.to_datetime(df["date"])
        actuals = filtered_df[(df["date"]>=start_date) & (df["date"]<=end_date)][["date","sales"]]
        merged = pd.merge(pred_df, actuals, on="date", how="left")

        # --- KPIs
        colA,colB,colC = st.columns(3)
        colA.metric("📅 Days", f"{len(merged)}")
        colB.metric("🔮 Total Forecasted", f"{int(merged['predicted_sales'].sum())} units")
        colC.metric("🗑️ Est. Waste / CO₂", f"{int(merged['waste'].sum())} kg / {int(merged['co2'].sum())} kg")

        st.write("---")

        # --- Table
        st.write("### 📊 Actual vs Predicted Data")
        st.dataframe(merged)

        # --- Chart: show both if actual available
        fig = px.bar(merged, x="date", y="predicted_sales",
                     color_discrete_sequence=["#4CAF50"],
                     title=f"Predicted vs Actual Sales ({start_date} → {end_date})",
                     labels={"predicted_sales":"Predicted Sales"})
        if merged["sales"].notna().any():
            fig.add_scatter(x=merged["date"], y=merged["sales"], mode="lines+markers",
                            name="Actual Sales", line=dict(color="#1E88E5"))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⏳ Select a date range then click **Predict / Compare**")

# -------------------- MAP OF GERMANY --------------------
st.subheader("🗺️ Store Locations in Germany")

city_coords = {
    "Berlin": [52.5200, 13.4050],
    "Munich": [48.1351, 11.5820],
    "Hamburg": [53.5511, 9.9937],
    "Cologne": [50.9375, 6.9603],
    "Frankfurt": [50.1109, 8.6821]
}

map_data = pd.DataFrame([
    {"city": c, "lat": city_coords[c][0], "lon": city_coords[c][1], "supermarket": s}
    for c in city_coords.keys()
    for s in ["Edeka","Rewe","Lidl","Aldi","Kaufland","Penny"]
])

if city != "All":
    map_data = map_data[map_data["city"] == city]
if supermarket != "All":
    map_data = map_data[map_data["supermarket"] == supermarket]

st.map(map_data, latitude="lat", longitude="lon", size=40, color="#2E7D32")

# -------------------- HISTORICAL SALES --------------------
st.subheader("📈 Full Historical Sales Trend")
fig2 = px.line(filtered_df, x="date", y="sales", color="product",
               title="Historical Sales Over Time")
st.plotly_chart(fig2, use_container_width=True)

# -------------------- FOOTER --------------------
st.write("---")
st.markdown("<p style='text-align:center;font-size:13px;'>SmartWaste © Student Project | M516 Submission</p>",
            unsafe_allow_html=True)
