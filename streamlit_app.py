# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import pydeck as pdk
from datetime import timedelta

# -------------------- Page config --------------------
st.set_page_config(
    page_title="GreenForecast • AI Retail",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Dark theme CSS --------------------
dark_css = """
<style>
    .reportview-container, .main, .block-container {
        background: linear-gradient(180deg, #0f1720 0%, #071016 100%);
        color: #e6f4ea;
    }
    .stButton>button {
        background-color: #0ea861;
        color: white;
    }
    .stTabs [role="tab"] {
        background-color: #081219;
        color: #cfeedd;
    }
    .css-1d391kg { /* Streamlit header size tweak */
        font-size: 18px;
    }
    .metric-label { color: #bfe9c6 !important; }
    .stMarkdown p { color: #d7f0db; }
    .card {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown(
    """
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div>
        <h1 style="margin:5px 0;color:#adf5c2">GreenForecast</h1>
        <div style="color:#9de8b8">AI Demand Forecasting & Order Optimization — German Retail</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:12px;color:#9de8b8"></div>
      </div>
    </div>
    <hr style="border:1px solid rgba(255,255,255,0.06)"/>
    """, unsafe_allow_html=True
)

# -------------------- Load data & model --------------------
DATA_PATH = "demo_german_sales.csv"
MODEL_PATH = "xgb_model.joblib"
SHAP_PATH = "shap_summary.png"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

@st.cache_resource
def load_model(path):
    try:
        mdl, feats = joblib.load(path)
        return mdl, feats
    except Exception as e:
        st.error(f"Could not load model file at {path}: {e}")
        return None, []

df = load_data(DATA_PATH)
model, model_features = load_model(MODEL_PATH)

# -------------------- Sidebar: Global filters --------------------
with st.sidebar:
    st.header("Filters")
    city = st.selectbox("City", ["All"] + sorted(df["city"].unique()))
    supermarket = st.selectbox("Supermarket", ["All"] + sorted(df["supermarket"].unique()))
    product = st.selectbox("Product", ["All"] + sorted(df["product"].unique()))
    st.markdown("---")
    st.write("Quick actions")
    if st.button("Reset filters"):
        city = "All"; supermarket = "All"; product = "All"
    st.markdown("<div style='font-size:12px;color:#cfeedd'></div>", unsafe_allow_html=True)

# Apply filters for use across pages
filtered = df.copy()
if city != "All":
    filtered = filtered[filtered["city"] == city]
if supermarket != "All":
    filtered = filtered[filtered["supermarket"] == supermarket]
if product != "All":
    filtered = filtered[filtered["product"] == product]

# -------------------- Tabs (pages) --------------------
tabs = st.tabs(["Overview", "Forecast", "Insights", "Optimize", "Map"])

# -------------------- Helpers --------------------
def prepare_input_for_date(base_df, selected_date):
    """
    Create a single-row input based on latest available row in base_df,
    but set the date to selected_date and update dayofweek/month/is_weekend.
    """
    latest = base_df.sort_values("date").tail(1).copy()
    if latest.empty:
        st.error("No historical rows available for selected filters.")
        return None
    row = latest.copy()
    row.loc[:, "date"] = pd.to_datetime(selected_date)
    row.loc[:, "dayofweek"] = pd.to_datetime(selected_date).weekday()
    row.loc[:, "month"] = pd.to_datetime(selected_date).month
    row.loc[:, "is_weekend"] = 1 if row["dayofweek"].iloc[0] >= 5 else 0
    # lags: use last observed values
    row.loc[:, "sales_lag1"] = latest["sales"].iloc[0]
    # sales_lag7: mean of last 7 if available else last
    if len(base_df) >= 7:
        row.loc[:, "sales_lag7"] = base_df.sort_values("date").tail(7)["sales"].mean()
    else:
        row.loc[:, "sales_lag7"] = latest["sales"].iloc[0]
    return row

def forecast_range(base_df, start_date, end_date, model, features):
    dates = pd.date_range(start=start_date, end=end_date)
    rows = []
    latest = base_df.sort_values("date").tail(14).iloc[-1:].copy()
    for d in dates:
        r = latest.copy()
        r.loc[:, "date"] = pd.to_datetime(d)
        r.loc[:, "dayofweek"] = d.weekday()
        r.loc[:, "month"] = d.month
        r.loc[:, "is_weekend"] = 1 if d.weekday() >= 5 else 0
        r.loc[:, "sales_lag1"] = base_df.sort_values("date").tail(1)["sales"].values[0]
        r.loc[:, "sales_lag7"] = base_df.sort_values("date").tail(7)["sales"].mean() if len(base_df)>=7 else r.loc[:, "sales_lag1"]
        X = r[features]
        pred = model.predict(X)[0]
        rows.append({"date": d, "predicted_sales": float(pred)})
    return pd.DataFrame(rows)

def optimize_order(predicted_mean, safety_factor=0.95):
    # simple heuristic optimization to suggest order quantity
    # tuned to be conservative: adds small buffer
    q = int(round(predicted_mean * (1 + (1 - safety_factor) * 0.5 + 0.10)))
    return max(0, q)

# -------------------- Page: Overview --------------------
with tabs[0]:
    st.subheader("Overview")
    total_history = int(filtered["sales"].sum())
    avg_daily = int(filtered.groupby("date")["sales"].sum().mean()) if not filtered.empty else 0
    avg_waste = int(filtered["waste"].mean()) if "waste" in filtered.columns else 0

    k1, k2, k3, k4 = st.columns([1.2,1,1,1])
    k1.metric("Total historical sales", f"{total_history:,}")
    k2.metric("Avg daily sales", f"{avg_daily:,}")
    k3.metric("Avg daily waste", f"{avg_waste:,}")
    k4.metric("Selected filters", f"{city} / {supermarket} / {product}")

    st.markdown("<div class='card'>Recent sales (by date)</div>", unsafe_allow_html=True)
    recent = filtered.groupby("date")["sales"].sum().reset_index().tail(90)
    fig = px.area(recent, x="date", y="sales", title="Recent Sales (90 days)", color_discrete_sequence=["#0ea861"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dfffe4")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Page: Forecast --------------------
with tabs[1]:
    st.subheader("Forecast")
    st.markdown("Choose a date range to forecast. The app will show actuals (if available) and predicted sales.")
    col1, col2 = st.columns([1,1])
    with col1:
        start_date = st.date_input("Start date", value=filtered["date"].max().date() + timedelta(days=1))
    with col2:
        end_date = st.date_input("End date", value=filtered["date"].max().date() + timedelta(days=7))

    if st.button("Run Forecast"):
        if model is None:
            st.error("Model not loaded. Check models/xgb_model.joblib")
        else:
            forecast_df = forecast_range(filtered, start_date, end_date, model, model_features)
            actuals = filtered[["date","sales"]].copy()
            merged = pd.merge(actuals, forecast_df, on="date", how="outer").sort_values("date")
            merged["sales"] = merged["sales"].fillna(0)
            merged["predicted_sales"] = merged["predicted_sales"].fillna(0)

            total_pred = int(round(forecast_df["predicted_sales"].sum()))
            total_act = int(round(actuals[(actuals["date"]>=pd.to_datetime(start_date)) & (actuals["date"]<=pd.to_datetime(end_date))]["sales"].sum()))
            waste_est = int(round(total_pred * 0.15))
            colA, colB, colC = st.columns(3)
            colA.metric("Predicted total", f"{total_pred:,}")
            colB.metric("Actual total (range)", f"{total_act:,}")
            colC.metric("Est. avoidable waste (kg)", f"{waste_est:,}")

            # grouped bar chart
            chart_df = merged.melt(id_vars=["date"], value_vars=["sales","predicted_sales"], var_name="Type", value_name="Sales")
            fig2 = px.bar(chart_df, x="date", y="Sales", color="Type", barmode="group",
                          color_discrete_map={"sales":"#2E7D32","predicted_sales":"#FFB300"},
                          title=f"Actual vs Predicted ({start_date} → {end_date})")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dfffe4")
            st.plotly_chart(fig2, use_container_width=True)

            # download button
            st.download_button("Download forecast CSV", forecast_df.to_csv(index=False).encode('utf-8'), file_name="forecast.csv", mime="text/csv")
    else:
        st.info("Select a date range and press Run Forecast to generate predictions.")

# -------------------- Page: Insights --------------------
with tabs[2]:
    st.subheader("Insights")
    st.markdown("Model explainability and quick analytics")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### Feature impact (SHAP)")
        if st.button("Show SHAP image (if available)"):
            try:
                st.image(SHAP_PATH, caption="SHAP summary (feature impact)", use_column_width=True)
            except Exception:
                st.warning("No SHAP image found. Run SHAP script offline and upload outputs/shap_summary.png")
    with col2:
        st.markdown("### Quick stats")
        top_products = filtered.groupby("product")["sales"].sum().reset_index().sort_values("sales", ascending=False).head(5)
        st.table(top_products.set_index("product"))
        corr = filtered[["temp","sales"]].corr().iloc[0,1] if "temp" in filtered.columns else np.nan
        st.markdown(f"**Temp vs Sales correlation:** {corr:.2f}")

# -------------------- Page: Optimize --------------------
with tabs[3]:
    st.subheader("Order Optimization")
    st.markdown("Use the forecast to compute recommended order quantities.")
    pred_mean = st.number_input("Enter expected daily demand (or use predicted total)", min_value=0, value=100)
    safety = st.slider("Safety factor (higher → less stockout risk)", 5, 30, 10)
    rec_q = optimize_order(pred_mean, safety/100.0)
    st.metric("Recommended order quantity (units)", f"{rec_q:,}")
    st.markdown("This is a simple heuristic suggestion. For production, integrate real inventory costs & constraints.")

# -------------------- Map (real supermarket locations) --------------------
with tabs[4]:
    st.subheader("Germany Map • Real Supermarket Locations")
    st.markdown("Interactive map: select brand or show all. Pins show brand + city.")

    try:
        stores = pd.read_csv("supermarkets_real.csv")
    except FileNotFoundError:
        st.warning("supermarkets_real.csv not found. Upload the CSV to your repo under")
        st.stop()

    # Normalize column names if necessary
    stores.columns = [c.strip() for c in stores.columns]

    # brand dropdown
    brands = ["All"] + sorted(stores["brand"].dropna().unique().tolist())
    sel_brand = st.selectbox("Select brand to show", brands, index=0)

    map_df = stores.copy()
    if sel_brand != "All":
        map_df = map_df[map_df["brand"] == sel_brand]

    # small jitter for visibility
    map_df["lat"] = map_df["lat"].astype(float) + np.random.uniform(-0.001, 0.001, len(map_df))
    map_df["lon"] = map_df["lon"].astype(float) + np.random.uniform(-0.001, 0.001, len(map_df))

    # brand -> color mapping (RGB)
    brand_colors = {
        "Edeka": [5,150,255],
        "Rewe": [255,80,80],
        "Lidl": [255,210,0],
        "Aldi": [0,200,120],
        "Netto": [255,120,0],
        "Kaufland": [180,60,200]
    }
    # default color
    default_color = [200,200,200]

    # attach color column
    def get_color(brand):
        return brand_colors.get(brand, default_color)
    map_df["color"] = map_df["brand"].apply(get_color)

    # prepare pydeck layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_radius=500,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True
    )

    # initial view set to Germany center
    view = pdk.ViewState(latitude=51.1657, longitude=10.4515, zoom=5, pitch=0)

    deck = pdk.Deck(layers=[layer], initial_view_state=view, map_style="mapbox://styles/mapbox/dark-v10",
                    tooltip={"html": "<b>{brand}</b><br/>{city}<br/>Lat: {lat} Lon: {lon}", "style":{"color":"white"}})

    st.pydeck_chart(deck, use_container_width=True)

    # small table of visible stores
    with st.expander("Show store list"):
        st.dataframe(map_df[["brand","city","lat","lon"]].reset_index(drop=True))
