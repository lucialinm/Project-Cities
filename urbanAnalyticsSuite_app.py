"""Urban Analytics Suite
---------------------------------
Streamlit application for spatial, predictive, and statistical analysis
of the short-term rental market (Airbnb) in Spanish cities.
"""

from __future__ import annotations
import os
from pathlib import Path
import contextlib
import orjson

# --- 1. IMPORTS --------------------------------------------------
import pandas as pd
import geopandas as gpd
import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import streamlit as st
from streamlit_folium import st_folium
import folium
import plotly.express as px
import matplotlib.pyplot as plt

with contextlib.suppress(ImportError):
    import shap

# --- 2. STREAMLIT CONFIGURATION ------------------------------------------
st.set_page_config(
    page_title="Urban Analytics Suite",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        background-color: white !important;
        color: #2c3e50;
    }

    h1, h2, h3 {
        color: #34495e;
        letter-spacing: .3px;
    }

    div.stButton > button:first-child {
    background: linear-gradient(90deg, #4a90e2, #357ABD) !important;
    border: none !important;
    color: white !important;
    font-weight: 600;
    padding: 0.5rem 1.2rem !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    }

    div.stButton > button:first-child * {
        background: transparent !important;
        color: inherit !important;
    }


    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 8px;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.05);
        background-color: white !important;
    }

    footer {
        text-align: center;
        opacity: .6;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# --- 3. CONSTANTS AND UTILITIES ------------------------------------------
DATA_DIR = Path(os.getenv("UAS_DATA_DIR", "."))  # Allows data path override

CITIES = {
    "Barcelona": "Barcelona",
    "Euskadi": "Euskadi",
    "Girona": "Girona",
    "Madrid": "Madrid",
    "Málaga": "Malaga",
    "Mallorca": "Mallorca",
    "Menorca": "Menorca",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
}

CITY_CENTERS = {
    "Barcelona": (41.3851, 2.1734),
    "Euskadi": (43.2630, -2.9350),  # Approx. center of Bilbao
    "Girona": (41.9818, 2.8245),
    "Madrid": (40.4168, -3.7038),
    "Málaga": (36.7213, -4.4216),
    "Mallorca": (39.6953, 3.0176),
    "Menorca": (39.9676, 4.0836),
    "Sevilla": (37.3891, -5.9845),
    "Valencia": (39.4699, -0.3763),
}

COLOR_SCALE = "PuBu"

# ------------------------------------------------------------------------
# 4. DATA LOADING AND CLEANING
# ------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_city_data(city_key: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Loads and cleans *listings* and *neighborhoods* data for a city.
    """

    city_folder = DATA_DIR / CITIES[city_key]
    listings_path = city_folder / "listings.csv"
    barrios_path = city_folder / "neighbourhoods.geojson"

    try:
        listings = pd.read_csv(listings_path)
        barrios = gpd.read_file(barrios_path)
    except Exception as exc:
        st.error(f"Error loading files for {city_key}: {exc}")
        return pd.DataFrame(), gpd.GeoDataFrame()

 # --- DATA CLEANING -------------------------------------------------
    listings = (
        listings
        .assign(
            price=lambda d: d["price"].replace(r"[\$,]", "", regex=True).astype(float)
        )
        .dropna(subset=["price", "neighbourhood"])
    )

    # Remove extreme outliers
    p01, p99 = listings["price"].quantile([0.01, 0.99])
    listings = listings.query("@p01 <= price <= @p99")

    return listings, barrios


# ------------------------------------------------------------------------
# 5. FEATURE ENGINEERING AND METRICS
# ------------------------------------------------------------------------

def aggregate_by_neighbourhood(listings: pd.DataFrame) -> pd.DataFrame:
    """Groups listings by neighborhood and computes key metrics."""
    agg = (
        listings.groupby("neighbourhood")
        .agg(num_airbnbs=("id", "count"), avg_price=("price", "mean"))
        .reset_index()
    )
    agg["tourist_pressure"] = np.sqrt(agg["num_airbnbs"] * agg["avg_price"])
    return agg


def merge_agg_with_geo(barrios: gpd.GeoDataFrame, agg: pd.DataFrame) -> gpd.GeoDataFrame:
    return barrios.merge(agg, how="left", on="neighbourhood")


# ------------------------------------------------------------------------
# 6. GEOGRAPHIC TAB -------------------------------------------------------
# ------------------------------------------------------------------------

def render_geo_tab(listings: pd.DataFrame, barrios: gpd.GeoDataFrame, city: str) -> None:
    """Renders the interactive map and metrics panel."""
    st.header(f"Interactive Map of {city}")

    agg = aggregate_by_neighbourhood(listings)
    barrios_agg = merge_agg_with_geo(barrios.copy(), agg)

    # -- Map setup
    start_coords = CITY_CENTERS.get(city, (listings["latitude"].median(), listings["longitude"].median()))
    m = folium.Map(location=start_coords, zoom_start=12, tiles="CartoDB positron")

    # Choropleth map
    choropleth = folium.Choropleth(
        geo_data=barrios_agg,
        data=barrios_agg,
        columns=["neighbourhood", "tourist_pressure"],
        key_on="feature.properties.neighbourhood",
        fill_color=COLOR_SCALE,
        fill_opacity=0.7,
        line_opacity=0.5,
        line_color="black",
        legend_name="Tourist Pressure Index",
    ).add_to(m)

    # Simple tooltip
    folium.GeoJsonTooltip(fields=["neighbourhood"], aliases=["Barrio:"], sticky=True).add_to(choropleth.geojson)

    # --- Layout ------------------------------------------------
    col_map, col_side = st.columns([3, 2], gap="large")

    with col_map:
        st_folium(m, width=700, height=500)

    with col_side:
        st.subheader("Explore by neighborhood")
        barrios_lista = sorted(barrios_agg["neighbourhood"].dropna().unique())

        selected_barrio = st.selectbox("Select a neighbourhood", barrios_lista)
        sel_row = agg[agg["neighbourhood"] == selected_barrio]

        if not sel_row.empty:
            st.metric("Active Airbnbs", int(sel_row["num_airbnbs"].iloc[0]))
            st.metric("Average Price", f"€{sel_row['avg_price'].iloc[0]:.2f}")
            st.metric("Tourist Pressure", round(sel_row["tourist_pressure"].iloc[0], 1))
        else:
            st.warning("No data available for this neighborhood.")


# ------------------------------------------------------------------------
# 7. PREDICTIVE TAB ------------------------------------------------------
# ------------------------------------------------------------------------

def train_price_model(listings: pd.DataFrame, city: str):
    """Trains an XGBoost model to predict Airbnb prices."""
    basic_cols = [
        "latitude", "longitude", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365",
    ]
    X = listings[basic_cols].copy()
    y = listings["price"].astype(float)

    # One hot encoding for categorical columns
    X = pd.concat([
        X,
        pd.get_dummies(listings["room_type"], prefix="room"),
        pd.get_dummies(listings["neighbourhood_group"], prefix="area"),
    ], axis=1)

     # How far is the listing from the city center?
    center = CITY_CENTERS.get(city, (X["latitude"].median(), X["longitude"].median()))
    X["distance_center"] = X.apply(lambda r: geodesic((r.latitude, r.longitude), center).km, axis=1)
    X["reviews_per_listing"] = listings["number_of_reviews"] / (listings["calculated_host_listings_count"] + 1)

   # Train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    return model, mae, mape, X.columns.tolist(), center

def render_predict_tab(listings: pd.DataFrame, city: str) -> None:
    st.header("Price Prediction Model")

    if st.button("Train / Update Model"):
        with st.spinner("Training model..."):
            model, mae, mape, feat_cols, center = train_price_model(listings, city)

        # Save everything for later use
        st.session_state.price_model = model
        st.session_state.mae = mae
        st.session_state.mape = mape
        st.session_state.model_features = feat_cols
        st.session_state.city_center = center

        st.success(f"Model trained · MAE €{mae:.0f} · MAPE {mape:.1f}%")

    if "price_model" not in st.session_state:
        st.info("Please train the model first to use the simulator.")
        return

    st.subheader("Price Simulator")
    col1, col2 = st.columns(2)

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=CITY_CENTERS[city][0], format="%f")
        lon = st.number_input("Longitude", value=CITY_CENTERS[city][1], format="%f")
        min_nights = st.slider("Minimum Nights", 1, 365, 2)
        num_reviews = st.slider("Number of Reviews", 0, 500, 10)
    with col2:
        reviews_month = st.slider("Reviews per Month", 0.0, 10.0, 1.0, step=0.1)
        availability = st.slider("Availability (days/year)", 0, 365, 180)
        host_listings = st.slider("Host's Number of Listings", 1, 50, 1)
        room_type = st.selectbox("Room Type", listings["room_type"].unique())
        area = st.selectbox("Area", listings["neighbourhood_group"].unique())
    
    # Dataframe with the input data
    input_df = pd.DataFrame([[lat, lon, min_nights, num_reviews, reviews_month,
                              host_listings, availability]],
                            columns=["latitude", "longitude", "minimum_nights",
                                     "number_of_reviews", "reviews_per_month",
                                     "calculated_host_listings_count", "availability_365"])
    # Add new features
    input_df["distance_center"] = geodesic((lat, lon), st.session_state.city_center).km
    input_df["reviews_per_listing"] = num_reviews / (host_listings + 1)

    # One‑hot manual
    for rt in listings["room_type"].unique():
        input_df[f"room_{rt}"] = int(rt == room_type)
    for ar in listings["neighbourhood_group"].unique():
        input_df[f"area_{ar}"] = int(ar == area)

    # Missing values
    for col in st.session_state.model_features:
        input_df[col] = input_df.get(col, 0)

    input_df = input_df[st.session_state.model_features]

    pred_price = st.session_state.price_model.predict(input_df)[0]

    st.markdown(
        f"<h2 style='text-align:center;'>Estimated Price: €{pred_price:.2f}</h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"Typical error margin ±€{st.session_state.mae:.0f}")

    # If SHAP installed, show feature contributions
    if "shap" in globals():
        explainer = shap.TreeExplainer(st.session_state.price_model)
        shap_values = explainer(input_df)
        st.subheader("Variable Contribution (SHAP)")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)
        
        valor_ref = st.session_state.price_model.predict(input_df)[0]
        
        azul_claro = '#4a90e2'
        azul_oscuro = '#1f2c56'

        for line in ax.lines:
            line.set_color(azul_oscuro)
        for i, patch in enumerate(ax.patches):
            contrib = shap_values.values[0][i]
        if contrib >= 0:
            patch.set_edgecolor(azul_claro)
            patch.set_facecolor(azul_claro)
        else:
            patch.set_edgecolor(azul_oscuro)
            patch.set_facecolor(azul_oscuro)

        st.pyplot(fig)
    else:
        st.info("Install SHAP (pip install shap) to see feature explanations.")


# ------------------------------------------------------------------------
# 8. STATISTICAL TAB -----------------------------------------------------
# ------------------------------------------------------------------------

def render_stats_tab(listings: pd.DataFrame, city: str) -> None:
    st.header(f"Statistical Analysis for {city}")

    barrio_stats = (
        listings.groupby("neighbourhood")
        .agg(
            num_airbnbs=("id", "count"),
            avg_price=("price", "mean"),
            std_price=("price", "std"),
            median_price=("price", "median")
        )
        .reset_index()
    )

    # No real data, simulate average income per neighborhood
    rng = np.random.default_rng(42)
    barrio_stats["income_mean"] = rng.normal(25000, 5000, len(barrio_stats))

    # --- 1. Relationship between of Airbnbs, price, and income ----------
    fig1 = px.scatter(
        barrio_stats,
        x="num_airbnbs",
        y="avg_price",
        size="income_mean",
        color="income_mean",
        hover_name="neighbourhood",
        labels={
            "num_airbnbs": "# of Airbnbs",
            "avg_price": "Average Price (€)",
            "income_mean": "Average Income (€)",
        },
        title="Listing Density vs. Price and Income",
    )
    st.plotly_chart(fig1, use_container_width=True)

   # --- 2. Correlation between income and prices -------------------------
    corr_val = barrio_stats["income_mean"].corr(barrio_stats["avg_price"])
    st.metric("Income–Price Correlation", f"{corr_val:.2f}")

   # --- 3. Outliers based on standard deviation --------------------------
    st.subheader("Neighborhoods with Price Outliers")
    outliers = barrio_stats[barrio_stats["std_price"] > barrio_stats["std_price"].mean() + barrio_stats["std_price"].std()]
    if not outliers.empty:
        st.write(outliers[["neighbourhood", "avg_price", "std_price"]].sort_values("std_price", ascending=False))
    else:
        st.info("No significant outliers detected.")

    # --- 4. Price distribution by room type -------------------------------
    st.subheader("Price Distribution by Room Type")
    fig2 = px.violin(
    listings,
    x="room_type",
    y="price",
    box=True,
    points="suspectedoutliers",
    color="room_type",
    title="Price Distribution by Room Type",
    labels={"price": "Price (€)", "room_type": "Room Type"},
    color_discrete_sequence=["#1f77b4", "#2ca02c", "#17becf"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- 5. Listing density heatmap ---------------------------------------
    st.subheader("Listing Density Map (Heatmap)")
    heat_df = listings.dropna(subset=["latitude", "longitude"])
    m = folium.Map(location=CITY_CENTERS[city], zoom_start=12, tiles="CartoDB positron")

    from folium.plugins import HeatMap
    HeatMap(data=heat_df[["latitude", "longitude"]].values, radius=9).add_to(m)
    st_folium(m, width=700, height=500)



# ------------------------------------------------------------------------
# 9. MAIN APP ------------------------------------------------------------
# ------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(layout="wide")
    st.image("imagen.jpg", use_container_width=True)
    #st.title("Urban Analytics Suite · Airbnbs")

   # Select city
    city = st.selectbox("Select a city", list(CITIES.keys()))
    #city = st.sidebar.selectbox("Select a city", list(CITIES.keys()))
    listings, barrios = load_city_data(city)
    if listings.empty or barrios.empty:
        st.warning("No data found for the selected city.")
        st.stop()

    # Main tabs
    tab_geo, tab_pred, tab_stats = st.tabs([
        "Geographic Environment", "Predictive Environment", "Statistics"
    ])

    # --- Geographic Environment ---
    with tab_geo:
        render_geo_tab(listings, barrios, city)

    # --- Predictive Environment ---
    with tab_pred:
        col_left, col_right = st.columns([3, 2], gap="medium")
        with col_left:
            fig_pred = render_predict_tab(listings, city)
        with col_right:
            if fig_pred is not None:
                st.plotly_chart(fig_pred, use_container_width=True)

    # --- Statistics ---
    with tab_stats:
        render_stats_tab(listings, city)


if __name__ == "__main__":
    main()

