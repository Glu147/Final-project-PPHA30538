from __future__ import annotations

from pathlib import Path

import altair as alt
import geopandas as gpd
import pandas as pd
import streamlit as st
import os
import subprocess

DATA_PATH = "data/derived-data/state_year_panel.csv"

if not os.path.exists(DATA_PATH):
    subprocess.run(["python", "preprocessing.py"])

REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVED = REPO_ROOT / "data" / "derived-data"
RAW = REPO_ROOT / "data" / "raw-data"


st.set_page_config(
    page_title="News Sentiment & FDIC Banking Stress (State-Year)",
    layout="wide",
)

st.title("News Sentiment & FDIC Banking Stress (State-Year)")
st.caption(
    "Presentation demo: combine a year-level regulatory-news sentiment index with FDIC state-year fundamentals "
    "to predict stress and visualize geographic concentration."
)

panel_path = DERIVED / "state_year_panel.csv"
if not panel_path.exists():
    st.error(
        "Missing derived file `data/derived-data/state_year_panel.csv`. "
        "Run `python preprocessing.py` first to build the presentation dataset."
    )
    st.stop()

df = pd.read_csv(panel_path)
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

years = sorted([int(y) for y in df["YEAR"].dropna().unique().tolist()])
if not years:
    st.error("No valid YEAR values found in derived panel.")
    st.stop()

if "StressScore" in df.columns:
    years_with_stress = (
        df.loc[df["StressScore"].notna(), "YEAR"].dropna().astype(int).unique().tolist()
    )
else:
    years_with_stress = []
default_year = max(years_with_stress) if years_with_stress else max(years)

with st.sidebar:
    st.header("Controls")
    year = st.slider("Year", min_value=min(years), max_value=max(years), value=default_year, step=1)
    metric = st.selectbox(
        "Map metric",
        options=[
            "StressScore",
            "p_bad_year",
            "sev_hat",
            "DROA",
            "ROA",
            "sent_neg_share",
            "news_count",
        ],
    )
    st.caption("Tip: `StressScore = p_bad_year × sev_hat`")

df_y = df[df["YEAR"] == year].copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment vs ΔROA (state-year)")
    plot_df = df.dropna(subset=["sent_neg_share", "DROA", "YEAR"]).copy()
    plot_df = plot_df[plot_df["YEAR"].between(year - 5, year)]

    if plot_df.empty:
        st.info("No sentiment/ΔROA data available for plotting.")
    else:
        chart = (
            alt.Chart(plot_df)
            .mark_circle(size=50, opacity=0.35)
            .encode(
                x=alt.X("sent_neg_share:Q", title="Negative probability (yearly mean)"),
                y=alt.Y("DROA:Q", title="ΔROA"),
                color=alt.Color("bad_year:N", title="bad_year"),
                tooltip=["STNAME:N", "YEAR:Q", "sent_neg_share:Q", "DROA:Q", "ROA:Q", "StressScore:Q"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

with col2:
    st.subheader("Yearly sentiment index (filtered headlines)")
    sent_cols = ["YEAR", "sent_mean", "sent_neg_share", "news_count"]
    sent = df[sent_cols].drop_duplicates().dropna(subset=["YEAR"]).sort_values("YEAR")
    sent = sent[sent["YEAR"].between(year - 10, year)]

    if sent.empty:
        st.info("No sentiment index available.")
    else:
        sent_melt = sent.melt(id_vars=["YEAR"], value_vars=["sent_mean", "sent_neg_share"], var_name="metric", value_name="value")
        chart2 = (
            alt.Chart(sent_melt)
            .mark_line(point=True)
            .encode(
                x=alt.X("YEAR:Q", title="Year"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("metric:N", title="Sentiment metric"),
                tooltip=["YEAR:Q", "metric:N", "value:Q"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart2, use_container_width=True)

st.subheader("Geographic stress map")
shp_zip = RAW / "C" / "shapefile" / "cb_2024_us_all_20m" / "cb_2024_us_state_20m.zip"
if not shp_zip.exists():
    st.error("Missing state shapefile zip under `data/raw-data/C/shapefile/cb_2024_us_all_20m/`.")
    st.stop()

gdf = gpd.read_file(f"zip://{shp_zip}")
gdf["NAME"] = gdf["NAME"].astype(str).str.strip()

value_col = metric if metric in df_y.columns else "StressScore"
map_df = df_y[["STNAME", value_col]].copy()
map_df["STNAME"] = map_df["STNAME"].astype(str).str.strip()
gdf2 = gdf.merge(map_df, left_on="NAME", right_on="STNAME", how="left")

import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(12, 7))
gdf2.plot(
    column=value_col,
    ax=ax,
    legend=True,
    cmap="OrRd",
    missing_kwds={"color": "lightgrey", "label": "Missing"},
)
ax.set_axis_off()
ax.set_title(f"{value_col} — {year}")
st.pyplot(fig, clear_figure=True)

with st.expander("Show state-year table (selected year)"):
    show_cols = ["STNAME", "YEAR", "ROA", "DROA", "bad_year", "severity", "p_bad_year", "sev_hat", "StressScore", "sent_mean", "sent_neg_share", "news_count"]
    show_cols = [c for c in show_cols if c in df_y.columns]
    st.dataframe(df_y[show_cols].sort_values("StressScore", ascending=False), use_container_width=True)

