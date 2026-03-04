from __future__ import annotations

from pathlib import Path

import altair as alt
import geopandas as gpd
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVED = REPO_ROOT / "data" / "derived-data"
RAW = REPO_ROOT / "data" / "raw-data"
OUTPUTS = REPO_ROOT / "outputs"

STATE_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District Of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def load_panel_data() -> tuple[pd.DataFrame, str]:
    panel_path = DERIVED / "state_year_panel.csv"
    if panel_path.exists():
        return pd.read_csv(panel_path), "derived"

    summary_path = OUTPUTS / "state_stress_summary_2014_2020.csv"
    sensitivity_path = OUTPUTS / "state_sentiment_sensitivity_2014_2020.csv"
    if not summary_path.exists() or not sensitivity_path.exists():
        st.error(
            "Missing `data/derived-data/state_year_panel.csv`, and fallback files under `outputs/` are also unavailable."
        )
        st.stop()

    summary = pd.read_csv(summary_path)
    sensitivity = pd.read_csv(sensitivity_path)
    df = summary.merge(
        sensitivity[["STNAME", "n_years", "d_droa_if_neg_news_plus_10pp_l1"]],
        on="STNAME",
        how="left",
    )

    df["YEAR"] = 2020
    df["ROA"] = pd.to_numeric(df.get("avg_roa"), errors="coerce")
    df["DROA"] = pd.to_numeric(df.get("avg_droa"), errors="coerce")
    df["severity"] = pd.to_numeric(df.get("avg_severity"), errors="coerce")
    df["bad_year"] = (pd.to_numeric(df.get("bad_year_share"), errors="coerce") > 0.5).astype("Int64")
    df["p_bad_year"] = pd.to_numeric(df.get("bad_year_share"), errors="coerce")
    df["sev_hat"] = pd.to_numeric(df.get("avg_severity_bad_year"), errors="coerce").fillna(df["severity"])
    df["StressScore"] = df["p_bad_year"] * df["sev_hat"]
    df["sent_mean"] = pd.NA
    df["sent_neg_share"] = pd.NA
    df["news_count"] = pd.to_numeric(df.get("n_years"), errors="coerce")
    return df, "fallback"


st.set_page_config(
    page_title="News Sentiment & FDIC Banking Stress (State-Year)",
    layout="wide",
)

st.title("News Sentiment & FDIC Banking Stress (State-Year)")
st.caption(
    "Presentation demo: combine a year-level regulatory-news sentiment index with FDIC state-year fundamentals "
    "to predict stress and visualize geographic concentration."
)

df, data_mode = load_panel_data()
if data_mode == "fallback":
    st.warning(
        "Using fallback data from `outputs/*.csv` because `data/derived-data/state_year_panel.csv` is missing. "
        "For full functionality, run `python preprocessing.py` locally with raw data."
    )

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
default_year = int(max(years_with_stress)) if years_with_stress else int(max(years))
min_year = int(min(years))
max_year = int(max(years))

with st.sidebar:
    st.header("Controls")
    if min_year == max_year:
        year = min_year
        st.caption(f"Year fixed at {year} (only one year available in current dataset).")
    else:
        year = st.slider("Year", min_value=min_year, max_value=max_year, value=default_year, step=1)
    available_metrics = [
        c
        for c in ["StressScore", "p_bad_year", "sev_hat", "DROA", "ROA", "sent_neg_share", "news_count"]
        if c in df.columns
    ]
    metric = st.selectbox(
        "Map metric",
        options=available_metrics,
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
    sent_cols = [c for c in ["YEAR", "sent_mean", "sent_neg_share", "news_count"] if c in df.columns]
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
value_col = metric if metric in df_y.columns else "StressScore"
map_df = df_y[["STNAME", value_col]].copy().dropna(subset=[value_col])
map_df["STNAME"] = map_df["STNAME"].astype(str).str.strip()

if shp_zip.exists():
    gdf = gpd.read_file(f"zip://{shp_zip}")
    gdf["NAME"] = gdf["NAME"].astype(str).str.strip()
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
else:
    st.info(
        "State shapefile is missing. Showing a fallback ranking chart instead. "
        "Add `data/raw-data/C/shapefile/cb_2024_us_all_20m/cb_2024_us_state_20m.zip` for the full map."
    )
    map_df["abbr"] = map_df["STNAME"].map(STATE_TO_ABBR)
    fallback_plot = (
        alt.Chart(map_df.sort_values(value_col, ascending=False).head(20))
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:Q", title=value_col),
            y=alt.Y("abbr:N", sort="-x", title="State"),
            tooltip=["STNAME:N", "abbr:N", f"{value_col}:Q"],
        )
        .properties(height=420)
    )
    st.altair_chart(fallback_plot, use_container_width=True)

with st.expander("Show state-year table (selected year)"):
    show_cols = ["STNAME", "YEAR", "ROA", "DROA", "bad_year", "severity", "p_bad_year", "sev_hat", "StressScore", "sent_mean", "sent_neg_share", "news_count"]
    show_cols = [c for c in show_cols if c in df_y.columns]
    sort_col = "StressScore" if "StressScore" in df_y.columns else show_cols[0]
    st.dataframe(df_y[show_cols].sort_values(sort_col, ascending=False), use_container_width=True)

