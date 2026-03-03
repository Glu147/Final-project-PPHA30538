from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "data" / "raw-data"
DERIVED_DIR = REPO_ROOT / "data" / "derived-data"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_fdic_state_year() -> pd.DataFrame:
    """
    FDIC state-year aggregates.

    Expected columns (from Summary_data_states.csv):
    ASSET, NETINC, NIM, DEP, BANKS, STNAME, YEAR, ...
    """
    fdic_path = RAW_DIR / "B" / "Summary_data_states.csv"
    df = pd.read_csv(fdic_path)

    # Normalize types
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["STNAME"] = df["STNAME"].astype(str).str.strip()

    for c in ["ASSET", "NETINC", "DEP", "BANKS", "NIM"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["YEAR", "STNAME", "ASSET", "NETINC"]).copy()

    # Derived fundamentals
    df["ROA"] = df["NETINC"] / df["ASSET"]
    df = df.sort_values(["STNAME", "YEAR"])
    df["ROA_L1"] = df.groupby("STNAME")["ROA"].shift(1)
    df["DROA"] = df["ROA"] - df["ROA_L1"]
    df["DROA_L1"] = df.groupby("STNAME")["DROA"].shift(1)

    for c in ["ASSET", "DEP", "BANKS", "NIM"]:
        if c in df.columns:
            df[f"{c}_L1"] = df.groupby("STNAME")[c].shift(1)
            denom = df[f"{c}_L1"].where(df[f"{c}_L1"].abs() > 0)
            df[f"D{c}"] = (df[c] - df[f"{c}_L1"]) / denom

    # Guardrails: remove infinities from early years / zero denominators
    df = df.replace([np.inf, -np.inf], np.nan)

    # Labels
    q20 = df.groupby("STNAME")["DROA"].transform(lambda s: s.quantile(0.2))
    df["bad_year"] = (df["DROA"] < q20).astype(int)
    df["severity"] = (-np.minimum(df["DROA"], 0.0)).astype(float)

    return df


def train_sentiment_model() -> tuple[TfidfVectorizer, LogisticRegression]:
    """
    Train a lightweight sentiment classifier from the bundled labeled text dataset.

    Uses data/raw-data/A/all-data.csv which appears to be in format:
      <label>,<text>
    and encoded as cp1252.
    """
    train_path = RAW_DIR / "A" / "all-data.csv"
    train_df = pd.read_csv(
        train_path,
        encoding="cp1252",
        header=None,
        names=["label", "text"],
    )
    train_df["label"] = train_df["label"].astype(str).str.strip().str.lower()
    train_df["text"] = train_df["text"].astype(str)
    train_df = train_df.dropna(subset=["label", "text"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )
    X = vectorizer.fit_transform(train_df["text"])
    y = train_df["label"]

    # Use single-process to stay compatible with sandboxed environments.
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=1,
    )
    clf.fit(X, y)
    return vectorizer, clf


def build_news_sentiment_yearly(
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    *,
    keywords: list[str],
    min_year: int,
    max_year: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Build a year-level sentiment index from raw headlines.

    We filter headlines by regulatory/policy keywords first, then score.
    """
    news_path = RAW_DIR / "A" / "raw_partner_headlines.csv"
    usecols = ["headline", "date"]

    kw_re = re.compile("|".join([re.escape(k) for k in keywords]), flags=re.IGNORECASE)

    rows = []
    for chunk in pd.read_csv(news_path, usecols=usecols, chunksize=chunksize):
        chunk["headline"] = chunk["headline"].astype(str)
        chunk = chunk[chunk["headline"].str.contains(kw_re, na=False)].copy()
        if chunk.empty:
            continue

        dt = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["YEAR"] = dt.dt.year
        chunk = chunk.dropna(subset=["YEAR"])
        chunk["YEAR"] = chunk["YEAR"].astype(int)
        chunk = chunk[(chunk["YEAR"] >= min_year) & (chunk["YEAR"] <= max_year)]
        if chunk.empty:
            continue

        Xh = vectorizer.transform(chunk["headline"])
        proba = clf.predict_proba(Xh)
        classes = list(clf.classes_)

        def _col(name: str) -> int | None:
            name = name.lower()
            return classes.index(name) if name in classes else None

        neg_i = _col("negative")
        pos_i = _col("positive")

        neg_p = proba[:, neg_i] if neg_i is not None else np.zeros(proba.shape[0])
        pos_p = proba[:, pos_i] if pos_i is not None else np.zeros(proba.shape[0])
        score = pos_p - neg_p

        out = pd.DataFrame(
            {
                "YEAR": chunk["YEAR"].to_numpy(),
                "sent_score": score,
                "sent_neg_p": neg_p,
            }
        )
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["YEAR", "sent_mean", "sent_neg_share", "news_count"])

    all_scored = pd.concat(rows, ignore_index=True)
    yearly = (
        all_scored.groupby("YEAR", as_index=False)
        .agg(
            sent_mean=("sent_score", "mean"),
            sent_neg_share=("sent_neg_p", "mean"),
            news_count=("sent_score", "size"),
        )
        .sort_values("YEAR")
    )
    return yearly


def fit_and_score_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Fit simple baseline models and generate StressScore for visualization.
    """
    df = panel.copy()

    feature_cols = [
        "ROA_L1",
        "DROA_L1",
        "DASSET",
        "DDEP",
        "DBANKS",
        "NIM",
        "sent_mean",
        "sent_neg_share",
        "news_count",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    model_df = df.dropna(subset=feature_cols + ["bad_year", "severity"]).copy()
    if model_df.empty:
        df["p_bad_year"] = np.nan
        df["sev_hat"] = np.nan
        df["StressScore"] = np.nan
        return df

    X = model_df[feature_cols].to_numpy()
    y_cls = model_df["bad_year"].to_numpy()
    y_reg = model_df["severity"].to_numpy()

    cls_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1)),
        ]
    )
    cls_pipe.fit(X, y_cls)
    p = cls_pipe.predict_proba(X)[:, 1]

    reg_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    if (y_cls == 1).any():
        reg_pipe.fit(X[y_cls == 1], y_reg[y_cls == 1])
    else:
        reg_pipe.fit(X, y_reg)
    sev_hat = np.maximum(reg_pipe.predict(X), 0.0)

    model_df["p_bad_year"] = p
    model_df["sev_hat"] = sev_hat
    model_df["StressScore"] = model_df["p_bad_year"] * model_df["sev_hat"]

    df = df.merge(
        model_df[["STNAME", "YEAR", "p_bad_year", "sev_hat", "StressScore"]],
        on=["STNAME", "YEAR"],
        how="left",
    )
    return df


def make_static_plots(panel: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    _safe_mkdir(OUTPUTS_DIR)

    plot_df = panel.dropna(subset=["sent_neg_share", "DROA"]).copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(plot_df["sent_neg_share"], plot_df["DROA"], alpha=0.25, s=12)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Regulatory-news negative probability (yearly mean)")
        ax.set_ylabel("ΔROA (state-year)")
        ax.set_title("News negativity vs next-period ROA change (state-year)")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "static_scatter_sent_vs_droa.png", dpi=200)
        plt.close(fig)

    # Map plot uses geopandas; keep it robust for presentation
    try:
        import geopandas as gpd

        # Choose a year where the mapped metric is actually available.
        year_candidates = panel.dropna(subset=["StressScore", "YEAR"])["YEAR"].astype(int)
        if year_candidates.empty:
            return
        map_year = int(year_candidates.max())
        map_df = panel[(panel["YEAR"] == map_year)].dropna(subset=["StressScore"]).copy()
        if not map_df.empty:
            shp_zip = RAW_DIR / "C" / "shapefile" / "cb_2024_us_all_20m" / "cb_2024_us_state_20m.zip"
            gdf = gpd.read_file(f"zip://{shp_zip}")
            gdf["NAME"] = gdf["NAME"].astype(str).str.strip()
            gdf = gdf.merge(map_df[["STNAME", "StressScore"]], left_on="NAME", right_on="STNAME", how="left")

            fig, ax = plt.subplots(figsize=(12, 7))
            gdf.plot(
                column="StressScore",
                ax=ax,
                legend=True,
                cmap="OrRd",
                missing_kwds={"color": "lightgrey", "label": "Missing"},
            )
            ax.set_axis_off()
            ax.set_title(f"State StressScore (News+FDIC) — {map_year}")
            fig.tight_layout()
            fig.savefig(OUTPUTS_DIR / "static_map_stressscore.png", dpi=200)
            plt.close(fig)
    except Exception:
        # Map is optional for early presentation; Streamlit map can be used instead.
        return


def main() -> None:
    _safe_mkdir(RAW_DIR)
    _safe_mkdir(DERIVED_DIR)
    _safe_mkdir(OUTPUTS_DIR)

    print("Loading FDIC state-year data.")
    fdic = load_fdic_state_year()

    # Limit news to the years we can evaluate for the presentation
    min_year = int(fdic["YEAR"].max() - 10) if not fdic.empty else 2010
    max_year = int(fdic["YEAR"].max()) if not fdic.empty else 2024

    print("Training lightweight sentiment model.")
    vec, clf = train_sentiment_model()

    print("Scoring regulatory/policy news headlines by year.")
    keywords = ["fdic", "sec", "fed", "regulation", "bank run", "regional bank", "stress test"]
    yearly_sent = build_news_sentiment_yearly(
        vec,
        clf,
        keywords=keywords,
        min_year=min_year,
        max_year=max_year,
    )

    panel = fdic.merge(yearly_sent, on="YEAR", how="left")
    panel = fit_and_score_panel(panel)

    out_parquet = DERIVED_DIR / "state_year_panel.parquet"
    out_csv = DERIVED_DIR / "state_year_panel.csv"
    try:
        panel.to_parquet(out_parquet, index=False)
    except ImportError:
        out_parquet = None
    panel.to_csv(out_csv, index=False)

    make_static_plots(panel)

    print("Wrote derived panel:")
    if out_parquet is not None:
        print(f"- {out_parquet}")
    print(f"- {out_csv}")
    print("Wrote static plots to outputs/:")
    print("- static_scatter_sent_vs_droa.png")
    print("- static_map_stressscore.png (if available)")


if __name__ == "__main__":
    main()

