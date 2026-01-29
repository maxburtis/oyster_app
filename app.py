import streamlit as st
import pandas as pd
import numpy as np
from pygam import LinearGAM, s
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🦪 Oyster Calculator")
st.markdown("**Sampling Protocol & App Guide:**")

with st.expander("Open README (Sampling protocol + how to run the app)", expanded=False):
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("README.md not found in the app directory.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Inputs")
CV = st.sidebar.number_input("Weight CV", value=0.35, min_value=0.05, max_value=1.0, step=0.05)
MARKET_WEIGHT = st.sidebar.number_input("Market size (g)", value=66.0, min_value=10.0)
MONTHS_AHEAD = st.sidebar.slider("Months to project", 1, 24, 12)

st.sidebar.header("Bagging / Splits")
DENSITY_G_PER_L = st.sidebar.number_input("Density (g/L)", value=1940.85, min_value=100.0)
SPLIT_FACTOR = st.sidebar.number_input("Split factor (x initial volume)", value=2.0, min_value=1.1, step=0.1)
INITIAL_VOLUME_L = st.sidebar.number_input("Initial bag volume (L)", value=3.0, min_value=0.1, step=0.5)


uploaded = st.sidebar.file_uploader("Upload bio CSV", type=["csv"])

# Offer README as a downloadable file
try:
    with open("README.md", "rb") as f:
        st.sidebar.download_button(
            "Download README",
            f,
            file_name="README.md",
            mime="text/markdown",
        )
except FileNotFoundError:
    pass

# -----------------------------
# Sample CSV (for first-time users)
# -----------------------------
sample_csv = pd.DataFrame({
    "Date": pd.to_datetime([
        "2026-04-15", "2026-04-29", "2026-05-13",
        "2026-04-15", "2026-04-29", "2026-05-13"
    ]),
    "Bag": [
        "SAMPLE_A", "SAMPLE_A", "SAMPLE_A",
        "SAMPLE_B", "SAMPLE_B", "SAMPLE_B"
    ],
    "Count": [200, 198, 198, 200, 199, 199],
    "weight_g": [4200, 4554, 4980, 4100, 4480, 4920],
})

st.sidebar.download_button(
    "Download sample CSV",
    sample_csv.to_csv(index=False),
    file_name="sample_oyster_sampling.csv",
    mime="text/csv",
)

# -----------------------------
# Helpers
# -----------------------------
def frac_ready(mean_g, cv, threshold):
    if mean_g <= 0:
        return np.nan
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean_g) - 0.5 * sigma**2
    z = (np.log(threshold) - mu) / sigma
    return 1 - norm.cdf(z)

def in_window(d, start_md, end_md):
    md = (d.month, d.day)
    if start_md <= end_md:
        return start_md <= md <= end_md
    else:
        return md >= start_md or md <= end_md

# -----------------------------
# Main logic
# -----------------------------
if uploaded:
    dd = pd.read_csv(uploaded, parse_dates=["Date"])
    dd = dd.sort_values(["Bag","Date"]).reset_index(drop=True)

    # Ensure Avg_Weight_g
    if "Avg_Weight_g" not in dd.columns:
        if {"weight_g","Count"}.issubset(dd.columns):
            dd["Avg_Weight_g"] = dd["weight_g"] / dd["Count"]
        else:
            st.error("CSV must contain Avg_Weight_g OR (weight_g + Count)")
            st.stop()

    # Age
    if "Age_days" not in dd.columns:
        dd["Age_days"] = (
            dd["Date"] - dd.groupby("Bag")["Date"].transform("min")
        ).dt.days.astype(float)

    dd = dd.dropna(subset=["Age_days","Avg_Weight_g"]).copy()
    dd["log_wt"] = np.log(dd["Avg_Weight_g"])

    # -----------------------------
    # Fit monotonic GAM
    # -----------------------------
    X = dd[["Age_days"]].values
    y = dd["log_wt"].values

    gam = LinearGAM(
        s(0, n_splines=8, constraints="monotonic_inc")
    ).fit(X, y)

    # -----------------------------
    # Build projection timeline (per-bag) + split schedule
    # -----------------------------
    dd["Date"] = pd.to_datetime(dd["Date"]).dt.normalize()

    # Observed bounds (global)
    obs_start = dd["Date"].min()
    obs_end   = dd["Date"].max()

    end_date = (obs_end + pd.Timedelta(days=30 * MONTHS_AHEAD)).normalize()

    # Use last observed Count per bag for biomass/volume projections
    if "Count" in dd.columns:
        last_counts = dd.groupby("Bag")["Count"].last()
    else:
        last_counts = None

    proj_bags = []
    for bag, g in dd.groupby("Bag"):
        g = g.sort_values("Date")
        b_start = g["Date"].min().normalize()
        b_end   = g["Date"].max().normalize()

        # per-bag observed month/day window
        start_md = (b_start.month, b_start.day)
        end_md   = (b_end.month, b_end.day)

        dates = pd.date_range(b_start, end_date, freq="D")
        growth = np.array([in_window(d, start_md, end_md) for d in dates], dtype=int)
        age_growing = np.cumsum(growth) - 1
        age_growing[age_growing < 0] = 0

        tmp = pd.DataFrame({
            "Bag": bag,
            "Date": dates,
            "GrowthDay": growth.astype(bool),
            "Age_growing_days": age_growing.astype(float),
        })

        tmp["pred_wt_g"] = np.exp(gam.predict(tmp[["Age_growing_days"]].values))
        tmp["pct_ready"] = tmp["pred_wt_g"].apply(lambda x: frac_ready(x, CV, MARKET_WEIGHT)) * 100

        if last_counts is not None:
            tmp["Count"] = float(last_counts.loc[bag])
            tmp["biomass_g"] = tmp["Count"] * tmp["pred_wt_g"]
            tmp["volume_L"] = tmp["biomass_g"] / float(DENSITY_G_PER_L)
        else:
            tmp["Count"] = np.nan
            tmp["biomass_g"] = np.nan
            tmp["volume_L"] = np.nan

        proj_bags.append(tmp)

    proj_all = pd.concat(proj_bags, ignore_index=True)

    # A global (mean-across-bags) series for plotting
    proj = (
        proj_all.groupby("Date", as_index=False)
                .agg(pred_wt_g=("pred_wt_g", "mean"),
                     pct_ready=("pct_ready", "mean"),
                     volume_L=("volume_L", "mean"))
    )

    # ---- Split schedule simulation (per bag) ----
    splits = []
    if last_counts is not None:
        for bag, g in proj_all.groupby("Bag"):
            g = g.sort_values("Date")
            split_n = 0

            for _, row in g.iterrows():
                crop_vol = row["volume_L"]
                if not np.isfinite(crop_vol):
                    continue

                # If each split doubles bag count, volume per bag halves
                vol_per_bag = crop_vol / (2 ** split_n)

                if vol_per_bag >= float(SPLIT_FACTOR) * float(INITIAL_VOLUME_L):
                    split_n += 1
                    splits.append({
                        "Bag": bag,
                        "Split #": split_n,
                        "Date": row["Date"],
                        "Vol_per_bag_L": vol_per_bag,
                    })

    splits_df = pd.DataFrame(splits)

    # -----------------------------
    # Plots
    # -----------------------------
    # --- stacked plots (top/middle/bottom) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot 1: mean weight over time
    ax1.scatter(dd["Date"], dd["Avg_Weight_g"], alpha=0.7, label="Observed (all bags)")
    ax1.plot(proj["Date"], proj["pred_wt_g"], linewidth=2, label="Predicted (mean across bags)")
    ax1.set_ylabel("Mean weight (g)")
    ax1.set_title("Mean oyster weight over time")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Plot 2: % at/above market
    ax2.plot(proj["Date"], proj["pct_ready"], linewidth=2, label="% at/above market (mean across bags)")
    ax2.axhline(50, linestyle="--", alpha=0.5)
    ax2.axhline(90, linestyle="--", alpha=0.5)
    ax2.set_ylabel("% at market")
    ax2.set_ylim(0, 100)
    ax2.set_title("% of crop ready for market")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Plot 3: projected volume + split threshold + split events
    if "volume_L" in proj.columns and proj["volume_L"].notna().any():
        ax3.plot(proj["Date"], proj["volume_L"], linewidth=2, label="Projected crop volume (L, mean across bags)")
        ax3.axhline(float(SPLIT_FACTOR) * float(INITIAL_VOLUME_L), linestyle="--", alpha=0.7,
                    label="Split threshold (x initial volume)")
        if not splits_df.empty:
            ax3.scatter(splits_df["Date"], splits_df["Vol_per_bag_L"], alpha=0.9, label="Split events (per-bag volume)")
        ax3.set_ylabel("Volume (L)")
        ax3.set_title("Bag volume & split schedule")
        ax3.grid(alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.01, 0.5, "No Count column provided; volume/split schedule disabled.", transform=ax3.transAxes)
        ax3.set_title("Bag volume & split schedule")

    ax3.set_xlabel("Date")

    st.pyplot(fig)


    # -----------------------------
    # Download
    # -----------------------------
    st.download_button(
        "Download forecast CSV",
        proj_all.to_csv(index=False),
        "oyster_market_forecast.csv",
        "text/csv"
    )


else:
    st.info("Upload a bio CSV to begin.")

# -----------------------------
# Footer credits
# -----------------------------
st.markdown(
    """
---
**Credits:**  
This work was funded by a **SARE (Sustainable Agriculture Research & Education)** grant.
"""
)
